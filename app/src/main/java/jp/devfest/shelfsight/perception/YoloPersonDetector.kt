package jp.devfest.shelfsight.perception

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import jp.devfest.shelfsight.assist.ProductRecognizer
import jp.devfest.shelfsight.model.NormalizedBox
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** COCO で学習した YOLO の person クラスで人を見つける。モデルは tools/export_person.py で作る。 */
class YoloPersonDetector(context: Context) : PersonDetector {
    private val appContext = context.applicationContext
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    /** 読み込みに時間がかかるので、最初の検出時にワーカースレッドで作る。 */
    private fun session(): OrtSession = session ?: OrtSession.SessionOptions().use { o ->
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        o.setIntraOpNumThreads(2)   // 商品の認識と CPU を分け合う
        env.createSession(appContext.assets.open(MODEL_ASSET).use { it.readBytes() }, o)
    }.also { session = it }

    override fun detect(image: Bitmap): List<NormalizedBox> {
        val s = session()
        val scale = SIZE.toFloat() / maxOf(image.width, image.height)
        val width = Math.round(image.width * scale).coerceAtLeast(1)
        val height = Math.round(image.height * scale).coerceAtLeast(1)
        val letter = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        Canvas(letter).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(image, null, Rect(0, 0, width, height), Paint(Paint.FILTER_BITMAP_FLAG))
        }
        val pixels = IntArray(SIZE * SIZE)
        letter.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        letter.recycle()
        val input = ByteBuffer.allocateDirect(3 * SIZE * SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (channel in intArrayOf(16, 8, 0)) for (p in pixels) input.put(((p shr channel) and 0xff) / 255f)
        input.rewind()
        val output = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                (r[0].value as Array<Array<FloatArray>>)[0]
            }
        }
        return decode(output, width, height)
    }

    override fun close() {
        session?.close()
        session = null
    }

    internal companion object {
        private const val MODEL_ASSET = "person/yolo26n_416.onnx"
        private const val SIZE = 416
        private const val MIN_SCORE = 0.4f
        private const val NMS_IOU = 0.5f

        /** YOLO の出力（[cx, cy, w, h, person, ...] の列、ピクセル）を枠にする。[width], [height] は余白を除いた画像の大きさ。 */
        fun decode(output: Array<FloatArray>, width: Int, height: Int): List<NormalizedBox> {
            val candidates = output[0].indices.filter { output[PERSON_SCORE][it] >= MIN_SCORE }.map { i ->
                floatArrayOf(
                    output[0][i] - output[2][i] / 2, output[1][i] - output[3][i] / 2,
                    output[0][i] + output[2][i] / 2, output[1][i] + output[3][i] / 2,
                ) to output[PERSON_SCORE][i]
            }
            return ProductRecognizer.nms(candidates, NMS_IOU).map { (b, _) ->
                NormalizedBox(
                    left = (b[0] / width).coerceIn(0f, 1f),
                    top = (b[1] / height).coerceIn(0f, 1f),
                    right = (b[2] / width).coerceIn(0f, 1f),
                    bottom = (b[3] / height).coerceIn(0f, 1f),
                )
            }
        }

        private const val PERSON_SCORE = 4   // 枠の 4 列のあとに、クラス 0 (person) の確からしさが続く
    }
}
