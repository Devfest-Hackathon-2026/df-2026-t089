package jp.devfest.shelfsight.walking

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.os.SystemClock
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

/** 1 枚の画像を判別した結果。[columns] は左から右の列ごとのカテゴリ。 */
class CoarseResult(
    val columns: List<String>,
    /** 列ごとの、選んだカテゴリの確率。平滑化で選び直した列は低くなる。 */
    val confidence: FloatArray,
    val preprocessMillis: Long,
    val inferenceMillis: Long,
)

/**
 * 歩行中に、カメラ画像の列ごとの大まかなカテゴリを端末内で判別する。画像は外に出ない。
 *
 * MobileCLIP の画像エンコーダ（ONNX）で列ごとに特徴を出し、[CoarseHead] でカテゴリにする。
 * 重いので GL スレッドでは呼ばない。
 */
class CoarseClassifier(modelFile: File, val head: CoarseHead, threads: Int = DEFAULT_THREADS) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = OrtSession.SessionOptions().use { options ->
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        // Pixel 8a では 4 本が最も速かった（XNNPACK を使うと 2 倍以上遅く、6 本でも遅くなった）。
        options.setIntraOpNumThreads(threads)
        env.createSession(modelFile.path, options)
    }
    private val inputName = session.inputNames.first()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val square = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)

    /** [image] は正立させた画像。 */
    fun classify(image: Bitmap): CoarseResult {
        val start = SystemClock.elapsedRealtime()
        val n = head.columns
        val plane = INPUT_SIZE * INPUT_SIZE
        val input = FloatBuffer.allocate(n * 3 * plane)
        val canvas = Canvas(square)
        val target = Rect(0, 0, INPUT_SIZE, INPUT_SIZE)
        for (c in 0 until n) {
            val crop = head.cropFor(c, image.width, image.height)
            canvas.drawBitmap(image, Rect(crop.left, crop.top, crop.right, crop.bottom), target, paint)
            square.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
            val base = c * 3 * plane
            // PC と同じく 0〜1 にするだけで、平均や分散では割らない。
            for (i in 0 until plane) {
                val p = pixels[i]
                input.put(base + i, (p shr 16 and 0xFF) / 255f)
                input.put(base + plane + i, (p shr 8 and 0xFF) / 255f)
                input.put(base + 2 * plane + i, (p and 0xFF) / 255f)
            }
        }
        val prepared = SystemClock.elapsedRealtime()

        val shape = longArrayOf(n.toLong(), 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val embeddings = OnnxTensor.createTensor(env, input, shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { output ->
                @Suppress("UNCHECKED_CAST")
                (output[0].value as Array<FloatArray>)
            }
        }
        val done = SystemClock.elapsedRealtime()

        val probs = embeddings.map { head.probabilities(it) }
        val path = head.smooth(probs)
        return CoarseResult(
            columns = path.map { head.classes[it] },
            confidence = FloatArray(n) { probs[it][path[it]] },
            preprocessMillis = prepared - start,
            inferenceMillis = done - prepared,
        )
    }

    override fun close() {
        session.close()
        square.recycle()
    }

    companion object {
        private const val INPUT_SIZE = 256
        private const val DEFAULT_THREADS = 4
        private const val ASSET_DIR = "coarse"

        /**
         * アセットのモデルを読み込む。モデルが入っていないビルド（大きいのでリポジトリに含めない）では null を返す。
         * ONNX Runtime はファイルから読む方がメモリを使わないので、初回にアプリの領域へ写す。
         */
        fun fromAssets(context: Context, name: String): CoarseClassifier? {
            val assets = context.assets
            val files = assets.list(ASSET_DIR).orEmpty()
            if ("$name.onnx" !in files || "$name.head.json" !in files) return null
            val head = CoarseHead.parse(assets.open("$ASSET_DIR/$name.head.json").bufferedReader().readText())
            val model = File(context.noBackupFilesDir, "$name.onnx")
            val size = assets.openFd("$ASSET_DIR/$name.onnx").use { it.length }
            if (!model.exists() || model.length() != size) {
                assets.open("$ASSET_DIR/$name.onnx").use { input -> model.outputStream().use { input.copyTo(it) } }
            }
            return CoarseClassifier(model, head)
        }
    }
}
