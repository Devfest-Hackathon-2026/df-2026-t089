package jp.devfest.shelfsight.perception

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import jp.devfest.shelfsight.model.NormalizedBox

/** 正立画像から人の範囲を見つける。 */
interface PersonDetector : AutoCloseable {
    fun detect(image: Bitmap): List<NormalizedBox>
}

/** MediaPipe Object Detector による実装。モデルはアプリに同梱し、端末内で実行する。 */
class MediaPipePersonDetector(context: Context) : PersonDetector {
    private val appContext = context.applicationContext
    private var detector: ObjectDetector? = null

    /** 読み込みに時間がかかるので、最初の検出時にワーカースレッドで作る。 */
    private fun detector(): ObjectDetector = detector ?: ObjectDetector.createFromOptions(
        appContext,
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_ASSET).build())
            .setRunningMode(RunningMode.IMAGE)
            .setScoreThreshold(MIN_SCORE)
            .setCategoryAllowlist(listOf(PERSON_CATEGORY))
            .build(),
    ).also { detector = it }

    override fun detect(image: Bitmap): List<NormalizedBox> {
        val result = detector().detect(BitmapImageBuilder(image).build())
        val width = image.width.toFloat()
        val height = image.height.toFloat()
        return result.detections().map { detection ->
            val box = detection.boundingBox()
            NormalizedBox(
                left = (box.left / width).coerceIn(0f, 1f),
                top = (box.top / height).coerceIn(0f, 1f),
                right = (box.right / width).coerceIn(0f, 1f),
                bottom = (box.bottom / height).coerceIn(0f, 1f),
            )
        }
    }

    override fun close() {
        detector?.close()
        detector = null
    }

    private companion object {
        const val MODEL_ASSET = "efficientdet_lite0.tflite"
        const val PERSON_CATEGORY = "person"
        const val MIN_SCORE = 0.5f
    }
}
