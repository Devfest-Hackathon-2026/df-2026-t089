package jp.devfest.shelfsight.perception

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import jp.devfest.shelfsight.model.Vec3

/** 正立画像から手の 21 点を見つける。 */
internal interface HandDetector : AutoCloseable {
    /** @param timestampMillis 呼び出しごとに増え続ける時刻 */
    fun detect(image: Bitmap, timestampMillis: Long): List<HandLandmarks>
}

/**
 * MediaPipe Hand Landmarker による実装。モデルはアプリに同梱し、端末内で実行する。
 * 手のひらの検出 → 21 点の 2 段で、一度見つけた手は次のフレームから手のひらの検出なしで追いかける（VIDEO モード）。
 *
 * @param minDetection 手のひらの検出の下限。横から物を持った手は手のひらが見えず、低く出る
 * @param minKeep 追いかけを続ける下限（21 点のモデルが「手がある」とする確からしさ）。物で指が隠れると下がる
 */
internal class MediaPipeHandDetector(
    context: Context,
    private val minDetection: Float = MIN_DETECTION_CONFIDENCE,
    private val minKeep: Float = MIN_KEEP_CONFIDENCE,
) : HandDetector {
    private val appContext = context.applicationContext
    private var landmarker: HandLandmarker? = null

    /** 読み込みに時間がかかるので、最初の検出時にワーカースレッドで作る。 */
    private fun landmarker(): HandLandmarker = landmarker ?: HandLandmarker.createFromOptions(
        appContext,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_ASSET).build())
            // VIDEO は前のフレームの手を追跡するので、パーから握っても見失いにくい。
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(MAX_HANDS)
            .setMinHandDetectionConfidence(minDetection)
            .setMinHandPresenceConfidence(minKeep)
            .setMinTrackingConfidence(minKeep)
            .build(),
    ).also { landmarker = it }

    override fun detect(image: Bitmap, timestampMillis: Long): List<HandLandmarks> {
        val result = landmarker().detectForVideo(BitmapImageBuilder(image).build(), timestampMillis)
        return result.landmarks().indices.map { i ->
            HandLandmarks(
                imagePoints = result.landmarks()[i].map { Vec3(it.x(), it.y(), it.z()) },
                worldPoints = result.worldLandmarks()[i].map { Vec3(it.x(), it.y(), it.z()) },
                // MediaPipe は鏡像の入力を前提に左右を返す。背面カメラの映像は鏡像ではないので入れ替える。
                isRightHand = result.handedness()[i].first().categoryName() == LEFT_LABEL,
            )
        }
    }

    override fun close() {
        landmarker?.close()
        landmarker = null
    }

    private companion object {
        const val MODEL_ASSET = "hand_landmarker.task"
        const val MAX_HANDS = 2
        const val LEFT_LABEL = "Left"
        // 握った手や指さしは、開いた手より信頼度が低く出やすい。
        const val MIN_DETECTION_CONFIDENCE = 0.35f

        // 一度見つけた手は、指が重なって見えにくくなっても追跡を続ける。
        const val MIN_KEEP_CONFIDENCE = 0.2f
    }
}
