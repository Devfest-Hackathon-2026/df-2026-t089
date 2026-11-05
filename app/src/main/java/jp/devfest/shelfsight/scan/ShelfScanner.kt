package jp.devfest.shelfsight.scan

import android.os.SystemClock
import android.util.Log
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import jp.devfest.shelfsight.ar.CapturedFrame
import jp.devfest.shelfsight.ar.JpegEncoder
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.model.WorldDetection
import jp.devfest.shelfsight.vision.ShelfClassifier
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * 撮影 → Gemma 判定 → 3D 位置の復元までをつなぐ。
 *
 * 推論は常に 1 件だけ実行する。結果は [results] に積み、ARCore の Anchor を扱える GL スレッドが取り出す。
 */
class ShelfScanner(
    private val classifier: ShelfClassifier,
    private val scope: CoroutineScope,
    private val onFailure: (Exception) -> Unit,
    private val onScanFinished: () -> Unit,
) {
    val results = ConcurrentLinkedQueue<WorldDetection>()

    @Volatile var enabled = false

    private val busy = AtomicBoolean(false)
    @Volatile private var lastScanned: Viewpoint? = null
    @Volatile private var retryAtMillis = 0L

    private data class Viewpoint(val position: Vec3, val forward: Vec3)

    /** GL スレッドから毎フレーム呼ばれる。画像のコピーを避けるため、先に撮影の要否を判定する。 */
    fun wantsCapture(cameraToWorld: FloatArray): Boolean {
        if (!enabled || busy.get() || SystemClock.elapsedRealtime() < retryAtMillis) return false
        val last = lastScanned ?: return true
        val now = viewpointOf(cameraToWorld)
        return ScanGate.hasMovedEnough(last.position, last.forward, now.position, now.forward)
    }

    fun submit(frame: CapturedFrame) {
        if (!busy.compareAndSet(false, true)) return
        scope.launch(Dispatchers.Default) {
            try {
                results += scan(frame)
                lastScanned = viewpointOf(frame.geometry.cameraToWorld)
                onScanFinished()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Scan failed", e)
                retryAtMillis = SystemClock.elapsedRealtime() + RETRY_DELAY_MS
                onFailure(e)
            } finally {
                busy.set(false)
            }
        }
    }

    private suspend fun scan(frame: CapturedFrame): List<WorldDetection> {
        val jpeg = JpegEncoder.toUprightJpeg(frame)
        return classifier.classify(jpeg).map { detection ->
            val result = Unprojector.unproject(detection.box, frame.geometry)
            WorldDetection(detection.category, result.position, result.hasMeasuredDepth)
        }
    }

    private fun viewpointOf(m: FloatArray) = Viewpoint(
        position = Vec3(m[12], m[13], m[14]),
        forward = Vec3(-m[8], -m[9], -m[10]),
    )

    private companion object {
        const val TAG = "ShelfScanner"
        const val RETRY_DELAY_MS = 5_000L
    }
}
