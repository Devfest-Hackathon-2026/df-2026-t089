package jp.devfest.shelfsight.labeling

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import jp.devfest.shelfsight.ar.CapturedFrame
import jp.devfest.shelfsight.ar.FrameCapture
import jp.devfest.shelfsight.ar.JpegEncoder
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.cloud.CloudException
import jp.devfest.shelfsight.cloud.CloudFailure
import jp.devfest.shelfsight.cloud.CloudResponseParser
import jp.devfest.shelfsight.cloud.CloudTasks
import jp.devfest.shelfsight.cloud.CloudVisionClient
import jp.devfest.shelfsight.cloud.PlacedShelfColumn
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.ocr.TextOcr
import jp.devfest.shelfsight.ocr.ProductReading
import jp.devfest.shelfsight.ocr.ProductResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class LabelState(
    val cloudAvailable: Boolean = false,
    val cloudEnabled: Boolean = true,
    val status: String = "",
    val columns: List<PlacedShelfColumn> = emptyList(),
    val product: ProductResult? = null,
)

/**
 * 立ち止まったときの棚の判別と、手に取った商品の読み取りを行う。
 *
 * - 棚：棚の前で立ち止まったときだけ、画像をクラウドに送って列ごとに判別する
 * - 商品：OCR で読み取る。Cloud Vision の API キーがあればクラウドで読み、なければ（または通信できなければ）端末内で読む
 *
 * 撮影だけを GL スレッドで行う。通信で OCR が待たされないよう、処理のスレッドは分ける。
 * 商品をカメラに近づけると視界がふさがれて AR の追跡が切れることが多いので、
 * 商品の読み取りは追跡が切れていても動くようにしている。
 *
 * @param client null なら API キーが未設定で、棚の判別だけを行わない
 * @param debugDir 判別の履歴と最後に使った画像を残す場所。null なら残さない
 */
class StopLabeler internal constructor(
    private val client: CloudVisionClient?,
    private val tasks: CloudTasks?,
    private val ocr: TextOcr,
    private val debugDir: File? = null,
    private val uprightRotationDegrees: () -> Int,
) {
    private val _state = MutableStateFlow(
        LabelState(cloudAvailable = client != null, status = if (client == null) "棚の判別：API キーが未設定です" else ""),
    )
    val state: StateFlow<LabelState> = _state.asStateFlow()

    private val cloudWorker = Executors.newSingleThreadExecutor { Thread(it, "cloud-labeler") }
    private val ocrWorker = Executors.newSingleThreadExecutor { Thread(it, "product-ocr") }
    private val cloudBusy = AtomicBoolean(false)
    private val ocrBusy = AtomicBoolean(false)
    private val parser = tasks?.let { CloudResponseParser(it.categories) }
    private val budget = RequestBudget()
    private val held = HeldProductDetector() // 手のスレッドのみが触る

    // 以下は GL スレッドのみが触る。
    private val dwell = DwellDetector()
    private val scene = SceneChangeDetector()
    private var dwellAt = NO_DWELL
    private var lastSignatureAt = Double.NEGATIVE_INFINITY

    @Volatile private var cloudEnabled = true
    @Volatile private var retryAfterSeconds = 0.0
    @Volatile private var dwellRetryPending = false

    /** 商品の読み取りの予約。この時刻を過ぎたら撮影する。 */
    @Volatile private var productDueAt = NOT_DUE
    @Volatile private var productTrigger = ""
    @Volatile private var productAttempt = 0

    fun setCloudEnabled(value: Boolean) {
        cloudEnabled = value
        _state.update { it.copy(cloudEnabled = value, status = if (value) "" else "棚の判別：オフ") }
    }

    /** 手のスレッドから呼ぶ。 */
    internal fun onHands(timeSeconds: Double, hands: List<HandCandidate>) {
        if (held.update(timeSeconds, hands) != null) requestProduct(timeSeconds, "hand")
    }

    /** GL スレッドから毎フレーム呼ぶ。 */
    fun onFrame(frame: Frame, camera: Camera) {
        val time = frame.timestamp / 1e9
        val tracking = camera.trackingState == TrackingState.TRACKING
        if (tracking) updateDwell(time, camera)
        watchScene(frame, camera, time, tracking)
        // 商品を優先する。手に取った瞬間は、棚より知りたいことが多いため。
        if (time >= productDueAt) readProductIfIdle(frame, camera, time)
        if (tracking && dwellAt != NO_DWELL && productDueAt == NOT_DUE) labelShelfIfIdle(frame, camera, time)
    }

    fun close() {
        cloudWorker.shutdown()
        ocrWorker.shutdown()
    }

    private fun updateDwell(time: Double, camera: Camera) {
        if (dwellRetryPending) {
            dwellRetryPending = false
            dwell.retry()
        }
        val pose = camera.pose
        val z = pose.zAxis
        // 前の通信を待つ間に立ち止まった分も覚えておく。ただし、もう立ち去っているかもしれないので長くは待たない。
        if (dwell.update(time, Vec3(pose.tx(), pose.ty(), pose.tz()), Vec3(-z[0], -z[1], -z[2]))) dwellAt = time
        if (time - dwellAt > DWELL_WAIT_SECONDS) dwellAt = NO_DWELL
    }

    /** 画像の変化で、カメラの前に商品が出てきたことを見つける。手が箱に隠れていても気づける。 */
    private fun watchScene(frame: Frame, camera: Camera, time: Double, tracking: Boolean) {
        if (time - lastSignatureAt < SIGNATURE_INTERVAL_SECONDS) return
        lastSignatureAt = time
        val small = FrameCapture.capture(frame, camera, uprightRotationDegrees(), depthEnabled = false) ?: return
        val signature = SceneSignature.of(small.nv21, small.width, small.height)
        // 歩いている間は見える景色が変わるだけなので、商品が出てきたとは考えない。
        if (tracking && !dwell.isStill(time)) {
            scene.rebaseline(signature)
            return
        }
        if (scene.update(time, signature)) requestProduct(time, "scene")
    }

    private fun requestProduct(time: Double, trigger: String) {
        productAttempt = 0
        productTrigger = trigger
        productDueAt = time
    }

    private fun readProductIfIdle(frame: Frame, camera: Camera, time: Double) {
        if (!ocrBusy.compareAndSet(false, true)) return
        val captured = FrameCapture.capture(frame, camera, uprightRotationDegrees(), depthEnabled = false, fullResolution = true)
        if (captured == null) {
            ocrBusy.set(false)
            return
        }
        val trigger = productTrigger
        productDueAt = NOT_DUE
        _state.update { it.copy(status = "商品を読み取っています") }
        ocrWorker.execute {
            try {
                readProduct(captured, time, trigger)
            } catch (e: RuntimeException) {
                Log.e(TAG, "Product OCR failed", e)
                recordLine("ocr_failed", trigger, 0L, e.message.orEmpty())
                _state.update { it.copy(status = "商品を読み取れませんでした") }
            } finally {
                ocrBusy.set(false)
            }
        }
    }

    private fun readProduct(frame: CapturedFrame, time: Double, trigger: String) {
        val start = SystemClock.elapsedRealtime()
        val bitmap = JpegEncoder.toUprightBitmap(frame, OCR_MAX_LONG_SIDE)
        val product = try {
            saveLastImage(bitmap, LAST_PRODUCT_IMAGE)
            ProductReading.from(ocr.read(bitmap))
        } finally {
            bitmap.recycle()
        }
        record("product", trigger, start, "$product")

        // ブレやラベルの向きで読めないことがあるので、持っている間は少し待って撮り直す。
        val willRetry = !product.readable && productAttempt + 1 < MAX_PRODUCT_ATTEMPTS && productDueAt == NOT_DUE
        if (willRetry) {
            productAttempt++
            productTrigger = "retry"
            productDueAt = time + PRODUCT_RETRY_SECONDS
        }
        _state.update {
            when {
                product.readable -> it.copy(status = "", product = product)
                willRetry -> it.copy(status = "文字が見えません。ラベルをカメラに向けてください")
                else -> it.copy(status = "", product = product)
            }
        }
    }

    private fun labelShelfIfIdle(frame: Frame, camera: Camera, time: Double) {
        if (client == null || tasks == null || !cloudEnabled || time < retryAfterSeconds) return
        if (!cloudBusy.compareAndSet(false, true)) return
        val captured = FrameCapture.capture(frame, camera, uprightRotationDegrees(), depthEnabled = true, fullResolution = true)
        if (captured == null) {
            cloudBusy.set(false)
            return
        }
        if (!budget.tryAcquire(time)) {
            cloudBusy.set(false)
            Log.w(TAG, "Request budget exhausted; skipping")
            return
        }
        dwellAt = NO_DWELL
        _state.update { it.copy(status = "棚を判別しています") }
        cloudWorker.execute {
            try {
                labelShelf(client, tasks, captured, time)
            } catch (e: RuntimeException) {
                Log.e(TAG, "Shelf labeling failed", e)
                onCloudFailure(CloudException(CloudFailure.BAD_RESPONSE, "unexpected", e), time)
            } finally {
                cloudBusy.set(false)
            }
        }
    }

    private fun labelShelf(client: CloudVisionClient, tasks: CloudTasks, frame: CapturedFrame, time: Double) {
        val start = SystemClock.elapsedRealtime()
        val output = try {
            client.generate(tasks.shelfColumns, JpegEncoder.toUprightJpeg(frame, CLOUD_MAX_LONG_SIDE))
        } catch (e: CloudException) {
            onCloudFailure(e, time)
            return
        }
        val result = parser!!.parseShelfColumns(output)
        record("shelf", "dwell", start, output)
        val placed = result.columns.map { column ->
            val p = Unprojector.unproject(NormalizedBox(column.xStart, 0.25f, column.xEnd, 0.75f), frame.geometry)
            PlacedShelfColumn(column, p.position, p.hasMeasuredDepth)
        }
        _state.update {
            if (result.isShelfFront) it.copy(status = "", columns = placed) else it.copy(status = "") // 前の棚の結果は消さない
        }
        // 立ち止まった時点で商品を持っていると、棚が映らず、画像の変化も起きない。
        if (!result.isShelfFront && productDueAt == NOT_DUE) requestProduct(time, "not_shelf")
    }

    private fun onCloudFailure(e: CloudException, time: Double) {
        Log.w(TAG, "Cloud request failed: ${e.message}", e)
        recordLine("shelf_failed", "dwell", 0L, e.message.orEmpty())
        retryAfterSeconds = time + when (e.failure) {
            CloudFailure.AUTH -> AUTH_RETRY_SECONDS
            CloudFailure.RATE_LIMIT -> RATE_LIMIT_RETRY_SECONDS
            else -> RETRY_SECONDS
        }
        dwellRetryPending = true
        _state.update { it.copy(status = "棚の判別：${e.failure.messageJa}") }
    }

    private fun record(kind: String, trigger: String, startMillis: Long, output: String) {
        val millis = SystemClock.elapsedRealtime() - startMillis
        Log.i(TAG, "$kind trigger=$trigger ms=$millis $output")
        recordLine(kind, trigger, millis, output)
    }

    /** logcat はすぐ流れるので、あとで原因を調べられるようファイルにも残す。 */
    private fun recordLine(kind: String, trigger: String, millis: Long, detail: String) {
        val dir = debugDir ?: return
        try {
            File(dir, LOG_FILE).appendText(
                "${System.currentTimeMillis()}\t$kind\t$trigger\t$millis\t${detail.replace('\n', ' ')}\n",
            )
        } catch (e: IOException) {
            Log.w(TAG, "Could not write label log", e)
        }
    }

    private fun saveLastImage(bitmap: Bitmap, name: String) {
        val dir = debugDir ?: return
        try {
            FileOutputStream(File(dir, name)).use { bitmap.compress(Bitmap.CompressFormat.JPEG, DEBUG_JPEG_QUALITY, it) }
        } catch (e: IOException) {
            Log.w(TAG, "Could not save debug image", e)
        }
    }

    private companion object {
        const val TAG = "StopLabeler"
        const val CLOUD_MAX_LONG_SIDE = 1920
        const val OCR_MAX_LONG_SIDE = 1920
        const val NO_DWELL = Double.NEGATIVE_INFINITY
        const val NOT_DUE = Double.POSITIVE_INFINITY
        const val DWELL_WAIT_SECONDS = 3.0
        const val SIGNATURE_INTERVAL_SECONDS = 0.2
        const val MAX_PRODUCT_ATTEMPTS = 3
        const val PRODUCT_RETRY_SECONDS = 1.0
        const val RETRY_SECONDS = 5.0
        const val RATE_LIMIT_RETRY_SECONDS = 30.0
        const val AUTH_RETRY_SECONDS = 300.0
        const val DEBUG_JPEG_QUALITY = 90
        const val LOG_FILE = "label_log.tsv"
        const val LAST_PRODUCT_IMAGE = "ocr_last.jpg"
    }
}
