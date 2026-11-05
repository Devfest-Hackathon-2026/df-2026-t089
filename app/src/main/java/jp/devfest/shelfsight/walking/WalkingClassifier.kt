package jp.devfest.shelfsight.walking

import android.os.Process
import android.os.SystemClock
import android.util.Log
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import java.io.File
import java.io.IOException
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import jp.devfest.shelfsight.ar.FrameCapture
import jp.devfest.shelfsight.ar.JpegEncoder
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class WalkingState(
    val available: Boolean = false,
    val enabled: Boolean = true,
    /** 左から右の列ごとのカテゴリ。まだ判別していなければ空。 */
    val columns: List<String> = emptyList(),
    val classes: List<String> = emptyList(),
    /** 最後の判別にかかった時間（撮影から結果まで）。 */
    val millis: Long = 0,
    val modelName: String = "",
)

/**
 * 歩いている間、カメラ画像の列ごとに大まかなカテゴリを端末内で判別し続ける。
 *
 * 撮影だけを GL スレッドで行い、判別は別のスレッドで行う。前の判別が終わるまで次は撮らないので、
 * 端末が遅ければ自然に間隔が空く。
 *
 * @param load モデルを読み込む。null を返したらモデルが入っていないビルドで、何もしない。
 *   ONNX Runtime の計算スレッドは読み込んだスレッドの優先度を引き継ぐので、画面のスレッドではなくここで読み込む。
 * @param debugDir 判別の履歴（時刻・かかった時間・結果）を残す場所。null なら残さない
 */
class WalkingClassifier(
    load: () -> CoarseClassifier?,
    private val debugDir: File?,
    private val uprightRotationDegrees: () -> Int,
) {
    private val _state = MutableStateFlow(WalkingState())
    val state: StateFlow<WalkingState> = _state.asStateFlow()

    private val worker = Executors.newSingleThreadExecutor { Thread(it, "walking-classifier") }
    @Volatile private var classifier: CoarseClassifier? = null
    private val busy = AtomicBoolean(false)
    @Volatile private var enabled = true
    private var lastShotMillis = 0L // GL スレッドのみ

    init {
        worker.execute {
            // AR の追跡や描画より優先しない。遅れても歩行中の判別は困らないが、追跡が止まると地図が崩れる。
            Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
            val loaded = try {
                load()
            } catch (e: Exception) {
                Log.e(TAG, "Could not load the walking classifier", e)
                null
            }
            classifier = loaded
            if (loaded != null) {
                _state.update { it.copy(available = true, classes = loaded.head.classes, modelName = loaded.head.modelName) }
            }
        }
    }

    fun setEnabled(value: Boolean) {
        enabled = value
        _state.update { it.copy(enabled = value, columns = if (value) it.columns else emptyList()) }
    }

    /** GL スレッドから毎フレーム呼ぶ。AR の追跡が切れていても判別はできるので、追跡の状態は見ない。 */
    fun onFrame(frame: Frame, camera: Camera) {
        val model = classifier ?: return
        val now = SystemClock.elapsedRealtime()
        if (!enabled || now - lastShotMillis < MIN_INTERVAL_MS || !busy.compareAndSet(false, true)) return
        val captured = FrameCapture.capture(frame, camera, uprightRotationDegrees(), depthEnabled = false, usualLongSide = CAPTURE_LONG_SIDE)
        if (captured == null) {
            busy.set(false)
            return
        }
        lastShotMillis = now
        val timestamp = frame.timestamp
        worker.execute {
            try {
                val image = JpegEncoder.toUprightBitmap(captured, CAPTURE_LONG_SIDE)
                val result = try {
                    model.classify(image)
                } finally {
                    image.recycle()
                }
                val total = SystemClock.elapsedRealtime() - now
                if (enabled) _state.update { it.copy(columns = result.columns, millis = total) }
                log(timestamp, total, result)
            } catch (e: RuntimeException) {
                Log.e(TAG, "Walking classification failed", e)
            } finally {
                busy.set(false)
            }
        }
    }

    fun close() {
        worker.execute { classifier?.close() } // 判別中に閉じないよう、同じスレッドで閉じる
        worker.shutdown()
    }

    /** あとで記録データ（同じ timestampNs を持つ）と突き合わせて精度を測れるよう、ファイルに残す。 */
    private fun log(timestampNs: Long, totalMillis: Long, result: CoarseResult) {
        Log.i(TAG, "ms=$totalMillis (prep ${result.preprocessMillis}, model ${result.inferenceMillis}) ${result.columns}")
        val dir = debugDir ?: return
        val confidence = result.confidence.joinToString(",") { String.format(Locale.US, "%.2f", it) }
        try {
            File(dir, LOG_FILE).appendText(
                "${System.currentTimeMillis()}\t$timestampNs\t$totalMillis\t${result.preprocessMillis}\t" +
                    "${result.inferenceMillis}\t${result.columns.joinToString(",")}\t$confidence\n",
            )
        } catch (e: IOException) {
            Log.w(TAG, "Could not write walking log", e)
        }
    }

    private companion object {
        const val TAG = "WalkingClassifier"
        const val LOG_FILE = "walking_log.tsv"
        const val MIN_INTERVAL_MS = 500L

        /** 列の切り出しが 256 画素を少し上回る大きさ。これより小さいと文字や絵柄がつぶれる。 */
        const val CAPTURE_LONG_SIDE = 1280
    }
}
