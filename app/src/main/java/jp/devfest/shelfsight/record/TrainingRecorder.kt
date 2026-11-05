package jp.devfest.shelfsight.record

import android.os.SystemClock
import android.util.Log
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import java.io.File
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import jp.devfest.shelfsight.ar.CapturedFrame
import jp.devfest.shelfsight.ar.FrameCapture
import jp.devfest.shelfsight.ar.JpegEncoder
import jp.devfest.shelfsight.mapping.StoreMapper
import jp.devfest.shelfsight.mapping.SurfaceBasis
import jp.devfest.shelfsight.mapping.SurfaceKind
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject

/**
 * 歩きながら、学習用のカメラ画像と、そのときのカメラと壁の位置を記録する。
 *
 * PC の tools/rectify.py がこれを読み、壁を正面から見た画像に直す。
 * 画像はセンサーの向きのまま保存するので、記録した内部パラメータをそのまま使える。
 * 来店者が写ることがあるので、記録は端末内だけに保存し、PC へは手で取り出す。
 *
 * 出力：<root>/training/<開始時刻>/frames.jsonl と 000001.jpg ...
 */
class TrainingRecorder internal constructor(private val root: File, private val mapper: StoreMapper) {
    private val _frames = MutableStateFlow(-1)

    /** 記録した枚数。記録していなければ -1。 */
    val frames: StateFlow<Int> = _frames.asStateFlow()

    private val worker = Executors.newSingleThreadExecutor { Thread(it, "training-recorder") }
    private val busy = AtomicBoolean(false)
    @Volatile private var session: File? = null
    private var lastShotMillis = 0L
    private var count = 0

    fun toggle() {
        if (session != null) {
            session = null
            _frames.value = -1
            return
        }
        val dir = File(root, "training/${System.currentTimeMillis()}")
        if (!dir.mkdirs()) {
            Log.w(TAG, "Could not create $dir")
            return
        }
        count = 0
        session = dir
        _frames.value = 0
    }

    /** GL スレッドから毎フレーム呼ぶ。 */
    fun onFrame(frame: Frame, camera: Camera) {
        val dir = session ?: return
        val now = SystemClock.elapsedRealtime()
        if (camera.trackingState != TrackingState.TRACKING || now - lastShotMillis < INTERVAL_MS || busy.get()) return
        val walls = mapper.surfaceFusion.latest.surfaces.filter { it.kind == SurfaceKind.WALL }
        val floors = mapper.surfaceFusion.latest.surfaces.filter { it.kind == SurfaceKind.FLOOR }
        val captured = FrameCapture.capture(frame, camera, 0, depthEnabled = false, usualLongSide = LONG_SIDE) ?: return
        lastShotMillis = now
        val wallJson = JSONArray()
        for (wall in walls) wallJson.put(basisJson(wall.id, wall.fromPoints, wall.currentBasis()))
        val floorY = floors.map { it.currentBasis().origin[1] }.minOrNull()
        busy.set(true)
        val index = ++count
        worker.execute {
            try {
                save(dir, index, captured, frame.timestamp, wallJson, floorY)
                _frames.value = index
            } catch (e: IOException) {
                Log.e(TAG, "Could not save training frame", e)
            } finally {
                busy.set(false)
            }
        }
    }

    fun close() {
        worker.shutdown()
    }

    private fun save(dir: File, index: Int, frame: CapturedFrame, timestamp: Long, walls: JSONArray, floorY: Float?) {
        val name = "%06d.jpg".format(index)
        File(dir, name).writeBytes(JpegEncoder.toSensorJpeg(frame, JPEG_QUALITY))
        val g = frame.geometry
        val line = JSONObject()
            .put("image", name)
            .put("timestampNs", timestamp)
            .put("width", frame.width)
            .put("height", frame.height)
            .put("fx", g.focalX.toDouble()).put("fy", g.focalY.toDouble())
            .put("cx", g.principalX.toDouble()).put("cy", g.principalY.toDouble())
            // 列優先の 4x4。カメラ座標は +X が画像の右、+Y が画像の上、-Z が前方。
            .put("cameraToWorld", JSONArray(g.cameraToWorld.map { it.toDouble() }))
            .put("floorY", floorY?.toDouble() ?: JSONObject.NULL)
            .put("walls", walls)
        File(dir, "frames.jsonl").appendText(line.toString() + "\n")
    }

    private fun basisJson(id: Int, fromPoints: Boolean, b: SurfaceBasis) = JSONObject()
        .put("id", id)
        .put("fromPoints", fromPoints)
        .put("origin", JSONArray(b.origin.map { it.toDouble() }))
        .put("axisU", JSONArray(b.axisU.map { it.toDouble() }))
        .put("axisV", JSONArray(b.axisV.map { it.toDouble() }))
        .put("uMin", b.uMin.toDouble()).put("uMax", b.uMax.toDouble())
        .put("vMin", b.vMin.toDouble()).put("vMax", b.vMax.toDouble())

    private companion object {
        const val TAG = "TrainingRecorder"
        const val INTERVAL_MS = 500L
        const val LONG_SIDE = 960
        const val JPEG_QUALITY = 88
    }
}
