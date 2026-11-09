package jp.devfest.shelfsight.assist

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.Image
import android.net.Uri
import android.os.Build
import android.util.Log
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.RecordingConfig
import com.google.ar.core.Session
import com.google.ar.core.exceptions.RecordingFailedException
import jp.devfest.shelfsight.ar.CapturedFrame
import jp.devfest.shelfsight.ar.FrameCapture
import jp.devfest.shelfsight.ar.JpegEncoder
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 店を歩いた 1 回分の ARCore のデータを、あとで PC で再生して実験できるように全部保存する（products/replay.py が読む）。
 *
 *   session.mp4    ARCore の Recording API（映像と IMU。Session.setPlaybackDatasetUri で実機に戻せる）
 *   meta.json      端末とカメラの設定
 *   poses.jsonl    毎フレームの位置と向き・内部パラメータ・追跡状態
 *   planes.jsonl   更新された平面（毎フレーム、更新分だけ）
 *   keyframes.jsonl / frames/ depth/ points/   KEYFRAME_MS ごとのカメラ画像、深度（補完済み・生・信頼度）、特徴点
 *   sensors.csv    端末の各センサーの生の値
 *   events.jsonl   記録中の操作（タップ・ダブルタップ・長押し・聞き取った言葉）
 *
 * onFrame は GL スレッドから、書き込みは 1 本の別スレッドで行う。
 */
class SessionRecorder(private val context: Context, root: File) : SensorEventListener {
    val name: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.JAPAN).format(Date())
    private val dir = File(root, name).apply { File(this, "frames").mkdirs(); File(this, "depth").mkdirs(); File(this, "points").mkdirs() }
    private val io = Executors.newSingleThreadExecutor()
    private val poses = writer("poses.jsonl")
    private val planes = writer("planes.jsonl")
    private val keyframes = writer("keyframes.jsonl")
    private val events = writer("events.jsonl")
    private val sensors = writer("sensors.csv").also { it.write("sensor,timestamp_ns,values\n") }
    private val sensorManager = context.getSystemService(SensorManager::class.java)
    private val encoding = AtomicBoolean(false)
    private var lastKeyframe = 0L
    private var keyframeCount = 0
    private var startMs = 0L
    private var lastStatus = 0L
    @Volatile private var lastFrameT = 0L
    @Volatile var hasMp4 = false
        private set

    private fun writer(file: String): BufferedWriter = File(dir, file).bufferedWriter()

    /** 記録を始める。MP4 が作れなくても、ほかのデータは記録する。 */
    fun start(session: Session, sensorRotation: Int) {
        startMs = System.currentTimeMillis()
        try {
            session.startRecording(RecordingConfig(session).setMp4DatasetUri(Uri.fromFile(File(dir, "session.mp4"))).setAutoStopOnPause(true))
            hasMp4 = true
        } catch (e: RecordingFailedException) {
            Log.e(TAG, "ARCore recording failed", e)
        }
        val c = session.cameraConfig
        val config = session.config
        File(dir, "meta.json").writeText(
            JSONObject().put("name", name).put("start_ms", startMs).put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT)
                .put("mp4", hasMp4).put("camera_id", c.cameraId).put("image", JSONArray(listOf(c.imageSize.width, c.imageSize.height)))
                .put("texture", JSONArray(listOf(c.textureSize.width, c.textureSize.height))).put("fps", JSONArray(listOf(c.fpsRange.lower, c.fpsRange.upper)))
                .put("plane_finding_mode", config.planeFindingMode.name).put("depth_mode", config.depthMode.name)
                .put("light_estimation_mode", config.lightEstimationMode.name).put("focus_mode", config.focusMode.name)
                .put("update_mode", config.updateMode.name)
                .put("sensor_rotation", sensorRotation).put("keyframe_ms", KEYFRAME_MS).put("long_side", LONG_SIDE).toString(1),
        )
        for (s in SENSOR_TYPES.mapNotNull { sensorManager.getDefaultSensor(it) }) sensorManager.registerListener(this, s, SensorManager.SENSOR_DELAY_GAME)
    }

    /** 今の状態の短い文（ボタンに出す）。1 秒に 1 回だけ文字を返す。 */
    fun statusIfDue(now: Long): String? {
        if (now - lastStatus < 1000L) return null
        lastStatus = now
        return "■ 停止 ${(now - startMs) / 1000}秒 $keyframeCount 枚"
    }

    /** 空きが足りなければ true（呼ぶ側が止める）。 */
    fun outOfSpace(): Boolean = dir.usableSpace < MIN_FREE_BYTES

    /** 毎フレーム GL スレッドから呼ぶ。 */
    fun onFrame(frame: Frame, camera: Camera, sensorRotation: Int) {
        val t = frame.timestamp
        lastFrameT = t
        val pose = camera.pose
        val line = JSONObject().put("t", t).put("pose", poseJson(pose)).put("display_pose", poseJson(camera.displayOrientedPose))
            .put("state", camera.trackingState.name).put("reason", camera.trackingFailureReason.name)
            .put("image_intrinsics", intrinsics(camera.imageIntrinsics)).put("texture_intrinsics", intrinsics(camera.textureIntrinsics)).toString()
        write(poses, line)
        for (p in frame.getUpdatedTrackables(Plane::class.java)) write(planes, planeJson(t, p))
        if (t - lastKeyframe >= KEYFRAME_MS * 1_000_000L && encoding.compareAndSet(false, true)) {
            val f = FrameCapture.capture(frame, camera, sensorRotation, depthEnabled = true, fullResolution = true)
            if (f == null) encoding.set(false) else {
                lastKeyframe = t
                saveKeyframe(keyframeCount++, t, f, rawDepth(frame), rawConfidence(frame), points(frame), camera.trackingState.name)
            }
        }
    }

    private fun saveKeyframe(n: Int, t: Long, f: CapturedFrame, raw: Grid?, confidence: Grid?, cloud: ByteArray?, state: String) {
        val id = "%05d".format(n)
        io.execute {
            try {
                File(dir, "frames/$id.jpg").writeBytes(JpegEncoder.toUprightJpeg(f, LONG_SIDE))
                val g = f.geometry
                val depth = g.depth
                if (depth != null) {
                    val b = ByteBuffer.allocate(depth.millimeters.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                    b.asShortBuffer().put(depth.millimeters)
                    File(dir, "depth/$id.d16").writeBytes(b.array())
                }
                raw?.let { File(dir, "depth/$id.raw16").writeBytes(it.bytes) }
                confidence?.let { File(dir, "depth/$id.conf8").writeBytes(it.bytes) }
                cloud?.let { File(dir, "points/$id.bin").writeBytes(it) }
                val o = JSONObject().put("n", n).put("t", t).put("wall_ms", System.currentTimeMillis()).put("image", "frames/$id.jpg").put("state", state)
                    .put("geom", JSONObject().put("w", g.imageWidth).put("h", g.imageHeight).put("fx", g.focalX.toDouble()).put("fy", g.focalY.toDouble())
                        .put("cx", g.principalX.toDouble()).put("cy", g.principalY.toDouble()).put("rot", g.uprightRotationDegrees)
                        .put("m", JSONArray(g.cameraToWorld.map { it.toDouble() })).put("u0", g.depthU0.toDouble()).put("du", g.depthUPerPixel.toDouble())
                        .put("v0", g.depthV0.toDouble()).put("dv", g.depthVPerPixel.toDouble()))
                    .put("depth", depth?.let { JSONArray(listOf(it.width, it.height)) } ?: JSONObject.NULL)
                    .put("raw", raw?.let { JSONArray(listOf(it.width, it.height)) } ?: JSONObject.NULL)
                    .put("points", (cloud?.size ?: 0) / POINT_BYTES)
                keyframes.write(o.toString() + "\n")
            } catch (e: IOException) {
                Log.w(TAG, "could not save keyframe $id", e)
            } finally {
                encoding.set(false)
            }
        }
    }

    private class Grid(val width: Int, val height: Int, val bytes: ByteArray)

    private fun rawDepth(frame: Frame): Grid? = grab(2) { frame.acquireRawDepthImage16Bits() }
    private fun rawConfidence(frame: Frame): Grid? = grab(1) { frame.acquireRawDepthConfidenceImage() }

    /** まだ用意できていないときは ARCore が例外を投げるので、その枚は無しにする。 */
    private fun grab(bytesPerPixel: Int, acquire: () -> Image): Grid? = try {
        acquire().use { image ->
            val plane = image.planes[0]
            val out = ByteArray(image.width * image.height * bytesPerPixel)
            val rowBytes = image.width * bytesPerPixel
            val buffer = plane.buffer
            for (row in 0 until image.height) {
                buffer.position(row * plane.rowStride)
                buffer.get(out, row * rowBytes, rowBytes)
            }
            Grid(image.width, image.height, out)
        }
    } catch (e: com.google.ar.core.exceptions.NotYetAvailableException) {
        null
    }

    /** 特徴点: 1 点ごとに x, y, z, 信頼度 (float) と id (int) の 20 バイト。 */
    private fun points(frame: Frame): ByteArray? = try {
        frame.acquirePointCloud().use { cloud ->
            val p = cloud.points
            val ids = cloud.ids
            val n = ids.remaining()
            val out = ByteBuffer.allocate(n * POINT_BYTES).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until n) {
                for (k in 0 until 4) out.putFloat(p.get(i * 4 + k))
                out.putInt(ids.get(i))
            }
            out.array()
        }
    } catch (e: com.google.ar.core.exceptions.ResourceExhaustedException) {
        null
    }

    private fun write(w: BufferedWriter, line: String) {
        io.execute { try { w.write(line); w.write("\n") } catch (e: IOException) { Log.w(TAG, "write failed", e) } }
    }

    /** 記録中の操作を残す。 */
    fun event(type: String, text: String? = null) {
        val o = JSONObject().put("type", type).put("t", lastFrameT).put("wall_ms", System.currentTimeMillis())
        if (text != null) o.put("text", text)
        write(events, o.toString())
    }

    override fun onSensorChanged(e: SensorEvent) {
        val line = "${e.sensor.type},${e.timestamp},${e.values.joinToString(" ")}"
        write(sensors, line)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /** ファイルを閉じる。ARCore の録画は呼ぶ側が止める（一時停止なら自動で止まる）。 */
    fun close() {
        sensorManager.unregisterListener(this)
        io.shutdown()
        if (!io.awaitTermination(5, TimeUnit.SECONDS)) Log.w(TAG, "writer did not finish in time")
        for (w in listOf(poses, planes, keyframes, events, sensors)) try { w.close() } catch (e: IOException) { Log.w(TAG, "close failed", e) }
    }

    private fun poseJson(p: Pose) = JSONArray(listOf(p.tx(), p.ty(), p.tz(), p.qx(), p.qy(), p.qz(), p.qw()).map { it.toDouble() })

    private fun intrinsics(c: com.google.ar.core.CameraIntrinsics): JSONArray {
        val f = c.focalLength
        val p = c.principalPoint
        val d = c.imageDimensions
        return JSONArray(listOf(f[0].toDouble(), f[1].toDouble(), p[0].toDouble(), p[1].toDouble(), d[0].toDouble(), d[1].toDouble()))
    }

    private fun planeJson(t: Long, p: Plane): String {
        val m = FloatArray(16).also { p.centerPose.toMatrix(it, 0) }
        val polygon = p.polygon
        val poly = JSONArray()
        for (i in 0 until polygon.remaining()) poly.put(polygon.get(i).toDouble())
        return JSONObject().put("t", t).put("id", System.identityHashCode(p)).put("type", p.type.name).put("state", p.trackingState.name)
            .put("subsumed", p.subsumedBy != null).put("m", JSONArray(m.map { it.toDouble() })).put("ex", p.extentX.toDouble()).put("ez", p.extentZ.toDouble())
            .put("poly", poly).toString()
    }

    companion object {
        private const val TAG = "SessionRecorder"
        /** キーフレーム（画像・深度）の間隔。案内の認識は 1 秒ごとなので、その 5 倍の密度で残す。 */
        const val KEYFRAME_MS = 200L
        /** 案内の認識に渡すのと同じ長辺。 */
        const val LONG_SIDE = 1280
        private const val POINT_BYTES = 20
        private const val MIN_FREE_BYTES = 1_000_000_000L
        private val SENSOR_TYPES = listOf(
            Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE, Sensor.TYPE_MAGNETIC_FIELD, Sensor.TYPE_GRAVITY, Sensor.TYPE_LINEAR_ACCELERATION,
            Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_GAME_ROTATION_VECTOR, Sensor.TYPE_ACCELEROMETER_UNCALIBRATED, Sensor.TYPE_GYROSCOPE_UNCALIBRATED,
            Sensor.TYPE_PRESSURE,
        )
    }
}
