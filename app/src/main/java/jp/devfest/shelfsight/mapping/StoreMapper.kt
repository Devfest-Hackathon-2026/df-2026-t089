package jp.devfest.shelfsight.mapping

import android.opengl.Matrix
import android.os.SystemClock
import android.util.Log
import com.google.ar.core.Anchor
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotTrackingException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.ResourceExhaustedException
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import jp.devfest.shelfsight.ar.FrameCapture
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.max
import kotlin.math.sqrt
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class MapUiState(
    val status: String = "準備しています",
    val pointCount: Int = 0,
    val walkedMeters: Float = 0f,
    val isRecording: Boolean = true,
    val showMap: Boolean = false,
)

/**
 * 地図の一区画。点は Anchor からの相対座標で持つ。
 *
 * ARCore は歩行中に推定位置を補正するため、ワールド座標で固定すると壁が二重になる。
 * Anchor は補正に追従するので、区画単位で位置が直る。
 */
internal class MapChunk(val anchor: Anchor) {
    val grid = VoxelGrid()

    /** 区画座標からワールド座標への変換。追跡が止まっても地図が消えないよう、最後の値を保つ。 */
    val worldFromChunk = FloatArray(16).also { anchor.pose.toMatrix(it, 0) }

    /** ワーカースレッドが書き、GL スレッドが読む。 */
    @Volatile var vertices: IntArray = IntArray(0)
    @Volatile var vertexVersion = 0

    // 以下は GL スレッドのみが触る。
    var vbo = 0
    var uploadedVersion = -1
    var uploadedVertexCount = 0

    val position: Vec3
        get() = Vec3(worldFromChunk[12], worldFromChunk[13], worldFromChunk[14])
}

/**
 * 歩行に合わせて地図を積み上げる。
 *
 * ARCore の操作は GL スレッドで行い、深度の統合はワーカースレッドに任せて描画を止めない。
 */
class StoreMapper {
    private val _state = MutableStateFlow(MapUiState())
    val state: StateFlow<MapUiState> = _state.asStateFlow()

    val orbit = OrbitCamera()

    @Volatile private var recording = true
    @Volatile private var fitRequested = false
    @Volatile private var blockingMessage: String? = null

    // 以下は GL スレッドのみが触る。
    internal val chunks = ArrayList<MapChunk>()
    private var trailLocal = FloatArray(3 * 1024)
    private var trailChunks = arrayOfNulls<MapChunk>(1024)
    internal var trailSize = 0
        private set
    private var lastTrailWorld: Vec3? = null
    private var walkedMeters = 0f
    internal var currentPosition: Vec3? = null
        private set
    internal var currentForward = Vec3(0f, 0f, -1f)
        private set
    private val poseHistory = PoseHistory<Pose>()
    private val stats = MappingStats()
    private var lastStatsLogMillis = 0L
    private var lastDepthTimestamp = 0L
    private var lastCaptureMillis = 0L
    private var lastPublishMillis = 0L
    private var lastStatus = ""

    /** 天井を映さない高さ。歩行中の端末の高さを基準にする。 */
    internal var ceilingClipY = Float.MAX_VALUE
        private set

    private val worker = Executors.newSingleThreadExecutor { Thread(it, "map-fusion") }
    private val busy = AtomicBoolean(false)
    private val totalVoxels = AtomicInteger(0)
    private val totalConfirmed = AtomicInteger(0)
    private val pauseCount = AtomicInteger(0)

    // ワーカースレッドのみが触る。
    private val dirtyChunks = LinkedHashSet<MapChunk>()
    private var lastRebuildMillis = 0L

    fun setRecording(enabled: Boolean) {
        if (recording && !enabled) pauseCount.incrementAndGet()
        recording = enabled
        _state.update { it.copy(isRecording = enabled) }
    }

    fun setShowMap(show: Boolean) {
        if (show) fitRequested = true
        _state.update { it.copy(showMap = show) }
    }

    /** 深度が使えないなど、歩いても解決しない問題を表示する。 */
    fun setBlockingMessage(message: String?) {
        blockingMessage = message
        if (message != null) _state.update { it.copy(status = message) }
    }

    /** GL スレッドから毎フレーム呼ぶ。 */
    internal fun onFrame(session: Session, frame: Frame, camera: Camera) {
        refreshChunkPoses()
        if (fitRequested) {
            fitRequested = false
            fitOrbitToTrail()
        }

        val tracking = camera.trackingState == TrackingState.TRACKING
        stats.onTracking(tracking)
        if (tracking) {
            val pose = camera.pose
            poseHistory.record(frame.timestamp, pose)
            val position = Vec3(pose.tx(), pose.ty(), pose.tz())
            val z = pose.zAxis
            currentPosition = position
            currentForward = Vec3(-z[0], -z[1], -z[2])
            if (recording && blockingMessage == null) {
                chunkAt(session, position)?.let { chunk ->
                    extendTrail(chunk, position)
                    captureIfDue(frame, camera, chunk)
                }
            }
        }
        publish(statusFor(camera, tracking))
        logStatsIfDue()
    }

    fun close() {
        worker.shutdown()
    }

    /** GL コンテキストが作り直されたら、VBO を作り直す。 */
    internal fun onGlContextLost() {
        chunks.forEach {
            it.vbo = 0
            it.uploadedVersion = -1
        }
    }

    /** 区画の補正を反映した軌跡のワールド座標を、x, y, z の順に [out] へ書く。 */
    internal fun writeTrailWorld(out: FloatArray) {
        for (i in 0 until trailSize) {
            val m = trailChunks[i]!!.worldFromChunk
            val x = trailLocal[i * 3]
            val y = trailLocal[i * 3 + 1]
            val z = trailLocal[i * 3 + 2]
            out[i * 3] = m[0] * x + m[4] * y + m[8] * z + m[12]
            out[i * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13]
            out[i * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14]
        }
    }

    private fun refreshChunkPoses() {
        for (chunk in chunks) {
            if (chunk.anchor.trackingState == TrackingState.TRACKING) {
                chunk.anchor.pose.toMatrix(chunk.worldFromChunk, 0)
            }
        }
    }

    /** 同じ場所に戻ったときは既存の区画を使い、Anchor が増え続けないようにする。 */
    private fun chunkAt(session: Session, position: Vec3): MapChunk? {
        val nearest = chunks
            .filter { it.anchor.trackingState == TrackingState.TRACKING }
            .minByOrNull { it.position.distanceTo(position) }
        if (nearest != null && nearest.position.distanceTo(position) < CHUNK_RADIUS_M) return nearest

        return try {
            val anchor = session.createAnchor(Pose.makeTranslation(position.x, position.y, position.z))
            MapChunk(anchor).also { chunks += it }
        } catch (e: NotTrackingException) {
            Log.w(TAG, "Could not create chunk anchor", e)
            nearest
        } catch (e: ResourceExhaustedException) {
            Log.w(TAG, "Could not create chunk anchor", e)
            nearest
        }
    }

    private fun extendTrail(chunk: MapChunk, position: Vec3) {
        val last = lastTrailWorld
        if (last != null) {
            val step = position.distanceTo(last)
            if (step < TRAIL_STEP_M) return
            // 再位置合わせによる瞬間移動は、歩いた距離に数えない。
            if (step < MAX_WALK_STEP_M) walkedMeters += step else stats.onTrailJump(step)
        }
        lastTrailWorld = position

        if (trailSize == trailChunks.size) {
            trailChunks = trailChunks.copyOf(trailSize * 2)
            trailLocal = trailLocal.copyOf(trailSize * 2 * 3)
        }
        val local = chunk.anchor.pose.inverse().transformPoint(
            floatArrayOf(position.x, position.y, position.z),
        )
        trailChunks[trailSize] = chunk
        local.copyInto(trailLocal, trailSize * 3)
        trailSize++
    }

    private fun captureIfDue(frame: Frame, camera: Camera, chunk: MapChunk) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastCaptureMillis < MIN_CAPTURE_INTERVAL_MS || busy.get()) return
        if (totalVoxels.get() >= MAX_VOXELS) return

        val sample = captureSample(frame, camera, chunk) ?: return
        lastCaptureMillis = now
        busy.set(true)
        worker.execute {
            try {
                integrate(chunk, sample)
            } catch (e: RuntimeException) {
                Log.e(TAG, "Depth fusion failed", e)
            } finally {
                busy.set(false)
            }
        }
    }

    private fun captureSample(frame: Frame, camera: Camera, chunk: MapChunk): DepthSample? {
        try {
            frame.acquireRawDepthImage16Bits().use { depth ->
                frame.acquireRawDepthConfidenceImage().use { confidence ->
                    if (depth.timestamp == lastDepthTimestamp) return null
                    val lagNanos = frame.timestamp - depth.timestamp
                    if (lagNanos > MAX_DEPTH_AGE_NANOS) {
                        lastDepthTimestamp = depth.timestamp
                        stats.onStale(lagNanos / NANOS_PER_MILLI)
                        return null
                    }
                    val depthPose = poseHistory.nearest(depth.timestamp, MAX_POSE_GAP_NANOS)
                    if (depthPose == null) {
                        lastDepthTimestamp = depth.timestamp
                        stats.onNoPose()
                        return null
                    }
                    val cameraFrame = FrameCapture.capture(frame, camera, 0, false) ?: return null
                    lastDepthTimestamp = depth.timestamp
                    stats.onAccepted(lagNanos / NANOS_PER_MILLI)

                    val width = depth.width
                    val height = depth.height
                    val depthPlane = depth.planes[0]
                    val shorts = depthPlane.buffer.order(ByteOrder.nativeOrder()).asShortBuffer()
                    val depthStride = depthPlane.rowStride / Short.SIZE_BYTES
                    val depthMm = ShortArray(width * height)
                    for (row in 0 until height) {
                        shorts.position(row * depthStride)
                        shorts.get(depthMm, row * width, width)
                    }

                    val confidencePlane = confidence.planes[0]
                    val bytes = confidencePlane.buffer
                    val confidenceValues = ByteArray(width * height)
                    for (row in 0 until height) {
                        bytes.position(row * confidencePlane.rowStride)
                        bytes.get(confidenceValues, row * width, width)
                    }

                    val chunkFromCamera = FloatArray(16)
                    chunk.anchor.pose.inverse().compose(depthPose).toMatrix(chunkFromCamera, 0)
                    return DepthSample(
                        width, height, depthMm, confidenceValues,
                        cameraFrame.nv21, cameraFrame.geometry, chunkFromCamera,
                    )
                }
            }
        } catch (_: NotYetAvailableException) {
            return null // 歩き始めて視差が得られるまでは深度がない
        }
    }

    /** ワーカースレッドで実行する。 */
    private fun integrate(chunk: MapChunk, sample: DepthSample) {
        val voxelsBefore = chunk.grid.size
        val confirmedBefore = chunk.grid.confirmedCount
        DepthFusion.integrate(sample, chunk.grid)
        totalVoxels.addAndGet(chunk.grid.size - voxelsBefore)
        totalConfirmed.addAndGet(chunk.grid.confirmedCount - confirmedBefore)
        dirtyChunks += chunk

        val now = SystemClock.elapsedRealtime()
        if (now - lastRebuildMillis < MIN_REBUILD_INTERVAL_MS) return
        lastRebuildMillis = now
        for (dirty in dirtyChunks) {
            dirty.vertices = dirty.grid.toVertices()
            dirty.vertexVersion++
        }
        dirtyChunks.clear()
    }

    private fun fitOrbitToTrail() {
        if (trailSize == 0) return
        val world = FloatArray(trailSize * 3)
        writeTrailWorld(world)

        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
        var sumY = 0f
        for (i in 0 until trailSize) {
            val x = world[i * 3]
            val z = world[i * 3 + 2]
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (z < minZ) minZ = z
            if (z > maxZ) maxZ = z
            sumY += world[i * 3 + 1]
        }
        val deviceY = sumY / trailSize
        ceilingClipY = deviceY + CEILING_CLEARANCE_M
        val halfX = (maxX - minX) / 2f
        val halfZ = (maxZ - minZ) / 2f
        orbit.fit(
            center = Vec3(minX + halfX, deviceY - DEVICE_HEIGHT_M, minZ + halfZ),
            radius = max(sqrt(halfX * halfX + halfZ * halfZ), MAX_DEPTH_FOR_FIT_M),
        )
    }

    private fun statusFor(camera: Camera, tracking: Boolean): String {
        blockingMessage?.let { return it }
        if (!tracking) {
            return when (camera.trackingFailureReason) {
                TrackingFailureReason.INSUFFICIENT_LIGHT -> "暗すぎます。明るい場所を映してください"
                TrackingFailureReason.EXCESSIVE_MOTION -> "動きが速すぎます。ゆっくり動かしてください"
                TrackingFailureReason.INSUFFICIENT_FEATURES -> "模様の多い場所を映してください"
                TrackingFailureReason.CAMERA_UNAVAILABLE -> "カメラを使用できません"
                else -> "端末をゆっくり動かしてください"
            }
        }
        if (totalVoxels.get() >= MAX_VOXELS) return "地図の容量が上限に達しました"
        if (!recording) return "記録を一時停止しています"
        if (totalConfirmed.get() == 0) return "端末をゆっくり左右に動かしてください"
        return "記録中：店内をゆっくり歩いてください"
    }

    private fun logStatsIfDue() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastStatsLogMillis < STATS_LOG_INTERVAL_MS) return
        lastStatsLogMillis = now
        Log.i(
            TAG,
            "stats ${stats.summary()} pauses=${pauseCount.get()} chunks=${chunks.size} " +
                "voxels=${totalVoxels.get()} points=${totalConfirmed.get()} walkedM=$walkedMeters",
        )
    }

    private fun publish(status: String) {
        val now = SystemClock.elapsedRealtime()
        // 読み上げが途切れないよう、状況が変わったときはすぐ反映し、数値の更新は間引く。
        if (status == lastStatus && now - lastPublishMillis < PUBLISH_INTERVAL_MS) return
        lastStatus = status
        lastPublishMillis = now
        _state.update {
            it.copy(status = status, pointCount = totalConfirmed.get(), walkedMeters = walkedMeters)
        }
    }

    private companion object {
        const val TAG = "StoreMapper"
        const val CHUNK_RADIUS_M = 2.5f
        const val TRAIL_STEP_M = 0.1f
        const val MAX_WALK_STEP_M = 2f
        const val MIN_CAPTURE_INTERVAL_MS = 150L
        const val MIN_REBUILD_INTERVAL_MS = 400L
        const val PUBLISH_INTERVAL_MS = 500L
        const val STATS_LOG_INTERVAL_MS = 10_000L
        const val NANOS_PER_MILLI = 1_000_000L

        /** これより古い深度は、対応する姿勢の信頼性が低いので地図に入れない。 */
        const val MAX_DEPTH_AGE_NANOS = 500L * NANOS_PER_MILLI

        /** 30fps のフレーム間隔の半分強。これ以上離れた姿勢は同じ瞬間のものとして扱わない。 */
        const val MAX_POSE_GAP_NANOS = 20L * NANOS_PER_MILLI

        /** メモリと描画負荷の上限。5cm 単位なら一般的なコンビニの表面を十分に覆える想定。 */
        const val MAX_VOXELS = 3_000_000

        /** 端末より上にある棚は残しつつ、天井は表示しない。 */
        const val CEILING_CLEARANCE_M = 0.6f
        const val DEVICE_HEIGHT_M = 1.2f
        const val MAX_DEPTH_FOR_FIT_M = 3f
    }
}
