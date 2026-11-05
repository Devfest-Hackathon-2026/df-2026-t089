package jp.devfest.shelfsight.mapping

import android.opengl.Matrix
import android.os.SystemClock
import android.util.Log
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import jp.devfest.shelfsight.ar.GlUtil

/** 描画できる形にした補正済みの面。 */
internal class DrawableSurface(
    /** 同じ平面なら補正をまたいでも変わらない番号。記録したデータで同じ壁を追うのに使う。 */
    val id: Int,
    val kind: SurfaceKind,
    val normal: FloatArray,
    val triangles: FloatBuffer,
    val triangleVertices: Int,
    val outline: FloatBuffer,
    val outlineVertices: Int,
    /** 基準にした平面や区画の今の姿勢。GL スレッドで更新される。 */
    private val reference: FloatArray,
    /** 補正した時点の基準の逆行列。 */
    private val referenceAtFitInverse: FloatArray,
    val fromPoints: Boolean,
    /** 補正した時点のワールド座標での面の座標系。 */
    val basis: SurfaceBasis,
) {
    /** 補正した後に ARCore が直した分を反映する行列を [out] に書く。 */
    fun writeModel(out: FloatArray) = Matrix.multiplyMM(out, 0, reference, 0, referenceAtFitInverse, 0)

    /** 今のワールド座標での面の座標系。GL スレッドで呼ぶ。 */
    fun currentBasis(): SurfaceBasis {
        val m = FloatArray(16)
        writeModel(m)
        fun point(p: FloatArray) = FloatArray(3) { i -> m[i] * p[0] + m[4 + i] * p[1] + m[8 + i] * p[2] + m[12 + i] }
        fun direction(d: FloatArray) = FloatArray(3) { i -> m[i] * d[0] + m[4 + i] * d[1] + m[8 + i] * d[2] }
        return SurfaceBasis(
            point(basis.origin), direction(basis.axisU), direction(basis.axisV),
            basis.uMin, basis.uMax, basis.vMin, basis.vMax,
        )
    }
}

internal class FusedSurfaces(
    val surfaces: List<DrawableSurface> = emptyList(),
    val covered: Set<SurfacePatch> = emptySet(),
)

/**
 * ARCore の平面を深度の点で補正する処理を、描画を止めないよう別スレッドで回す。
 *
 * 点から見つけた平面は ARCore が追跡してくれないので、近くの区画に結びつけて補正に追従させる。
 */
internal class SurfaceFusion {
    @Volatile var latest = FusedSurfaces()
        private set

    private val worker = Executors.newSingleThreadExecutor { Thread(it, "surface-fit") }
    private val busy = AtomicBoolean(false)
    @Volatile private var nextFitMillis = 0L

    // ワーカースレッドのみが触る。
    private val fitter = SurfaceFitter()
    private val foundPlanes = ArrayList<FoundPlane>()
    private var points = FloatArray(3 * 100_000)
    private var farPoints = FloatArray(3 * 100_000)

    /** GL スレッドから呼ぶ。前回の補正が終わっていなければ何もしない。 */
    fun requestFit(patches: Collection<SurfacePatch>, chunks: List<MapChunk>, deviceY: Float) {
        val now = SystemClock.elapsedRealtime()
        if (now < nextFitMillis || !busy.compareAndSet(false, true)) return
        val patchSnapshots = patches.filter { it.vertexCount >= 3 }.map { PatchSnapshot(it) }
        val chunkSnapshots = chunks.map { ChunkSnapshot(it, it.vertices, it.evidenceVertices, it.worldFromChunk.copyOf()) }
        worker.execute {
            val start = SystemClock.elapsedRealtime()
            try {
                run(patchSnapshots, chunkSnapshots, deviceY + CEILING_ABOVE_DEVICE_M)
            } catch (e: RuntimeException) {
                Log.e(TAG, "Surface fit failed", e)
            } finally {
                val elapsed = SystemClock.elapsedRealtime() - start
                // 店が広くなって処理が重くなっても、1 コアを占有し続けないようにする。
                nextFitMillis = SystemClock.elapsedRealtime() + maxOf(MIN_FIT_INTERVAL_MS, elapsed * FIT_IDLE_RATIO)
                busy.set(false)
            }
        }
    }

    fun close() {
        worker.shutdown()
    }

    private fun run(patches: List<PatchSnapshot>, chunks: List<ChunkSnapshot>, ceilingY: Float) {
        val start = SystemClock.elapsedRealtime()
        val count = collectPoints(chunks) { it.vertices }
        val farCount = collectFarPoints(chunks)
        val foundSpecs = foundPlanes.mapNotNull { plane ->
            chunks.firstOrNull { it.chunk === plane.chunk }?.let { plane to plane.toSpec(it.matrix) }
        }
        val specs = patches.map { it.spec } + foundSpecs.map { it.second }
        val result = fitter.fit(points, count, specs, ceilingY, farPoints, farCount)

        for ((plane, spec) in foundSpecs) {
            val surface = result.surfaces.firstOrNull { spec in it.members }
            when {
                surface == null -> plane.misses++
                surface.reference !== spec -> foundPlanes.remove(plane) // ARCore が同じ面を見つけた
                else -> {
                    plane.misses = 0
                    plane.area = surface.area
                }
            }
        }
        foundPlanes.removeAll { it.misses >= MAX_MISSES }
        for (spec in result.discovered) foundPlanes += FoundPlane.attach(spec, chunks)

        val patchOf = patches.associateBy { it.spec }
        val chunkOf = foundSpecs.associate { (plane, spec) -> spec to chunks.first { it.chunk === plane.chunk } }
        val drawn = result.surfaces.mapNotNull { surface ->
            val patch = patchOf[surface.reference]
            val chunk = chunkOf[surface.reference]
            when {
                patch != null -> drawable(surface, patch.patch.worldFromPlane, patch.pose, false, System.identityHashCode(patch.patch))
                chunk != null -> drawable(
                    surface, chunk.chunk.worldFromChunk, chunk.matrix, true,
                    foundSpecs.firstOrNull { it.second === surface.reference }?.let { System.identityHashCode(it.first) } ?: 0,
                )
                else -> null
            }
        }
        latest = FusedSurfaces(
            drawn,
            result.surfaces.flatMap { it.members }.mapNotNull { patchOf[it]?.patch }.toSet(),
        )
        Log.i(
            TAG,
            "fit points=$count far=$farCount detected=${patches.size} found=${foundPlanes.size} surfaces=${drawn.size} " +
                "areaM2=${result.surfaces.sumOf { it.area.toDouble() }.toInt()} ms=${SystemClock.elapsedRealtime() - start}",
        )
    }

    private fun collectFarPoints(chunks: List<ChunkSnapshot>): Int {
        val swap = points
        points = farPoints
        val count = collectPoints(chunks) { it.evidence }
        farPoints = points
        points = swap
        return count
    }

    private fun collectPoints(chunks: List<ChunkSnapshot>, verticesOf: (ChunkSnapshot) -> IntArray): Int {
        val total = chunks.sumOf { verticesOf(it).size / VoxelGrid.INTS_PER_VERTEX }
        if (points.size < total * 3) points = FloatArray(total * 3 + total / 2)
        var out = 0
        for (snapshot in chunks) {
            val m = snapshot.matrix
            val v = verticesOf(snapshot)
            var i = 0
            while (i < v.size) {
                val x = Float.fromBits(v[i])
                val y = Float.fromBits(v[i + 1])
                val z = Float.fromBits(v[i + 2])
                points[out++] = m[0] * x + m[4] * y + m[8] * z + m[12]
                points[out++] = m[1] * x + m[5] * y + m[9] * z + m[13]
                points[out++] = m[2] * x + m[6] * y + m[10] * z + m[14]
                i += VoxelGrid.INTS_PER_VERTEX
            }
        }
        return out / 3
    }

    private fun drawable(
        surface: FittedSurface,
        reference: FloatArray,
        atFit: FloatArray,
        fromPoints: Boolean,
        id: Int,
    ): DrawableSurface? {
        val inverse = FloatArray(16)
        if (!Matrix.invertM(inverse, 0, atFit, 0)) return null
        return DrawableSurface(
            id, surface.kind, surface.normal,
            GlUtil.floatBuffer(surface.triangles), surface.triangles.size / 3,
            GlUtil.floatBuffer(surface.outline), surface.outline.size / 3,
            reference, inverse, fromPoints, surface.basis,
        )
    }

    private class ChunkSnapshot(val chunk: MapChunk, val vertices: IntArray, val evidence: IntArray, val matrix: FloatArray)

    /** GL スレッドで作る。ARCore のオブジェクトをワーカースレッドで触らないようにする。 */
    private class PatchSnapshot(val patch: SurfacePatch) {
        val pose: FloatArray = patch.worldFromPlane.copyOf()
        val spec = PlaneSpec(
            origin = floatArrayOf(pose[12], pose[13], pose[14]),
            normal = floatArrayOf(pose[4], pose[5], pose[6]),
            kind = patch.kind,
            polygon = patch.polygon.let { p ->
                FloatArray(p.size / 2 * 3).also { out ->
                    for (i in 0 until p.size / 2) {
                        val x = p[i * 2]
                        val z = p[i * 2 + 1]
                        out[i * 3] = pose[0] * x + pose[8] * z + pose[12]
                        out[i * 3 + 1] = pose[1] * x + pose[9] * z + pose[13]
                        out[i * 3 + 2] = pose[2] * x + pose[10] * z + pose[14]
                    }
                }
            },
            area = patch.area,
        )
    }

    /** 点から見つけた平面。原点と法線は区画のローカル座標で持つ。 */
    private class FoundPlane(
        val chunk: MapChunk,
        val origin: FloatArray,
        val normal: FloatArray,
        val kind: SurfaceKind,
        var area: Float,
        val fromFarPoints: Boolean,
    ) {
        var misses = 0

        fun toSpec(m: FloatArray) = PlaneSpec(
            floatArrayOf(
                m[0] * origin[0] + m[4] * origin[1] + m[8] * origin[2] + m[12],
                m[1] * origin[0] + m[5] * origin[1] + m[9] * origin[2] + m[13],
                m[2] * origin[0] + m[6] * origin[1] + m[10] * origin[2] + m[14],
            ),
            floatArrayOf(
                m[0] * normal[0] + m[4] * normal[1] + m[8] * normal[2],
                m[1] * normal[0] + m[5] * normal[1] + m[9] * normal[2],
                m[2] * normal[0] + m[6] * normal[1] + m[10] * normal[2],
            ),
            kind, null, area, fromFarPoints,
        )

        companion object {
            /** いちばん近い区画に結びつける。区画の変換は回転と平行移動だけなので、転置で逆変換できる。 */
            fun attach(spec: PlaneSpec, chunks: List<ChunkSnapshot>): FoundPlane {
                val o = spec.origin
                val nearest = chunks.minBy { c ->
                    val m = c.matrix
                    val dx = m[12] - o[0]
                    val dy = m[13] - o[1]
                    val dz = m[14] - o[2]
                    dx * dx + dy * dy + dz * dz
                }
                val m = nearest.matrix
                val dx = o[0] - m[12]
                val dy = o[1] - m[13]
                val dz = o[2] - m[14]
                val n = spec.normal
                return FoundPlane(
                    nearest.chunk,
                    floatArrayOf(
                        m[0] * dx + m[1] * dy + m[2] * dz,
                        m[4] * dx + m[5] * dy + m[6] * dz,
                        m[8] * dx + m[9] * dy + m[10] * dz,
                    ),
                    floatArrayOf(
                        m[0] * n[0] + m[1] * n[1] + m[2] * n[2],
                        m[4] * n[0] + m[5] * n[1] + m[6] * n[2],
                        m[8] * n[0] + m[9] * n[1] + m[10] * n[2],
                    ),
                    spec.kind,
                    spec.area,
                    spec.fromFarPoints,
                )
            }
        }
    }

    private companion object {
        const val TAG = "SurfaceFusion"
        const val MIN_FIT_INTERVAL_MS = 1500L
        const val FIT_IDLE_RATIO = 3L

        /** 天井の点を面の材料に入れない高さ。胸の高さの端末から 1m 上まで使う。 */
        const val CEILING_ABOVE_DEVICE_M = 1.0f

        /** 位置の補正で一時的に点が外れても、すぐには消さない。 */
        const val MAX_MISSES = 3
    }
}
