package jp.devfest.shelfsight

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.mapping.DepthFusion
import jp.devfest.shelfsight.mapping.DepthSample
import jp.devfest.shelfsight.mapping.OrbitCamera
import jp.devfest.shelfsight.mapping.VoxelGrid
import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MappingLogicTest {
    private val red = 0xFF0000FF.toInt()

    private fun VoxelGrid.positions(): List<Vec3> {
        val v = toVertices()
        return (0 until v.size / VoxelGrid.INTS_PER_VERTEX).map { i ->
            Vec3(Float.fromBits(v[i * 4]), Float.fromBits(v[i * 4 + 1]), Float.fromBits(v[i * 4 + 2]))
        }
    }

    private fun assertNear(expected: Vec3, actual: Vec3, tolerance: Float = 0.03f) {
        assertTrue("expected $expected but was $actual", expected.distanceTo(actual) <= tolerance)
    }

    @Test
    fun `a point seen once is treated as noise`() {
        val grid = VoxelGrid()
        grid.add(1f, 1f, 1f, red)
        assertEquals(1, grid.size)
        assertEquals(0, grid.confirmedCount)
        assertEquals(0, grid.toVertices().size)
    }

    @Test
    fun `a point seen twice in one voxel becomes one vertex at the voxel center`() {
        val grid = VoxelGrid(voxelSize = 0.1f)
        grid.add(0.21f, -0.11f, 0.01f, red)
        grid.add(0.29f, -0.19f, 0.09f, red)
        assertEquals(1, grid.confirmedCount)
        assertNear(Vec3(0.25f, -0.15f, 0.05f), grid.positions().single(), 0.001f)
    }

    @Test
    fun `negative and positive coordinates do not collide`() {
        val grid = VoxelGrid()
        repeat(2) {
            grid.add(0.02f, 0f, 0f, red)
            grid.add(-0.02f, 0f, 0f, red)
        }
        assertEquals(2, grid.confirmedCount)
    }

    @Test
    fun `all voxels survive table growth and colors stay opaque`() {
        val grid = VoxelGrid()
        val count = 50_000
        repeat(2) {
            for (i in 0 until count) grid.add(i * 0.05f + 0.01f, 0.01f, -i * 0.05f - 0.01f, red)
        }
        assertEquals(count, grid.size)
        assertEquals(count, grid.confirmedCount)
        val vertices = grid.toVertices()
        assertEquals(count * VoxelGrid.INTS_PER_VERTEX, vertices.size)
        assertEquals(red, vertices[3])
    }

    @Test
    fun `repeated observations blend color without overflowing channels`() {
        val grid = VoxelGrid()
        val white = 0xFFFFFFFF.toInt()
        repeat(20) { grid.add(0f, 0f, 0f, white) }
        assertEquals(white, grid.toVertices()[3] or 0x03030303)
    }

    private val identity = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)

    /** 深度 2x2、カメラ画像 100x100。深度画像がカメラ画像全体に対応する。 */
    private fun sample(
        depth: Int,
        confidence: Int = 255,
        chunkFromCamera: FloatArray = identity,
    ) = DepthSample(
        depthWidth = 2,
        depthHeight = 2,
        depthMm = ShortArray(4) { depth.toShort() },
        confidence = ByteArray(4) { confidence.toByte() },
        nv21 = ByteArray(100 * 100 * 3 / 2) { 128.toByte() },
        geometry = CaptureGeometry(
            imageWidth = 100, imageHeight = 100,
            focalX = 100f, focalY = 100f, principalX = 50f, principalY = 50f,
            cameraToWorld = identity,
            uprightRotationDegrees = 0,
            depth = null,
            depthU0 = 0f, depthUPerPixel = 0.01f, depthV0 = 0f, depthVPerPixel = 0.01f,
        ),
        chunkFromCamera = chunkFromCamera,
    )

    private fun fused(vararg samples: DepthSample) =
        VoxelGrid().also { grid -> samples.forEach { DepthFusion.integrate(it, grid) } }

    @Test
    fun `depth pixels unproject in front of the camera with image up as plus y`() {
        // 深度画像の左上ピクセル = カメラ画像 (25, 25)
        val points = fused(sample(2000), sample(2000)).positions()
        assertEquals(4, points.size)
        assertTrue(points.any { it.distanceTo(Vec3(-0.5f, 0.5f, -2f)) < 0.05f })
        assertTrue(points.all { it.z < 0f })
    }

    @Test
    fun `chunk transform is applied to fused points`() {
        val moved = identity.copyOf().also { it[12] = 10f }
        val points = fused(sample(2000, chunkFromCamera = moved), sample(2000, chunkFromCamera = moved)).positions()
        assertTrue(points.any { it.distanceTo(Vec3(9.5f, 0.5f, -2f)) < 0.05f })
    }

    @Test
    fun `low confidence and out of range depth are rejected`() {
        assertEquals(0, fused(sample(2000, confidence = 50)).size)
        assertEquals(0, fused(sample(0)).size)
        assertEquals(0, fused(sample(100)).size)
        assertEquals(0, fused(sample(5000)).size)
        assertEquals(0, fused(sample(40000)).size) // 符号なしとして読めていれば範囲外
    }

    @Test
    fun `orbit camera starts above target and keeps distance when rotating`() {
        val orbit = OrbitCamera()
        orbit.fit(Vec3(1f, 0f, 2f), radius = 5f)
        val before = orbit.eye()
        assertTrue(before.y > 0f)
        orbit.rotate(300f, 0f)
        assertEquals(before.distanceTo(orbit.center()), orbit.eye().distanceTo(orbit.center()), 1e-3f)
    }

    @Test
    fun `orbit pan drags the map with the finger and stays on the ground plane`() {
        val orbit = OrbitCamera()
        orbit.fit(Vec3(0f, 0f, 0f), radius = 5f)
        // yaw 0 では画面右が +X。指を右へ動かすと地図が右へ動くので、注視点は左へ動く。
        orbit.pan(100f, 0f)
        assertTrue(orbit.center().x < 0f)
        assertEquals(0f, orbit.center().y, 0f)
    }

    @Test
    fun `orbit zoom in moves the camera closer and ignores invalid factors`() {
        val orbit = OrbitCamera()
        orbit.fit(Vec3(0f, 0f, 0f), radius = 5f)
        val before = orbit.eye().distanceTo(orbit.center())
        orbit.zoom(2f)
        val zoomed = orbit.eye().distanceTo(orbit.center())
        assertEquals(before / 2f, zoomed, 1e-3f)
        orbit.zoom(0f)
        assertEquals(zoomed, orbit.eye().distanceTo(orbit.center()), 1e-6f)
    }
}

class MappingHistoryTest {
    private val millis = 1_000_000L

    @org.junit.Test
    fun `pose history returns the pose closest to the depth timestamp`() {
        val history = jp.devfest.shelfsight.mapping.PoseHistory<String>()
        history.record(100 * millis, "a")
        history.record(133 * millis, "b")
        history.record(166 * millis, "c")
        assertEquals("b", history.nearest(133 * millis, 20 * millis))
        assertEquals("b", history.nearest(140 * millis, 20 * millis))
    }

    @org.junit.Test
    fun `pose history rejects timestamps with no pose nearby`() {
        val history = jp.devfest.shelfsight.mapping.PoseHistory<String>()
        assertEquals(null, history.nearest(0L, 20 * millis))
        history.record(1000 * millis, "a")
        assertEquals(null, history.nearest(900 * millis, 20 * millis))
    }

    @org.junit.Test
    fun `pose history forgets the oldest pose and ignores repeated frames`() {
        val history = jp.devfest.shelfsight.mapping.PoseHistory<String>(capacity = 2)
        history.record(100 * millis, "a")
        history.record(200 * millis, "b")
        history.record(200 * millis, "duplicate")
        assertEquals("b", history.nearest(200 * millis, 0L))
        history.record(300 * millis, "c")
        assertEquals(null, history.nearest(100 * millis, 20 * millis))
    }

    @org.junit.Test
    fun `stats count separate stale episodes and report lag`() {
        val stats = jp.devfest.shelfsight.mapping.MappingStats()
        stats.onAccepted(30)
        stats.onAccepted(70)
        stats.onAccepted(70)
        stats.onStale(60_000)
        stats.onStale(60_500)
        stats.onAccepted(40)
        stats.onStale(9_000)
        assertEquals(70L, stats.medianLagMs())
        val summary = stats.summary()
        assertTrue(summary, "accepted=4 " in summary)
        assertTrue(summary, "stale=3 " in summary)
        assertTrue(summary, "staleEpisodes=2 " in summary)
        assertTrue(summary, "staleMaxMs=60500 " in summary)
    }
}
