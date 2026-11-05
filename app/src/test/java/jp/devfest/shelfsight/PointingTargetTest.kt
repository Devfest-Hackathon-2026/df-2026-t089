package jp.devfest.shelfsight

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.DepthSnapshot
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.perception.PointingTarget
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class PointingTargetTest {
    private val identity = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)

    private fun geometry(depthMm: Int?, rotation: Int = 0) = CaptureGeometry(
        imageWidth = 640, imageHeight = 480,
        focalX = 500f, focalY = 500f, principalX = 320f, principalY = 240f,
        cameraToWorld = identity,
        uprightRotationDegrees = rotation,
        depth = depthMm?.let { mm -> DepthSnapshot(160, 120, ShortArray(160 * 120) { mm.toShort() }) },
        depthU0 = 0f, depthUPerPixel = 1f / 640f, depthV0 = 0f, depthVPerPixel = 1f / 480f,
    )

    @Test
    fun `pointing straight ahead hits the wall in front`() {
        val hit = PointingTarget.find(Vec3(0f, 0f, -0.4f), Vec3(0f, 0f, -1f), geometry(depthMm = 2000))
        assertNotNull(hit)
        assertEquals(0.5f, hit!!.u, 0.01f)
        assertEquals(0.5f, hit.v, 0.01f)
        assertEquals(2f, hit.depthM, 0.01f)
    }

    @Test
    fun `pointing right moves the target to the right of the image`() {
        val direction = Vec3(0.2f, 0f, -0.98f)
        val hit = PointingTarget.find(Vec3(0f, 0f, -0.4f), direction, geometry(depthMm = 2000))!!
        assertEquals(0.5f + 500f * (0.2f * 1.63f) / 2f / 640f, hit.u, 0.03f)
        assertEquals(0.5f, hit.v, 0.01f)
    }

    @Test
    fun `depth of the hand itself is not a target`() {
        assertNull(PointingTarget.find(Vec3(0f, 0f, -0.4f), Vec3(0f, 0f, -1f), geometry(depthMm = 450)))
    }

    @Test
    fun `no depth means no target`() {
        assertNull(PointingTarget.find(Vec3(0f, 0f, -0.4f), Vec3(0f, 0f, -1f), geometry(depthMm = null)))
    }

    @Test
    fun `pointing out of the image gives no target`() {
        assertNull(PointingTarget.find(Vec3(0f, 0f, -0.4f), Vec3(1f, 0f, 0f), geometry(depthMm = 2000)))
    }

    @Test
    fun `pointing backward gives no target`() {
        assertNull(PointingTarget.find(Vec3(0f, 0f, -0.1f), Vec3(0f, 0f, 1f), geometry(depthMm = 2000)))
    }

    @Test
    fun `projection inverts unprojection for every rotation`() {
        for (rotation in listOf(0, 90, 180, 270)) {
            val g = geometry(depthMm = 2000, rotation = rotation)
            val world = Unprojector.worldAt(0.25f, 0.75f, 1.5f, g)
            val projected = PointingTarget.project(world, g)!!
            assertEquals("u at $rotation", 0.25f, projected.u, 0.001f)
            assertEquals("v at $rotation", 0.75f, projected.v, 0.001f)
            assertEquals("depth at $rotation", 1.5f, projected.depthM, 0.001f)
        }
    }
}
