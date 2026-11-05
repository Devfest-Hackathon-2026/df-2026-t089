package jp.devfest.shelfsight

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.DepthSnapshot
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.NormalizedBox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UnprojectorTest {
    private val identity = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)
    private val center = NormalizedBox(0.4f, 0.4f, 0.6f, 0.6f)

    private fun geometry(
        depth: DepthSnapshot? = null,
        rotation: Int = 0,
        cameraToWorld: FloatArray = identity,
    ) = CaptureGeometry(
        imageWidth = 1000, imageHeight = 500,
        focalX = 500f, focalY = 500f, principalX = 500f, principalY = 250f,
        cameraToWorld = cameraToWorld,
        uprightRotationDegrees = rotation,
        depth = depth,
        depthU0 = 0f, depthUPerPixel = 1f / 1000f, depthV0 = 0f, depthVPerPixel = 1f / 500f,
    )

    private fun flatDepth(mm: Int) = DepthSnapshot(100, 50, ShortArray(100 * 50) { mm.toShort() })

    @Test
    fun `image center with depth lands straight ahead at that distance`() {
        val result = Unprojector.unproject(center, geometry(flatDepth(2000)))
        assertTrue(result.hasMeasuredDepth)
        assertEquals(0f, result.position.x, 1e-3f)
        assertEquals(0f, result.position.y, 1e-3f)
        assertEquals(-2f, result.position.z, 1e-3f)
    }

    @Test
    fun `missing depth falls back to assumed distance and is flagged`() {
        val result = Unprojector.unproject(center, geometry(depth = null))
        assertFalse(result.hasMeasuredDepth)
        assertEquals(-1f, result.position.z, 1e-3f)
    }

    @Test
    fun `invalid depth pixels are ignored`() {
        assertFalse(Unprojector.unproject(center, geometry(flatDepth(0))).hasMeasuredDepth)
    }

    @Test
    fun `depths above 32767 mm are read as unsigned and rejected by range`() {
        assertFalse(Unprojector.unproject(center, geometry(flatDepth(40000))).hasMeasuredDepth)
    }

    @Test
    fun `upper right of image maps to positive x and y in camera space`() {
        val box = NormalizedBox(0.8f, 0.0f, 1.0f, 0.2f) // センサー座標 (900, 50)
        val p = Unprojector.unproject(box, geometry(flatDepth(1000))).position
        assertEquals(0.8f, p.x, 1e-3f)
        assertEquals(0.4f, p.y, 1e-3f)
        assertEquals(-1f, p.z, 1e-3f)
    }

    @Test
    fun `camera translation is applied`() {
        val moved = identity.copyOf().also { it[12] = 1f; it[13] = 2f; it[14] = 3f }
        val p = Unprojector.unproject(center, geometry(flatDepth(1000), cameraToWorld = moved)).position
        assertEquals(1f, p.x, 1e-3f)
        assertEquals(2f, p.y, 1e-3f)
        assertEquals(2f, p.z, 1e-3f)
    }

    @Test
    fun `upright to sensor inverts each clockwise rotation`() {
        // 時計回り 90 度: センサーの左上は正立画像の右上になる
        assertEquals(0f to 0f, Unprojector.uprightToSensor(1f, 0f, 90))
        // 180 度: センサーの左上は正立画像の右下になる
        assertEquals(0f to 0f, Unprojector.uprightToSensor(1f, 1f, 180))
        // 時計回り 270 度: センサーの左上は正立画像の左下になる
        assertEquals(0f to 0f, Unprojector.uprightToSensor(0f, 1f, 270))
        assertEquals(0.25f to 0.75f, Unprojector.uprightToSensor(0.25f, 0.75f, 0))
    }
}
