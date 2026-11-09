package jp.devfest.shelfsight

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.DepthSnapshot
import jp.devfest.shelfsight.assist.NearProduct
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class NearProductTest {
    private val identity = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)

    private fun geometry(depth: DepthSnapshot?) = CaptureGeometry(
        imageWidth = 1000, imageHeight = 500,
        focalX = 500f, focalY = 500f, principalX = 500f, principalY = 250f,
        cameraToWorld = identity,
        uprightRotationDegrees = 0,
        depth = depth,
        depthU0 = 0f, depthUPerPixel = 1f / 1000f, depthV0 = 0f, depthVPerPixel = 1f / 500f,
    )

    private fun depth(mm: (Float, Float) -> Int) = DepthSnapshot(100, 50, ShortArray(100 * 50) { i ->
        mm((i % 100 + 0.5f) / 100, (i / 100 + 0.5f) / 50).toShort()
    })

    @Test
    fun `no depth gives nothing`() {
        val r = NearProduct.measure(geometry(null))
        assertNull(r.pick)
        assertNull(r.centerM)
    }

    @Test
    fun `a far shelf is not near`() {
        val r = NearProduct.measure(geometry(depth { _, _ -> 800 }))
        assertNull(r.pick)
        assertEquals(0.8f, r.centerM!!, 1e-3f)
    }

    @Test
    fun `invalid depth in the center is not near`() {
        val r = NearProduct.measure(geometry(depth { _, _ -> 0 }))
        assertNull(r.pick)
        assertNull(r.centerM)
        assertEquals(0f, r.validFraction, 1e-6f)
    }

    @Test
    fun `a near flat shelf is cut to 30 cm square around the center`() {
        val r = NearProduct.measure(geometry(depth { _, _ -> 350 }))
        assertNotNull(r.pick)
        val p = r.pick!!
        assertEquals(0.35f, p.distanceM, 1e-3f)
        // 0.15 m / 0.35 m × 焦点 500 px / 幅 1000 px = 0.214、高さは / 500 px で 0.429
        assertEquals(0.5f - 0.214f, p.box[0], 0.01f)
        assertEquals(0.5f + 0.214f, p.box[2], 0.01f)
        assertEquals(0.5f - 0.429f, p.box[1], 0.01f)
        assertEquals(0.5f + 0.429f, p.box[3], 0.01f)
    }

    @Test
    fun `a product held out in front is cut to its own extent`() {
        // 中央の 0.3..0.7 × 0.3..0.7 だけが 0.3 m、まわりは 1 m
        val r = NearProduct.measure(geometry(depth { u, v -> if (u in 0.3f..0.7f && v in 0.3f..0.7f) 300 else 1000 }))
        val p = r.pick!!
        assertEquals(0.3f, p.distanceM, 1e-3f)
        assertEquals(0.3f, p.box[0], 0.04f)
        assertEquals(0.7f, p.box[2], 0.04f)
        assertEquals(0.3f, p.box[1], 0.04f)
        assertEquals(0.7f, p.box[3], 0.04f)
    }
}
