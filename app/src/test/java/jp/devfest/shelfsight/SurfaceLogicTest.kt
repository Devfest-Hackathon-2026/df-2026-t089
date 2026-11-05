package jp.devfest.shelfsight

import jp.devfest.shelfsight.mapping.SurfaceKind
import jp.devfest.shelfsight.mapping.SurfaceLogic
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SurfaceLogicTest {
    @Test
    fun `area of a rectangle does not depend on winding`() {
        val ccw = floatArrayOf(0f, 0f, 2f, 0f, 2f, 3f, 0f, 3f)
        val cw = floatArrayOf(0f, 3f, 2f, 3f, 2f, 0f, 0f, 0f)
        assertEquals(6f, SurfaceLogic.polygonArea(ccw), 1e-4f)
        assertEquals(6f, SurfaceLogic.polygonArea(cw), 1e-4f)
    }

    @Test
    fun `area of an empty polygon is zero`() {
        assertEquals(0f, SurfaceLogic.polygonArea(FloatArray(0)), 0f)
    }

    @Test
    fun `the floor is the lowest wide plane, not a small one below it`() {
        val floor = SurfaceLogic.floorHeight(listOf(-1.3f to 12f, -1.9f to 0.1f, -0.4f to 2f))
        assertEquals(-1.3f, floor!!, 1e-4f)
    }

    @Test
    fun `no floor is known until a wide plane is seen`() {
        assertNull(SurfaceLogic.floorHeight(listOf(-1.3f to 0.2f)))
    }

    @Test
    fun `planes near the floor are floor and higher ones are raised`() {
        assertEquals(SurfaceKind.FLOOR, SurfaceLogic.horizontalKind(-1.25f, -1.3f))
        assertEquals(SurfaceKind.RAISED, SurfaceLogic.horizontalKind(-0.5f, -1.3f))
        assertEquals(SurfaceKind.FLOOR, SurfaceLogic.horizontalKind(-0.5f, null))
    }
}
