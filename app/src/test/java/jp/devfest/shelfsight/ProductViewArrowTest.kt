package jp.devfest.shelfsight

import java.nio.FloatBuffer
import jp.devfest.shelfsight.assist.ProductViewArrow
import jp.devfest.shelfsight.assist.ShelfMap
import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Test

class ProductViewArrowTest {
    @Test
    fun `sight line runs from where the product was seen to its point`() {
        val point = Vec3(2f, 1f, -3f)
        val vertices = FloatBuffer.allocate(ProductViewArrow.FLOATS)
        ProductViewArrow.append(point, Vec3(2f, 1.4f, 0f), vertices)
        assertEquals(ProductViewArrow.FLOATS, vertices.position())
        assertEquals(1.4f, vertices.get(1), 1e-5f)
        assertEquals(0f, vertices.get(2), 1e-5f)
        assertEquals(point.x, vertices.get(3), 1e-5f)
        assertEquals(point.y, vertices.get(4), 1e-5f)
        assertEquals(point.z, vertices.get(5), 1e-5f)
    }

    @Test
    fun `new sighting updates where it was seen without changing an earlier render snapshot`() {
        val map = ShelfMap()
        val p = Vec3(1f, 1f, 2f)
        map.add("product", null, null, p, 0.1f, 0.2f, 1L, Vec3(0f, 1f, 0f))
        val before = map.snapshot().single()
        map.add("product", null, null, p, 0.1f, 0.2f, 2L, Vec3(3f, 1f, 2f))
        val after = map.snapshot().single()
        assertEquals(3f, after.cameraAt!!.x, 1e-5f)
        assertEquals(0f, before.cameraAt!!.x, 1e-5f)
    }
}
