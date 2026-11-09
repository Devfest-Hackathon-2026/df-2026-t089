package jp.devfest.shelfsight

import java.nio.FloatBuffer
import jp.devfest.shelfsight.assist.ProductViewArrow
import jp.devfest.shelfsight.assist.ShelfMap
import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Test

class ProductViewArrowTest {
    @Test
    fun `product arrow ends at the measured point and follows camera forward`() {
        val point = Vec3(2f, 1f, -3f)
        val vertices = FloatBuffer.allocate(ProductViewArrow.FLOATS)
        ProductViewArrow.append(point, Vec3(0f, 0f, -1f), vertices)
        assertEquals(ProductViewArrow.FLOATS, vertices.position())
        assertEquals(2f, vertices.get(0), 1e-5f)
        assertEquals(1f, vertices.get(1), 1e-5f)
        assertEquals(-2.62f, vertices.get(2), 1e-5f) // tail is behind the product
        assertEquals(point.x, vertices.get(3), 1e-5f)
        assertEquals(point.y, vertices.get(4), 1e-5f)
        assertEquals(point.z, vertices.get(5), 1e-5f)
    }

    @Test
    fun `new sighting updates the direction without changing an earlier render snapshot`() {
        val map = ShelfMap()
        val p = Vec3(1f, 1f, 2f)
        map.add("product", null, null, p, 0.1f, 0.2f, 1L, Vec3(0f, -0.4f, -2f))
        val before = map.snapshot().single()
        assertEquals(-1f, before.viewForward!!.z, 1e-5f)
        map.add("product", null, null, p, 0.1f, 0.2f, 2L, Vec3(3f, 0f, 0f))
        val after = map.snapshot().single()
        assertEquals(1f, after.viewForward!!.x, 1e-5f)
        assertEquals(-1f, before.viewForward!!.z, 1e-5f)
    }
}
