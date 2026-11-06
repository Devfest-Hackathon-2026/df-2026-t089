package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.ProductRecognizer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ProductSpreadTest {
    /** 中心が (x, y) の小さな枠。 */
    private fun box(x: Float, y: Float) = floatArrayOf(x - 0.02f, y - 0.02f, x + 0.02f, y + 0.02f)

    @Test
    fun `boxes clustered in one corner do not take all the slots`() {
        // 確からしい 12 個は左上に固まり、その後に右下と左下に 1 つずつ
        val clustered = (0 until 12).map { box(0.05f + it * 0.01f, 0.1f) }
        val far = listOf(box(0.9f, 0.9f), box(0.1f, 0.9f))
        val picked = ProductRecognizer.spread(clustered + far)
        assertEquals(ProductRecognizer.MAX_CROPS, picked.size)
        assertTrue(far.all { f -> picked.any { it === f } })
    }

    @Test
    fun `every occupied cell is taken even past the limit`() {
        val grid = (0 until 4).flatMap { r -> (0 until 4).map { c -> box(c * 0.25f + 0.125f, r * 0.25f + 0.125f) } }
        assertEquals(16, ProductRecognizer.spread(grid).size)
    }

    @Test
    fun `keeps confidence order and takes all when few`() {
        val boxes = listOf(box(0.9f, 0.9f), box(0.1f, 0.1f), box(0.12f, 0.1f))
        assertEquals(boxes, ProductRecognizer.spread(boxes))
    }
}
