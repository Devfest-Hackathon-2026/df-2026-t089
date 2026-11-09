package jp.devfest.shelfsight.assist

import org.junit.Assert.assertEquals
import org.junit.Test

class StoreLabelsTest {
    private fun fixture(axis: Int, faces: Map<String, Int>, themes: Map<String, List<String?>>) = StoreFixture(
        "island", axis, 5.0, 1.0, 4.0, 1.0, 1.5, 3, 1.0,
        faces.mapValues { (_, n) -> List(n) { listOf("cat$it" to 1.0) } }, themes,
    )

    @Test fun `consecutive modules with one genre become one centred label`() {
        val f = fixture(0, mapOf("+" to 3), mapOf("+" to listOf("drinks", "drinks", "bread")))
        val out = StoreLabels.shelves(StoreView(0.0, doubleArrayOf(0.0, 0.0, 9.0, 9.0), listOf(f)))
        assertEquals(listOf("drinks", "bread"), out.map { it.text })
        assertEquals(2.0, out[0].v, 1e-9)   // modules 0 and 1 span v = 1..3
        assertEquals(3.5, out[1].v, 1e-9)
        assertEquals(5.0, out[0].u, 1e-9)   // one face stays on the centre line
        assertEquals(1.5, out[0].top, 1e-9)
    }

    @Test fun `two faces sit on their own sides and axis 1 swaps u and v`() {
        val themes = mapOf("+" to listOf("a", "a", "a"), "-" to listOf("b", "b", "b"))
        val out = StoreLabels.shelves(StoreView(0.0, null, listOf(fixture(1, mapOf("+" to 3, "-" to 3), themes))))
        assertEquals(listOf("a", "b"), out.map { it.text })
        assertEquals(2.5, out[0].u, 1e-9)
        assertEquals(5.25, out[0].v, 1e-9)
        assertEquals(4.75, out[1].v, 1e-9)
    }

    @Test fun `a module without a theme shows its top category and empty faces show nothing`() {
        val f = fixture(0, mapOf("+" to 3), mapOf("+" to listOf(null, null, "bread")))
        assertEquals(listOf("cat0", "cat1", "bread"), StoreLabels.shelves(StoreView(0.0, null, listOf(f))).map { it.text })
        val empty = StoreFixture("register", 0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1.0, emptyMap())
        assertEquals(0, StoreLabels.shelves(StoreView(0.0, null, listOf(empty))).size)
    }
}
