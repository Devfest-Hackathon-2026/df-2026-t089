package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp
import kotlin.math.ln

class NeighborConfidenceTest {
    private fun top(a: String, b: String, p: Float) = listOf(a to p, b to 1 - p)
    private fun sigmoid(z: Double) = (1 / (1 + exp(-z))).toFloat()

    /** neighbor_confidence.py の self_check と同じ並び。 */
    @Test fun `features match the python script`() {
        val c = NeighborConfidence(floatArrayOf(1f, 2f, 3f, 4f), 0.5f, 4, 3f)
        // 周りがない: logit(0.5) = 0 なので intercept だけ
        assertEquals(sigmoid(0.5), c.adjust(Vec3(0f, 0f, 0f), 1f, top("A", "B", 0.5f)), 1e-5f)
        c.adjust(Vec3(1f, 0f, 0f), 1f, top("B", "A", 0.6f))
        // 周りは A(.5) と B（A が .4）: 一致の割合 .5、A の平均確率 .45
        val z = 1.0 * ln(0.8 / 0.2) + 2 * 1 + 3 * 0.5 + 4 * 0.45 + 0.5
        assertEquals(sigmoid(z), c.adjust(Vec3(0.5f, 0f, 0f), 1f, top("A", "B", 0.8f)), 1e-4f)
    }

    @Test fun `far points and the same spot are not neighbours`() {
        val c = NeighborConfidence(floatArrayOf(0f, 1f, 0f, 0f), 0f, 4, 3f)
        c.adjust(Vec3(0f, 0f, 0f), 0.1f, top("A", "B", 0.9f))
        assertEquals(0.5f, c.adjust(Vec3(5f, 0f, 0f), 0.1f, top("A", "B", 0.9f)), 1e-5f)
        assertEquals(0.5f, c.adjust(Vec3(5.05f, 0f, 0f), 0.1f, top("A", "B", 0.9f)), 1e-5f)
        assertTrue(c.adjust(Vec3(5.2f, 0f, 0f), 0.1f, top("A", "B", 0.9f)) > 0.7f)
    }

    @Test fun `in one photo the neighbours are boxes within three long sides`() {
        val c = NeighborConfidence(floatArrayOf(1f, 2f, 3f, 4f), 0.5f, 4, 3f)
        val tops = listOf(top("A", "B", 0.5f), top("A", "B", 0.8f), top("B", "A", 0.6f))
        val s = c.inImage(listOf(0f to 0f, 1f to 0f, 2f to 0f), listOf(2f, 2f, 2f), tops)
        // 1 枚目の周りは A(.8) と B（A が .4）: logit 0、一致 .5、A の平均 .6
        assertEquals(sigmoid(2 + 3 * 0.5 + 4 * 0.6 + 0.5), s[0], 1e-4f)
        assertEquals(sigmoid(0.5), c.inImage(listOf(0f to 0f), listOf(1f), tops.take(1))[0], 1e-5f)
    }

    @Test fun `without weights the first probability is kept`() {
        assertEquals(0.83f, NeighborConfidence.parse(null).adjust(Vec3(0f, 0f, 0f), 1f, top("A", "B", 0.83f)), 0f)
    }
}
