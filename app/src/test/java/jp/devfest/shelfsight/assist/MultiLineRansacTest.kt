package jp.devfest.shelfsight.assist

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.hypot

class MultiLineRansacTest {
    private val rng = Random(3)
    private fun jitter() = rng.nextGaussian() * 0.02

    @Test fun `finds two perpendicular faces despite scattered outliers`() {
        val along = (0 until 40).map { doubleArrayOf(it * 0.1, 2.0 + jitter()) }
        val across = (0 until 30).map { doubleArrayOf(5.0 + jitter(), it * 0.1 + 3.0) }
        val noise = (0 until 10).map { doubleArrayOf(rng.nextDouble() * 8 - 1, rng.nextDouble() * 8 - 1 - 4) }
        val lines = MultiLineRansac.fit(along + across + noise)
        assertEquals(2, lines.size)
        val long = lines.first { abs(it[1] - it[3]) < 0.2 }
        assertEquals(2.0, (long[1] + long[3]) / 2, 0.05)
        assertEquals(3.9, hypot(long[2] - long[0], long[3] - long[1]), 0.2)
    }

    @Test fun `two separate shelves on one line stay two segments`() {
        val left = (0 until 20).map { doubleArrayOf(it * 0.1, 0.0) }
        val right = (0 until 20).map { doubleArrayOf(5.0 + it * 0.1, 0.0) }
        assertEquals(2, MultiLineRansac.fit(left + right).size)
    }

    @Test fun `too few points give no line and the same seed gives the same result`() {
        assertEquals(0, MultiLineRansac.fit((0 until 5).map { doubleArrayOf(it * 0.1, 0.0) }).size)
        val pts = (0 until 30).map { doubleArrayOf(it * 0.1, 1.0 + jitter()) }
        assertEquals(MultiLineRansac.fit(pts).single().toList(), MultiLineRansac.fit(pts).single().toList())
    }
}
