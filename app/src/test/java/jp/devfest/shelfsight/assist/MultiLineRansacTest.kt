package jp.devfest.shelfsight.assist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.hypot

class MultiLineRansacTest {
    private val rng = Random(3)
    private fun jitter() = rng.nextGaussian() * 0.02
    private fun length(l: DoubleArray) = hypot(l[2] - l[0], l[3] - l[1])

    @Test fun `a long face becomes several segments shorter than one metre`() {
        val face = (0 until 60).map { doubleArrayOf(it * 0.1, 2.0 + jitter()) }
        val lines = MultiLineRansac.fit(face)
        assertTrue(lines.size >= 5)
        assertTrue(lines.all { length(it) < 1.0 })
        assertTrue(lines.all { abs(it[1] - 2.0) < 0.1 && abs(it[3] - 2.0) < 0.1 })
        assertTrue(lines.sumOf { length(it) } > 3.5)
    }

    @Test fun `both faces of a corner are found`() {
        val along = (0 until 30).map { doubleArrayOf(it * 0.1, 0.0 + jitter()) }
        val across = (0 until 30).map { doubleArrayOf(3.0 + jitter(), it * 0.1 + 0.3) }
        val lines = MultiLineRansac.fit(along + across)
        assertTrue(lines.any { abs(it[1] - it[3]) < 0.1 })
        assertTrue(lines.any { abs(it[0] - it[2]) < 0.1 })
    }

    @Test fun `sparse points and scatter give no segment, and the same seed repeats`() {
        assertEquals(0, MultiLineRansac.fit((0 until 5).map { doubleArrayOf(it * 0.1, 0.0) }).size)
        val scatter = (0 until 200).map { doubleArrayOf(rng.nextDouble() * 8, rng.nextDouble() * 8) }
        assertTrue(MultiLineRansac.fit(scatter).size <= 2)
        val pts = (0 until 30).map { doubleArrayOf(it * 0.1, 1.0 + jitter()) }
        assertEquals(MultiLineRansac.fit(pts).map { it.toList() }, MultiLineRansac.fit(pts).map { it.toList() })
    }
}
