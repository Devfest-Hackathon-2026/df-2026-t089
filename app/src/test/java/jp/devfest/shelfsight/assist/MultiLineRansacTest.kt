package jp.devfest.shelfsight.assist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

class MultiLineRansacTest {
    private val rng = Random(3)
    private fun jitter() = rng.nextGaussian() * 0.02
    private fun length(l: DoubleArray) = hypot(l[2] - l[0], l[3] - l[1])

    @Test fun `a long face becomes several segments shorter than one metre`() {
        val face = (0 until 60).map { doubleArrayOf(it * 0.1, 2.0 + jitter()) }
        val lines = MultiLineRansac.fit(face, 0.0)
        assertTrue(lines.size >= 5)
        assertTrue(lines.all { length(it) < 1.0 })
        assertTrue(lines.all { abs(it[1] - 2.0) < 0.1 && abs(it[3] - 2.0) < 0.1 })
        assertTrue(lines.sumOf { length(it) } > 3.5)
    }

    @Test fun `both faces of a corner are found`() {
        val along = (0 until 30).map { doubleArrayOf(it * 0.1, 0.0 + jitter()) }
        val across = (0 until 30).map { doubleArrayOf(3.0 + jitter(), it * 0.1 + 0.3) }
        val lines = MultiLineRansac.fit(along + across, 0.0)
        assertTrue(lines.any { abs(it[1] - it[3]) < 1e-9 })
        assertTrue(lines.any { abs(it[0] - it[2]) < 1e-9 })
    }

    @Test fun `every segment is parallel or perpendicular to the store axes`() {
        val yaw = 0.6
        val rotate = { u: Double, v: Double -> doubleArrayOf(u * cos(yaw) - v * sin(yaw), u * sin(yaw) + v * cos(yaw)) }
        val face = (0 until 40).map { rotate(it * 0.1, jitter()) } + (0 until 30).map { rotate(4.2 + jitter(), it * 0.1) }
        val lines = MultiLineRansac.fit(face, yaw)
        assertTrue(lines.isNotEmpty())
        for (l in lines) {
            val along = ((l[2] - l[0]) * cos(yaw) + (l[3] - l[1]) * sin(yaw)) / length(l)
            assertTrue(abs(abs(along) - 1.0) < 1e-9 || abs(along) < 1e-9)
        }
    }

    @Test fun `a diagonal gives no long segment`() {
        val diagonal = (0 until 60).map { doubleArrayOf(it * 0.07, it * 0.07) }
        assertTrue(MultiLineRansac.fit(diagonal, 0.0).none { length(it) > 0.5 })
    }

    @Test fun `sparse points and scatter give no segment, and the same seed repeats`() {
        assertEquals(0, MultiLineRansac.fit((0 until 5).map { doubleArrayOf(it * 0.1, 0.0) }, 0.0).size)
        val scatter = (0 until 200).map { doubleArrayOf(rng.nextDouble() * 8, rng.nextDouble() * 8) }
        assertTrue(MultiLineRansac.fit(scatter, 0.0).size <= 2)
        val pts = (0 until 30).map { doubleArrayOf(it * 0.1, 1.0 + jitter()) }
        assertEquals(MultiLineRansac.fit(pts, 0.0).map { it.toList() }, MultiLineRansac.fit(pts, 0.0).map { it.toList() })
    }
}
