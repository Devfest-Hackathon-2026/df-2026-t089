package jp.devfest.shelfsight.assist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

class MirrorCorrectionTest {
    private fun ray(bearing: Double, depth: Double) = doubleArrayOf(0.0, 0.0, depth * cos(bearing), depth * sin(bearing))

    @Test fun `a shelf seen obliquely is not glass`() {
        // A straight face at x = 1 seen across a wide range of depths within a few bearings.
        val rays = (0 until 40).map { i ->
            val z = 3.0 + i * 0.05
            doubleArrayOf(0.0, 0.0, 1.0, z)
        }
        assertEquals(0, MirrorCorrection.correct(rays).moved.size)
    }

    @Test fun `scatter along one bearing is pulled back along the same ray`() {
        val rng = Random(5)
        val rays = (0 until 40).map { ray(0.3 + rng.nextDouble() * 0.1, 1.5 + rng.nextDouble() * 3.0) }
        val result = MirrorCorrection.correct(rays)
        assertTrue(result.moved.size > 20)
        for (m in result.moved) {
            assertTrue("closer to the camera", hypot(m[2], m[3]) < hypot(m[0], m[1]))
            assertEquals("same bearing", kotlin.math.atan2(m[1], m[0]), kotlin.math.atan2(m[3], m[2]), 1e-6)
        }
        assertEquals(rays.size, result.points.size)
    }

    @Test fun `different viewpoints are judged separately and cell means thin the points`() {
        val tight = (0 until 10).map { doubleArrayOf(0.0, 0.0, 2.0, 0.01 * it) }
        val elsewhere = (0 until 10).map { doubleArrayOf(5.0, 5.0, 8.0 + it * 0.5, 5.0) }
        assertEquals(0, MirrorCorrection.correct(tight).moved.size)
        assertTrue(MirrorCorrection.correct(elsewhere).moved.isNotEmpty())
        val cells = MirrorCorrection.cellMeans(listOf(doubleArrayOf(0.01, 0.01), doubleArrayOf(0.03, 0.03), doubleArrayOf(1.0, 1.0)))
        assertEquals(2, cells.size)
        assertTrue(cells.any { abs(it[0] - 0.02) < 1e-9 })
    }
}
