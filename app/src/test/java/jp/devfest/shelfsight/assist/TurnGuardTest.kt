package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class TurnGuardTest {
    /** -z から右回りに deg 度向いた人。 */
    private fun facing(deg: Double) = Listener(Vec3(0f, 1.3f, 0f), sin(Math.toRadians(deg)).toFloat(), -cos(Math.toRadians(deg)).toFloat())

    @Test
    fun `turning more than 90 degrees stops the description once`() {
        val g = TurnGuard()
        g.start(facing(0.0), turn = 1, now = 0L)
        assertFalse(g.turned(facing(60.0), 1, speaking = true, now = 500L))
        assertFalse(g.turned(facing(-89.0), 1, speaking = true, now = 600L))
        assertTrue(g.turned(facing(95.0), 1, speaking = true, now = 700L))
        assertFalse(g.turned(facing(180.0), 1, speaking = true, now = 800L))   // 1 回だけ
    }

    @Test
    fun `turning left counts the same, across the back`() {
        val g = TurnGuard()
        g.start(facing(170.0), turn = 1, now = 0L)
        assertFalse(g.turned(facing(-110.0), 1, speaking = true, now = 100L))   // 170° → 250°: 80°
        assertTrue(g.turned(facing(-95.0), 1, speaking = true, now = 200L))     // 170° → 265°: 95°
    }

    @Test
    fun `another voice or the end of the description stops watching`() {
        val g = TurnGuard()
        g.start(facing(0.0), turn = 1, now = 0L)
        assertFalse(g.turned(facing(120.0), 2, speaking = true, now = 100L))   // ほかの声で上書きされた
        assertFalse(g.turned(facing(120.0), 1, speaking = true, now = 200L))

        g.start(facing(0.0), turn = 3, now = 0L)
        assertTrue(g.turned(facing(120.0), 3, speaking = false, now = 1000L))   // 合成を待っている間も見る
        g.start(facing(0.0), turn = 3, now = 0L)
        assertFalse(g.turned(facing(120.0), 3, speaking = false, now = TurnGuard.START_GRACE_MS + 1))   // 話し終わった
    }

    @Test
    fun `angle between headings`() {
        assertEquals(90f, TurnGuard.angleDeg(0f, -1f, 1f, 0f), 1e-4f)
        assertEquals(-90f, TurnGuard.angleDeg(0f, -1f, -1f, 0f), 1e-4f)
        assertEquals(180f, kotlin.math.abs(TurnGuard.angleDeg(0f, -1f, 0f, 1f)), 1e-4f)
    }
}
