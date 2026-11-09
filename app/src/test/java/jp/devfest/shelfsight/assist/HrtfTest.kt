package jp.devfest.shelfsight.assist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI

class HrtfTest {
    private val real by lazy { Hrtf(File("src/main/assets/hrtf/sadie_d1_ku100.bin").readBytes()) }

    /** Two directions: front (left ear impulse at tap 0) and 90 degrees left in file coordinates (right ear impulse at tap 3). */
    private fun synthetic(): Hrtf {
        val b = ByteBuffer.allocate(8 + 2 * 8 + 2 * 2 * Hrtf.TAPS * 2).order(ByteOrder.LITTLE_ENDIAN)
        b.putInt(2).putInt(Hrtf.TAPS)
        b.putFloat(0f).putFloat(0f).putFloat(90f).putFloat(0f)
        for (dir in 0..1) for (ear in 0..1) for (tap in 0 until Hrtf.TAPS) {
            val one = (dir == 0 && ear == 0 && tap == 0) || (dir == 1 && ear == 1 && tap == 3)
            b.putShort(if (one) 32767 else 0)
        }
        return Hrtf(b.array())
    }

    @Test fun `azimuth is positive to the right`() {
        val h = synthetic()
        assertEquals(0, h.nearest(0f, 0f))
        assertEquals(1, h.nearest((-PI / 2).toFloat(), 0f))
    }

    @Test fun `convolve delays by the impulse tap and rings past the end of the sound`() {
        val h = synthetic()
        val pcm = floatArrayOf(0.5f, -0.25f)
        assertEquals(0.5f, h.convolve(pcm, 0, 0, 0), 1e-4f)
        assertEquals(0f, h.convolve(pcm, 0, 1, 1), 1e-4f)
        assertEquals(0.5f, h.convolve(pcm, 3, 1, 1), 1e-4f)
        assertEquals(-0.25f, h.convolve(pcm, 4, 1, 1), 1e-4f)
        assertEquals(0f, h.convolve(pcm, 2 + Hrtf.TAPS, 0, 0), 1e-4f)
    }

    private fun energy(az: Float, ear: Int): Float {
        val impulse = floatArrayOf(1f)
        val dir = real.nearest(az, 0f)
        return (0 until Hrtf.TAPS).sumOf { i -> real.convolve(impulse, i, dir, ear).toDouble().let { it * it } }.toFloat()
    }

    @Test fun `the measured set is loud in the ear nearest the sound`() {
        assertTrue(real.count > 1000)
        val right = (PI / 2).toFloat()
        assertTrue(energy(right, 1) > 4 * energy(right, 0))
        assertTrue(energy(-right, 0) > 4 * energy(-right, 1))
    }
}
