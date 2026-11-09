package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class SpatialTestLogicTest {
    private val chime = SpatialMixer.makeChime()

    /** In-place radix-2 FFT. */
    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { re[i] = re[j].also { re[j] = re[i] }; im[i] = im[j].also { im[j] = im[i] } }
        }
        var len = 2
        while (len <= n) {
            val w = -2 * PI / len
            for (start in 0 until n step len) for (k in 0 until len / 2) {
                val c = cos(w * k); val s = sin(w * k)
                val a = start + k; val b = a + len / 2
                val xr = re[b] * c - im[b] * s; val xi = re[b] * s + im[b] * c
                re[b] = re[a] - xr; im[b] = im[a] - xi
                re[a] += xr; im[a] += xi
            }
            len *= 2
        }
    }

    /** Energy in each 2 kHz band, as a share of the total, keyed by the band's lower edge in Hz. */
    private fun bandShares(pcm: FloatArray = chime): Map<Int, Double> {
        val n = 1 shl 15
        val re = DoubleArray(n) { pcm.getOrElse(it) { 0f }.toDouble() }
        val im = DoubleArray(n)
        fft(re, im)
        val bands = HashMap<Int, Double>()
        for (k in 1 until n / 2) {
            val band = (k * SpatialMixer.RATE / n) / 2000 * 2000
            bands[band] = (bands[band] ?: 0.0) + re[k] * re[k] + im[k] * im[k]
        }
        val total = bands.values.sum()
        return bands.mapValues { it.value / total }
    }

    @Test fun `chime lasts 360 ms in two pulses and leaves headroom`() {
        assertEquals(0.36, chime.size.toDouble() / SpatialMixer.RATE, 0.001)
        fun peak(fromS: Double, toS: Double) = chime.slice((fromS * SpatialMixer.RATE).toInt() until (toS * SpatialMixer.RATE).toInt()).maxOf { abs(it) }
        assertTrue(peak(0.0, 0.1) > 0.3f)
        assertTrue(peak(0.2, 0.3) > 0.3f)
        assertTrue(peak(0.165, 0.195) < 0.02f)
        assertTrue(chime.maxOf { abs(it) } <= 0.7f)
    }

    @Test fun `chime has broad energy above 4 kHz`() {
        val shares = bandShares()
        val report = shares.toSortedMap().entries.joinToString { "${it.key}=${"%.3f".format(it.value)}" }
        assertTrue("share above 4 kHz: $report", shares.filterKeys { it >= 4000 }.values.sum() > 0.35)
        for (from in 4000 until 16000 step 2000) assertTrue("band at $from Hz: $report", (shares[from] ?: 0.0) > 0.01)
    }

    @Test fun `approach alert is three short pulses with energy above 4 kHz`() {
        val alert = SpatialMixer.makeApproachAlert()
        assertEquals(0.175, alert.size.toDouble() / SpatialMixer.RATE, 0.001)
        fun peak(fromS: Double, toS: Double) = alert.slice((fromS * SpatialMixer.RATE).toInt() until (toS * SpatialMixer.RATE).toInt()).maxOf { abs(it) }
        for (start in listOf(0.0, 0.065, 0.13)) assertTrue(peak(start, start + 0.04) > 0.2f)
        for (start in listOf(0.046, 0.111)) assertTrue(peak(start, start + 0.018) < 0.02f)
        assertTrue(alert.maxOf { abs(it) } <= 0.7f)
        assertTrue(bandShares(alert).filterKeys { it >= 4000 }.values.sum() > 0.25)
    }

    @Test fun `demo chimes five times from one spot then moves silently`() {
        assertEquals(setOf(0, 1, 2, 3, 4), (0 until 70).mapNotNull { SpatialDemo.chimeSlot(it * 0.05) }.toSet())
        val first = SpatialDemo.offset(0.0)
        assertEquals(first, SpatialDemo.offset(3.4))
        assertEquals(null, SpatialDemo.chimeSlot(3.6))
        val mid = SpatialDemo.offset(5.0)
        val next = SpatialDemo.offset(6.6)
        assertTrue(mid != first && mid != next)
        assertEquals(next, SpatialDemo.offset(9.5))
        assertEquals(5, SpatialDemo.chimeSlot(6.5))
    }

    @Test fun `demo moves in small steps, starts ahead and covers all directions`() {
        assertEquals(0f, SpatialDemo.offset(0.0).x, 1e-3f)
        assertTrue(SpatialDemo.offset(0.0).z > 0f)
        val step = 0.02
        val points = (0 until (6.5 * SpatialDemo.SPOT_COUNT / step).toInt()).map { SpatialDemo.offset(it * step) }
        for ((a, b) in points.zipWithNext()) assertTrue(sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)) < 0.1f)
        assertTrue(points.any { it.y > 0.8f } && points.any { it.y < -0.8f })
        assertTrue(points.any { it.x > 1f } && points.any { it.x < -1f } && points.any { it.z > 1f } && points.any { it.z < -1f })
    }

    @Test fun `demo offsets are relative to where the listener first faced`() {
        // Facing +x: right is +z, so a spot ahead lands on +x from the listener.
        val p = SpatialDemo.world(Listener(Vec3(1f, 0f, 1f), 1f, 0f), 0.0)
        assertEquals(3f, p.x, 1e-3f)
        assertEquals(1f, p.z, 1e-3f)
        val l = Listener(Vec3(0f, 0f, 0f), 0f, -1f)
        assertEquals(0f, Binaural.azimuth(l, SpatialDemo.world(l, 0.0)), 1e-3f)
        val right = SpatialDemo.world(l, 6.6)   // the second spot is 40 degrees to the right
        assertEquals(40.0, Math.toDegrees(Binaural.azimuth(l, right).toDouble()), 0.5)
    }
}
