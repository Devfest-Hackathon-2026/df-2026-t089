package jp.devfest.shelfsight.assist

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
    private fun bandShares(): Map<Int, Double> {
        val n = 1 shl 15
        val re = DoubleArray(n) { chime.getOrElse(it) { 0f }.toDouble() }
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

    @Test fun `demo source circles, rises, falls and changes distance within range`() {
        val points = (0 until 2000).map { SpatialDemo.offset(it * 0.05) }
        val distances = points.map { sqrt(it.x * it.x + it.y * it.y + it.z * it.z) }
        assertTrue(distances.min() >= SpatialDemo.MIN_DISTANCE_M - 1e-3 && distances.min() < 0.8)
        assertTrue(distances.max() <= SpatialDemo.MAX_DISTANCE_M + 1e-3 && distances.max() > 3.9)
        assertTrue(points.any { it.y > 1f } && points.any { it.y < -1f })
        assertTrue(points.any { it.x > 1f } && points.any { it.x < -1f } && points.any { it.z > 1f } && points.any { it.z < -1f })
    }
}
