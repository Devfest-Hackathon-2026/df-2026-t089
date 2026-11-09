package jp.devfest.shelfsight.assist

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.sin

/** 測定した頭部インパルス応答（SADIE II の KU100）。方向ごとに左右の耳の FIR を持つ。tools/export_hrtf.py で作る。 */
class Hrtf(bytes: ByteArray) {
    val count: Int
    private val dirX: FloatArray
    private val dirY: FloatArray
    private val dirZ: FloatArray
    private val ir: FloatArray

    init {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        count = b.int
        require(b.int == TAPS) { "HRIR length must be $TAPS" }
        dirX = FloatArray(count); dirY = FloatArray(count); dirZ = FloatArray(count)
        for (i in 0 until count) {
            val az = Math.toRadians(b.float.toDouble())
            val el = Math.toRadians(b.float.toDouble())
            // ファイルの方位は左回りなので、右が正の向きに直す
            dirX[i] = (-cos(el) * sin(az)).toFloat()
            dirY[i] = sin(el).toFloat()
            dirZ[i] = (cos(el) * cos(az)).toFloat()
        }
        ir = FloatArray(count * 2 * TAPS) { b.short / 32767f }
    }

    /** いちばん近い測定方向。[azimuth] は右が正、[elevation] は上が正（ラジアン）。 */
    fun nearest(azimuth: Float, elevation: Float): Int {
        val x = cos(elevation) * sin(azimuth)
        val y = sin(elevation)
        val z = cos(elevation) * cos(azimuth)
        var best = 0
        var bestDot = -2f
        for (i in 0 until count) {
            val d = dirX[i] * x + dirY[i] * y + dirZ[i] * z
            if (d > bestDot) { bestDot = d; best = i }
        }
        return best
    }

    /** [pcm] の [frame] 番目の出力。[ear] は 0 が左、1 が右。音の終わりの余韻のため、[frame] は pcm を [TAPS] 超えてよい。 */
    fun convolve(pcm: FloatArray, frame: Int, direction: Int, ear: Int): Float {
        val start = (direction * 2 + ear) * TAPS
        val first = maxOf(0, frame - pcm.size + 1)
        val last = minOf(TAPS - 1, frame)
        var sum = 0f
        for (i in first..last) sum += ir[start + i] * pcm[frame - i]
        return sum
    }

    companion object {
        const val TAPS = 256

        fun load(context: Context) = Hrtf(context.assets.open("hrtf/sadie_d1_ku100.bin").use { it.readBytes() })
    }
}
