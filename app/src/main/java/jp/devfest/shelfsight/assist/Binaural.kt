package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.sqrt

/** 聞く人の位置と、水平面での正面の向き（単位ベクトル x, z）。 */
data class Listener(val position: Vec3, val forwardX: Float, val forwardZ: Float)

/** 聞く人から見た音の方向と距離。耳へ届く音そのものは [Hrtf] で作る。 */
object Binaural {
    /** 右が正の方位角（ラジアン、-π..π）。0 が正面。 */
    fun azimuth(l: Listener, source: Vec3): Float {
        val dx = source.x - l.position.x
        val dz = source.z - l.position.z
        // 右向きのベクトルは正面を時計回りに 90° 回したもの（y が上、正面が -z のとき右は +x）
        val rightX = -l.forwardZ
        val rightZ = l.forwardX
        return atan2(dx * rightX + dz * rightZ, dx * l.forwardX + dz * l.forwardZ)
    }

    fun horizontalDistance(l: Listener, source: Vec3): Float {
        val dx = source.x - l.position.x
        val dz = source.z - l.position.z
        return sqrt(dx * dx + dz * dz)
    }

    /** 距離による音量。1m までは 1、それより遠いと距離に反比例（遠くても小さくなりすぎない）。 */
    fun distanceGain(d: Float): Float = (1f / max(d, 1f)).coerceAtLeast(0.25f)

    /** 時計の文字盤の向き（12 時が正面）。声で方向を伝えるときに使う。 */
    fun clock(az: Float): Int {
        val h = ((Math.round(az / (2 * PI.toFloat()) * 12f) % 12) + 12) % 12
        return if (h == 0) 12 else h
    }
}
