package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/** 聞く人の位置と、水平面での正面の向き（単位ベクトル x, z）。 */
data class Listener(val position: Vec3, val forwardX: Float, val forwardZ: Float)

/** 片耳ずつの鳴らし方。 */
data class EarParams(
    val gainL: Float, val gainR: Float,
    /** 遅らせる量（秒）。 */
    val delayL: Float, val delayR: Float,
    /** 1 次ローパスの係数（1 で素通し）。頭の陰になる耳ほど高音を落とす。 */
    val lowpassL: Float, val lowpassR: Float,
)

/**
 * イヤホン向けの簡単な立体音響。左右の耳に届く時間の差（ITD）と大きさ・音色の差（ILD、頭の陰）、距離による減衰で、
 * 音の方向を作る。後ろの音は少しこもらせて、前と区別しやすくする。
 */
object Binaural {
    private const val HEAD_RADIUS_M = 0.0875f
    private const val SPEED_OF_SOUND = 343f
    private const val SHADOW_HZ = 1200f
    private const val BEHIND_HZ = 5000f

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

    fun params(az: Float, distance: Float, sampleRate: Int): EarParams {
        val s = sin(az)
        // 横から来る角度（前後どちらでも同じ）。Woodworth の式で時間差を出す
        val lateral = abs(atan2(s, abs(cos(az))))
        val itd = HEAD_RADIUS_M / SPEED_OF_SOUND * (lateral + sin(lateral))
        val g = distanceGain(distance)
        // 等パワーのパン。正面で左右 0.71、真横で 1 と 0.2 くらい
        val gl = sqrt(0.5f * (1f - 0.92f * s)) * g
        val gr = sqrt(0.5f * (1f + 0.92f * s)) * g
        val behind = cos(az) < 0f
        val shadow = { amount: Float -> lowpassCoeff(20000f - (20000f - SHADOW_HZ) * amount, sampleRate) }
        val back = if (behind) lowpassCoeff(BEHIND_HZ, sampleRate) else 1f
        return EarParams(
            gainL = gl, gainR = gr,
            delayL = if (s > 0) itd else 0f, delayR = if (s < 0) itd else 0f,
            lowpassL = min(back, shadow(max(0f, s))), lowpassR = min(back, shadow(max(0f, -s))),
        )
    }

    /** 1 次ローパス y += a(x - y) の a。 */
    fun lowpassCoeff(cutoffHz: Float, sampleRate: Int): Float =
        (1f - exp(-2f * PI.toFloat() * cutoffHz / sampleRate)).coerceIn(0f, 1f)

    /** 時計の文字盤の向き（12 時が正面）。声で方向を伝えるときに使う。 */
    fun clock(az: Float): Int {
        val h = ((Math.round(az / (2 * PI.toFloat()) * 12f) % 12) + 12) % 12
        return if (h == 0) 12 else h
    }
}
