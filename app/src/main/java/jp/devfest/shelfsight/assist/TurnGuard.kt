package jp.devfest.shelfsight.assist

import kotlin.math.abs
import kotlin.math.atan2

/**
 * 棚や商品の説明を始めたときの向きから、[MAX_TURN_DEG] より大きく向きを変えたら説明をやめる。
 * 向いている棚が変わったので、続きを聞いても今の前の棚の話ではなくなるため。
 *
 * [start] で説明を始めたときの向きと、そのときの声の世代（[Voice.turn]）を覚える。
 * [turned] を毎フレーム呼び、やめるべきときに true を返す（1 回だけ）。ほかの声で上書きされたり（世代が変わる）、
 * 話し終わったりしたら、見るのをやめる。複数のスレッドから呼んでよい。
 */
class TurnGuard {
    private var forwardX = 0f
    private var forwardZ = 0f
    private var turn = 0
    private var startedAt = 0L
    private var active = false

    @Synchronized
    fun start(l: Listener, turn: Int, now: Long) {
        forwardX = l.forwardX
        forwardZ = l.forwardZ
        this.turn = turn
        startedAt = now
        active = true
    }

    @Synchronized
    fun clear() {
        active = false
    }

    /** [speaking] は声が鳴っているか。合成を待つ間は鳴っていないので、始めてから [START_GRACE_MS] は待つ。 */
    @Synchronized
    fun turned(l: Listener, turn: Int, speaking: Boolean, now: Long): Boolean {
        if (!active) return false
        if (turn != this.turn || (!speaking && now - startedAt > START_GRACE_MS)) {
            active = false
            return false
        }
        if (abs(angleDeg(forwardX, forwardZ, l.forwardX, l.forwardZ)) <= MAX_TURN_DEG) return false
        active = false
        return true
    }

    companion object {
        const val MAX_TURN_DEG = 90f
        const val START_GRACE_MS = 3000L

        /** 水平の向き (ax, az) から (bx, bz) への角度（度、-180〜180）。 */
        fun angleDeg(ax: Float, az: Float, bx: Float, bz: Float): Float =
            Math.toDegrees(atan2((ax * bz - az * bx).toDouble(), (ax * bx + az * bz).toDouble())).toFloat()
    }
}
