package jp.devfest.shelfsight.labeling

import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.scan.ScanGate
import kotlin.math.abs
import kotlin.math.max

/**
 * 立ち止まった瞬間を見つける。
 *
 * 1 回立ち止まるごとに 1 度だけ知らせ、十分に動くまで同じ場所では知らせない。
 * 歩いている間はクラウドを呼ばないので、通信量と料金を抑えられる。
 */
internal class DwellDetector(
    private val dwellSeconds: Double = 1.5,
    private val moveRadiusM: Float = 0.12f,
    private val turnCosine: Float = 0.978f, // およそ 12 度
) {
    private class Pose(val time: Double, val position: Vec3, val forward: Vec3)

    private var anchor: Pose? = null
    private var firedForAnchor = false
    private var lastFired: Pose? = null

    /** @return 今ちょうど立ち止まったと判定して、判別を依頼すべきなら true */
    fun update(timeSeconds: Double, position: Vec3, forward: Vec3): Boolean {
        val current = anchor
        if (current == null || hasMoved(current, position, forward)) {
            anchor = Pose(timeSeconds, position, forward)
            firedForAnchor = false
            return false
        }
        if (firedForAnchor || timeSeconds - current.time < dwellSeconds) return false
        firedForAnchor = true
        val last = lastFired
        if (last != null && !ScanGate.hasMovedEnough(last.position, last.forward, position, forward)) return false
        lastFired = Pose(timeSeconds, position, forward)
        return true
    }

    /** カメラが少しの間動いていなければ true。 */
    fun isStill(timeSeconds: Double): Boolean {
        val current = anchor ?: return false
        return timeSeconds - current.time >= STILL_SECONDS
    }

    /** 判別に失敗したら、動かなくてももう一度知らせるようにする。 */
    fun retry() {
        firedForAnchor = false
        lastFired = null
    }

    private fun hasMoved(from: Pose, position: Vec3, forward: Vec3): Boolean {
        if (from.position.distanceTo(position) > moveRadiusM) return true
        val cos = from.forward.x * forward.x + from.forward.y * forward.y + from.forward.z * forward.z
        return cos < turnCosine
    }

    private companion object {
        const val STILL_SECONDS = 0.4
    }
}

/** 検出された手のうち、商品を持っている可能性があるもの。 */
internal class HandCandidate(val box: NormalizedBox, val isPointing: Boolean, val distanceM: Float?)

/**
 * 商品を手に取ってカメラに見せている瞬間を見つける。
 *
 * 手だけでは商品を持っているか分からないので、読み取った文字で判断する。
 * 手を動かさずにいる間は 1 度だけ知らせ、同じ商品を繰り返し送らない。
 */
internal class HeldProductDetector(
    private val holdSeconds: Double = 0.8,
    private val maxCenterMove: Float = 0.08f,
    private val rearmCenterMove: Float = 0.2f,
    private val maxDistanceM: Float = 0.8f,
) {
    private var heldSince = 0.0
    private var heldX = 0f
    private var heldY = 0f
    private var tracking = false
    private var fired = false
    private var firedX = 0f
    private var firedY = 0f

    /** @return 商品を読み取る範囲の手の位置。今は読み取らないなら null */
    fun update(timeSeconds: Double, hands: List<HandCandidate>): NormalizedBox? {
        val hand = hands
            .filter { !it.isPointing && (it.distanceM == null || it.distanceM <= maxDistanceM) }
            .maxByOrNull { (it.box.right - it.box.left) * (it.box.bottom - it.box.top) }
        if (hand == null) {
            tracking = false
            fired = false // 手を下ろしたら、次に持った商品を読む
            return null
        }
        val x = (hand.box.left + hand.box.right) / 2f
        val y = (hand.box.top + hand.box.bottom) / 2f
        if (fired && max(abs(x - firedX), abs(y - firedY)) > rearmCenterMove) fired = false
        if (!tracking || max(abs(x - heldX), abs(y - heldY)) > maxCenterMove) {
            tracking = true
            heldSince = timeSeconds
            heldX = x
            heldY = y
            return null
        }
        if (fired || timeSeconds - heldSince < holdSeconds) return null
        fired = true
        firedX = x
        firedY = y
        return hand.box
    }

    fun retry() {
        fired = false
    }
}

/** 誤動作でクラウドを呼び続けて料金がかさまないよう、1 分あたりの回数を制限する。 */
internal class RequestBudget(private val maxPerMinute: Int = 12) {
    private val times = ArrayDeque<Double>()

    @Synchronized
    fun tryAcquire(timeSeconds: Double): Boolean {
        while (times.isNotEmpty() && timeSeconds - times.first() >= WINDOW_SECONDS) times.removeFirst()
        if (times.size >= maxPerMinute) return false
        times.addLast(timeSeconds)
        return true
    }

    private companion object {
        const val WINDOW_SECONDS = 60.0
    }
}

/** 画像を粗い明るさの格子にしたもの。カメラの前の変化を見るだけなので、これで十分で計算も軽い。 */
internal object SceneSignature {
    const val GRID = 16

    /** 端の映り込みに左右されないよう、中央 80% だけを見る。 */
    fun of(nv21: ByteArray, width: Int, height: Int): FloatArray {
        val out = FloatArray(GRID * GRID)
        for (row in 0 until GRID) {
            val y0 = height / 10 + row * height * 4 / 5 / GRID
            for (col in 0 until GRID) {
                val x0 = width / 10 + col * width * 4 / 5 / GRID
                var sum = 0
                for (dy in 0..1) for (dx in 0..1) {
                    sum += nv21[(y0 + dy * 2) * width + x0 + dx * 2].toInt() and 0xFF
                }
                out[row * GRID + col] = sum / 4f
            }
        }
        return out
    }
}

/**
 * カメラが動いていないのに景色が変わり、そのまま止まった瞬間を見つける。商品をカメラに見せたときに起きる。
 *
 * 箱を持つと手が隠れて手の検出が効かず、近すぎると深度も当てにならないので、画像そのものを比べる。
 * 歩いている間は景色が変わり続けて止まらないので反応しない。
 */
internal class SceneChangeDetector(
    private val stableSeconds: Double = 0.8,
    private val changedCellDelta: Float = 25f,
    private val minChangedFraction: Float = 0.2f,
    private val stableCellDelta: Float = 12f,
    private val maxUnstableFraction: Float = 0.1f,
    private val darkLuma: Float = 25f,
) {
    private var baseline: FloatArray? = null
    private var previous: FloatArray? = null

    /** 商品が現れる前の景色。商品を下げて元に戻ったときは、新しいものが現れたと扱わない。 */
    private var beforeProduct: FloatArray? = null
    private var stableSince = NOT_STABLE

    /** @return 新しいものが現れて止まったと判定したら true */
    fun update(timeSeconds: Double, signature: FloatArray): Boolean {
        // 端末を伏せて置いたときなど。真っ暗な画像からは何も読めない。
        if (signature.average() < darkLuma) {
            stableSince = NOT_STABLE
            previous = signature
            return false
        }
        val base = baseline
        if (base == null) {
            baseline = signature
            previous = signature
            return false
        }
        val last = previous
        previous = signature
        if (fractionChanged(base, signature, changedCellDelta) < minChangedFraction) {
            stableSince = NOT_STABLE
            return false
        }
        val moving = last == null || fractionChanged(last, signature, stableCellDelta) > maxUnstableFraction
        if (moving || stableSince == NOT_STABLE) {
            stableSince = timeSeconds
            return false
        }
        if (timeSeconds - stableSince < stableSeconds) return false
        val before = beforeProduct
        val returnedToBefore = before != null && fractionChanged(before, signature, changedCellDelta) < minChangedFraction
        beforeProduct = if (returnedToBefore) null else base
        baseline = signature
        stableSince = NOT_STABLE
        return !returnedToBefore
    }

    /** カメラが動いたときに呼ぶ。今見えている景色を基準にする。 */
    fun rebaseline(signature: FloatArray) {
        baseline = signature
        beforeProduct = null
        previous = signature
        stableSince = NOT_STABLE
    }

    private fun fractionChanged(a: FloatArray, b: FloatArray, delta: Float): Float {
        var count = 0
        for (i in a.indices) if (abs(a[i] - b[i]) > delta) count++
        return count.toFloat() / a.size
    }

    private companion object {
        const val NOT_STABLE = Double.NEGATIVE_INFINITY
    }
}
