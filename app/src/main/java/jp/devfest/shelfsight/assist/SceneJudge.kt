package jp.devfest.shelfsight.assist

/** カメラに何が主に写っているか。 */
enum class Scene { NONE, SHELVES, ONE_SHELF, ONE_PRODUCT }

/** 1 回の解析で分かったこと。 */
data class SceneEvidence(
    /** 棚の境界で区切った区間のうち、8 分類のどれかになったもの。 */
    val shelves: Int,
    /** 商品の枠（正立画像の 0..1、[left, top, right, bottom]）。 */
    val productBoxes: List<FloatArray>,
)

/**
 * 写っているものを決める。毎回の結果はぶれるので、同じ判定が [STABLE] 回続いたときだけ切り替える。
 *
 * - 棚が 2 つ以上 → [Scene.SHELVES]
 * - 棚が 1 つで、商品がいくつも見える → [Scene.ONE_SHELF]
 */
class SceneJudge {
    var scene = Scene.NONE
        private set
    private var candidate = Scene.NONE
    private var count = 0

    /** 判定を足し、切り替わったら true。 */
    fun update(e: SceneEvidence): Boolean {
        val now = classify(e)
        if (now == candidate) count++ else { candidate = now; count = 1 }
        if (candidate != scene && count >= STABLE) { scene = candidate; return true }
        return false
    }

    companion object {
        const val STABLE = 2
        const val MIN_PRODUCTS_ON_SHELF = 4

        fun area(b: FloatArray) = (b[2] - b[0]).coerceAtLeast(0f) * (b[3] - b[1]).coerceAtLeast(0f)

        /** 画面の中央とみなす範囲（中心からの距離、画像の幅・高さに対する割合）。中央を含む枠がないとき、これより近い枠を選ぶ。 */
        const val CENTER_RADIUS = 0.15f

        /**
         * 画面の中央にある商品の枠（長押しで調べる 1 つ）。なければ null。
         * 中央の点を含む枠のうち、枠の中心が画面の中央にいちばん近いもの。含む枠がなければ、中心が [CENTER_RADIUS] 以内でいちばん近いもの。
         */
        fun centerProduct(boxes: List<FloatArray>): FloatArray? {
            fun dist(b: FloatArray) = kotlin.math.hypot((b[0] + b[2]) / 2 - 0.5f, (b[1] + b[3]) / 2 - 0.5f)
            val containing = boxes.filter { it[0] <= 0.5f && it[2] >= 0.5f && it[1] <= 0.5f && it[3] >= 0.5f }
            if (containing.isNotEmpty()) return containing.minBy(::dist)
            return boxes.filter { dist(it) <= CENTER_RADIUS }.minByOrNull(::dist)
        }

        fun classify(e: SceneEvidence): Scene = when {
            e.shelves >= 2 -> Scene.SHELVES
            e.productBoxes.size >= MIN_PRODUCTS_ON_SHELF -> Scene.ONE_SHELF
            e.shelves == 1 -> Scene.ONE_SHELF
            else -> Scene.NONE
        }
    }
}
