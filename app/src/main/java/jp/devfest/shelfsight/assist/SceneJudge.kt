package jp.devfest.shelfsight.assist

/** カメラに何が主に写っているか。 */
enum class Scene { NONE, SHELVES, ONE_SHELF, ONE_PRODUCT }

/** 長押しで調べるもの。 */
enum class Focus { PRODUCT, SHELF, STEP_BACK, NOTHING }

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
        /** 棚の説明で、大きすぎる枠（手前の商品や棚の区画）があれば棚とはみなさない。画面に占める割合。 */
        const val LARGE_BOX_AREA = 0.12f
        const val MIN_PRODUCTS_ON_SHELF = 4

        fun area(b: FloatArray) = (b[2] - b[0]).coerceAtLeast(0f) * (b[3] - b[1]).coerceAtLeast(0f)

        /** 長押しで棚を説明するのに必要な商品の数と、商品が広がっている範囲（画面に対する幅・高さ）。 */
        const val SHELF_MIN_PRODUCTS = 6
        const val SHELF_MIN_WIDTH = 0.5f
        const val SHELF_MIN_HEIGHT = 0.35f

        /**
         * 棚の広い範囲を写しているか。商品 1 つ（[Focus.PRODUCT]）はここでは選ばない。
         * - 棚の広い範囲に小さな商品が並んでいる → [Focus.SHELF]
         * - それ以外（商品が少ない、大きめの物が写っている）→ [Focus.STEP_BACK]
         */
        fun focus(boxes: List<FloatArray>): Pair<Focus, FloatArray?> {
            if (boxes.isEmpty()) return Focus.NOTHING to null
            val union = floatArrayOf(boxes.minOf { it[0] }, boxes.minOf { it[1] }, boxes.maxOf { it[2] }, boxes.maxOf { it[3] })
            val spread = boxes.size >= SHELF_MIN_PRODUCTS && union[2] - union[0] >= SHELF_MIN_WIDTH && union[3] - union[1] >= SHELF_MIN_HEIGHT
            val noneLarge = boxes.none { area(it) >= LARGE_BOX_AREA }
            return (if (spread && noneLarge) Focus.SHELF else Focus.STEP_BACK) to null
        }

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
