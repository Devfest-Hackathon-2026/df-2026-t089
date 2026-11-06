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
 * - 商品が 1 つ大きく写っている（近づいている）→ [Scene.ONE_PRODUCT]
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
        /** 画面に占める割合がこれ以上の商品は「1 つ大きく写っている」。 */
        const val ONE_PRODUCT_AREA = 0.12f
        /** 1 つ大きく写っている商品は、2 番目の商品よりこれだけ大きい。 */
        const val ONE_PRODUCT_RATIO = 2.5f
        const val MIN_PRODUCTS_ON_SHELF = 4

        fun area(b: FloatArray) = (b[2] - b[0]).coerceAtLeast(0f) * (b[3] - b[1]).coerceAtLeast(0f)

        /** 画面の真ん中にかかる、大きく写った 1 つの商品。なければ null。 */
        fun mainProduct(boxes: List<FloatArray>): FloatArray? {
            val sorted = boxes.sortedByDescending { area(it) }
            val first = sorted.firstOrNull() ?: return null
            val second = sorted.getOrNull(1)?.let { area(it) } ?: 0f
            val centered = first[0] < 0.75f && first[2] > 0.25f && first[1] < 0.75f && first[3] > 0.25f
            return first.takeIf { area(it) >= ONE_PRODUCT_AREA && area(it) >= second * ONE_PRODUCT_RATIO && centered }
        }

        fun classify(e: SceneEvidence): Scene = when {
            mainProduct(e.productBoxes) != null -> Scene.ONE_PRODUCT
            e.shelves >= 2 -> Scene.SHELVES
            e.productBoxes.size >= MIN_PRODUCTS_ON_SHELF -> Scene.ONE_SHELF
            e.shelves == 1 -> Scene.ONE_SHELF
            else -> Scene.NONE
        }
    }
}
