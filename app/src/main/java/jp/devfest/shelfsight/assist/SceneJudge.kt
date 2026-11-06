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

        /** 長押しで棚を説明するのに必要な商品の数と、商品が広がっている範囲（画面に対する幅・高さ）。 */
        const val SHELF_MIN_PRODUCTS = 6
        const val SHELF_MIN_WIDTH = 0.5f
        const val SHELF_MIN_HEIGHT = 0.35f

        /** 画面のこれ以上を占める枠は、他の枠があっても手前の商品 1 つ（近づくと商品の中の文字や絵も枠になる）。 */
        const val CLOSE_PRODUCT_AREA = 0.25f
        /** 枠が少なく、合わせて画面のこれ以上を占めるときも、商品 1 つを近くで写している（横に並んだ 2 つの商品は、ここまで広がらない）。 */
        const val CLOSE_UNION_AREA = 0.6f
        const val CLOSE_MAX_BOXES = 4

        /**
         * 長押しのとき、何を調べるか。商品 1 つなら調べる範囲（正立画像の 0..1）も返す。
         * - 商品 1 つが大きく写っている、または画面いっぱいに写っている → [Focus.PRODUCT]
         * - 棚の広い範囲に小さな商品が並んでいる → [Focus.SHELF]
         * - その間（中くらいの商品がいくつか、または商品が少ない） → [Focus.STEP_BACK]
         */
        fun focus(boxes: List<FloatArray>): Pair<Focus, FloatArray?> {
            if (boxes.isEmpty()) return Focus.NOTHING to null
            mainProduct(boxes)?.let { return Focus.PRODUCT to it }
            val largest = boxes.maxBy { area(it) }
            val union = floatArrayOf(boxes.minOf { it[0] }, boxes.minOf { it[1] }, boxes.maxOf { it[2] }, boxes.maxOf { it[3] })
            // 調べる範囲は、枠が少なければ全部を合わせた範囲（商品の部分ごとに枠が出ることがある）
            if (area(largest) >= CLOSE_PRODUCT_AREA) return Focus.PRODUCT to if (boxes.size <= CLOSE_MAX_BOXES) union else largest
            if (boxes.size <= CLOSE_MAX_BOXES && area(union) >= CLOSE_UNION_AREA && area(largest) >= ONE_PRODUCT_AREA / 2) {
                return Focus.PRODUCT to union
            }
            val spread = boxes.size >= SHELF_MIN_PRODUCTS && union[2] - union[0] >= SHELF_MIN_WIDTH && union[3] - union[1] >= SHELF_MIN_HEIGHT
            val noneLarge = boxes.none { area(it) >= ONE_PRODUCT_AREA }
            return (if (spread && noneLarge) Focus.SHELF else Focus.STEP_BACK) to null
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
