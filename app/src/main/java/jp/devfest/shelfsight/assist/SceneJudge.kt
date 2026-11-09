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
    /** スマホのすぐ前に商品がある（[NearProduct]、画面の中央の深度だけで決める）。 */
    val near: Boolean = false,
)

/**
 * 写っているものを決める（iOS 版の SceneJudge と同じ）。毎回の結果はぶれるので、同じ判定が [STABLE] 回続いたときだけ切り替える。
 *
 * - スマホのすぐ前に商品がある（[NearProduct]、画面の中央の深度だけで決める）→ [Scene.ONE_PRODUCT]
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

        /** 長押しで棚を説明するのに必要な商品の数と、商品が広がっている範囲（画面に対する幅・高さ）。 */
        const val SHELF_MIN_PRODUCTS = 6
        const val SHELF_MIN_WIDTH = 0.5f
        const val SHELF_MIN_HEIGHT = 0.35f

        /**
         * 長押しのとき、何を調べるか。商品 1 つなら調べる範囲（正立画像の 0..1）も返す。
         * - スマホのすぐ前に商品がある（[nearBox] は [NearProduct] で決めた調べる範囲）→ [Focus.PRODUCT]
         * - 棚の広い範囲に小さな商品が並んでいる → [Focus.SHELF]
         * - それ以外（商品が少ない、広がっていない）→ [Focus.STEP_BACK]。枠の大きさでは決めない
         */
        fun focus(boxes: List<FloatArray>, nearBox: FloatArray? = null): Pair<Focus, FloatArray?> {
            if (nearBox != null) return Focus.PRODUCT to nearBox
            if (boxes.isEmpty()) return Focus.NOTHING to null
            val w = boxes.maxOf { it[2] } - boxes.minOf { it[0] }
            val h = boxes.maxOf { it[3] } - boxes.minOf { it[1] }
            val spread = boxes.size >= SHELF_MIN_PRODUCTS && w >= SHELF_MIN_WIDTH && h >= SHELF_MIN_HEIGHT
            return (if (spread) Focus.SHELF else Focus.STEP_BACK) to null
        }

        /** 長押しで調べるものと範囲。[byCenterBox] は、ARCore で判定できず画面の中央の枠で選んだとき true。 */
        data class LongPressTarget(val focus: Focus, val box: FloatArray?, val byCenterBox: Boolean)

        /**
         * [focus] に、Android 版だけの代わりを足したもの。ARCore で近い商品を判定できなかったとき（深度に対応していない機種、
         * または深度で画面の中央の距離を測れなかったとき、[depthMeasured] が false）だけ、棚でなければ画面の中央の商品の枠を調べる。
         * 深度で測れて商品が遠いと分かったときは使わない（iOS 版と同じく、棚の説明か「一歩下がって」になる）。
         */
        fun focusWithFallback(boxes: List<FloatArray>, nearBox: FloatArray?, depthMeasured: Boolean): LongPressTarget {
            val (f, b) = focus(boxes, nearBox)
            if (nearBox == null && !depthMeasured && f != Focus.SHELF) {
                centerProduct(boxes)?.let { return LongPressTarget(Focus.PRODUCT, it, byCenterBox = true) }
            }
            return LongPressTarget(f, b, byCenterBox = false)
        }

        fun classify(e: SceneEvidence): Scene = when {
            e.near -> Scene.ONE_PRODUCT
            e.shelves >= 2 -> Scene.SHELVES
            e.productBoxes.size >= MIN_PRODUCTS_ON_SHELF -> Scene.ONE_SHELF
            e.shelves == 1 -> Scene.ONE_SHELF
            else -> Scene.NONE
        }
    }
}
