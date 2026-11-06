package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3

/**
 * 写っている 1 つの棚。歩行中の判別（CoarseClassifier）が列ごとに出したカテゴリのうち、同じカテゴリが続く列のまとまり。
 * カテゴリが変わる列の間が棚の境界。[left]..[right] は画像の横位置（0..1）。
 */
data class PlacedShelf(val category: String, val confidence: Float, val left: Float, val right: Float, val position: Vec3 = Vec3(0f, 0f, 0f)) {
    /** 声で言うときの名前（「飲み物・お酒」→「飲み物とお酒」）。 */
    val spoken: String get() = category.replace("・", "と")
}

/**
 * 言った棚を覚えておき、同じ場所の同じ棚を何度も言わないようにする。
 * 通路を歩きながら同じ棚が写り続けても、1 度だけ聞こえる。
 */
class ShelfMemory {
    private class Said(val category: String, val position: Vec3, val at: Long)

    private val said = ArrayList<Said>()

    @Synchronized
    fun isNew(s: PlacedShelf, now: Long): Boolean =
        said.none { it.category == s.category && now - it.at < REMEMBER_MS && it.position.distanceTo(s.position) < SAME_PLACE_M }

    @Synchronized
    fun remember(s: PlacedShelf, now: Long) {
        said.removeAll { now - it.at > REMEMBER_MS }
        said += Said(s.category, s.position, now)
    }

    @Synchronized
    fun clear() = said.clear()

    companion object {
        const val REMEMBER_MS = 60_000L
        const val SAME_PLACE_M = 2.0f
        /** 歩行中の判別の「棚ではない」カテゴリ（coarse の head.json の classes）。 */
        const val NO_PRODUCT = "商品なし"

        /** 列ごとのカテゴリ（左から右）を、同じカテゴリが続くまとまりに分ける。「商品なし」は棚にしない。 */
        fun shelves(columns: List<String>, confidence: FloatArray): List<PlacedShelf> {
            val n = columns.size
            val out = mutableListOf<PlacedShelf>()
            var start = 0
            for (i in 1..n) {
                if (i < n && columns[i] == columns[start]) continue
                if (columns[start] != NO_PRODUCT) {
                    val conf = (start until i).map { confidence[it] }.average().toFloat()
                    out += PlacedShelf(columns[start], conf, start.toFloat() / n, i.toFloat() / n)
                }
                start = i
            }
            return out
        }
    }
}
