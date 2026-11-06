package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3

/** 店の中に置いた 1 つの棚。[left]..[right] は写っていた画像の横位置（0..1）。 */
data class PlacedShelf(val category: Shelf8, val prob: Float, val left: Float, val right: Float, val position: Vec3)

/**
 * 言った棚を覚えておき、同じ場所の同じ棚を何度も言わないようにする。
 * 通路を歩きながら同じ棚が写り続けても、1 度だけ聞こえる。
 */
class ShelfMemory {
    private class Said(val category: Shelf8, val position: Vec3, val at: Long)

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

        /** 隣り合う同じ種類の棚を 1 つにまとめる（同じ種類が続く 1 つの棚を、境界で分けてしまったとき）。 */
        fun mergeNeighbors(shelves: List<PlacedShelf>): List<PlacedShelf> {
            val out = mutableListOf<PlacedShelf>()
            for (s in shelves.sortedBy { it.left }) {
                val last = out.lastOrNull()
                if (last != null && last.category == s.category) {
                    out[out.size - 1] = last.copy(
                        prob = maxOf(last.prob, s.prob),
                        right = s.right,
                        position = Vec3((last.position.x + s.position.x) / 2, (last.position.y + s.position.y) / 2, (last.position.z + s.position.z) / 2),
                    )
                } else {
                    out += s
                }
            }
            return out
        }
    }
}
