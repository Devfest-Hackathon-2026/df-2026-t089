package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3

/**
 * MobileCLIP の判定を待つ、点ごとの切り抜き。判定には 1 枚 0.2 秒ほどかかり全部は追いつかないので、
 * 今いる場所に近く、新しく、大きく写り、検出の確からしさが高いものから取り出す。
 */
internal class CropQueue<T>(private val capacity: Int) {
    class Entry<T>(val crop: T, val box: FloatArray, val position: Vec3, val camera: Vec3,
        val widthM: Float, val heightM: Float, val area: Float, val time: Long, val confidence: Float = 0f)

    private class Rejected(val position: Vec3, val area: Float)

    private val items = ArrayList<Entry<T>>()
    private val rejected = ArrayDeque<Rejected>()

    @get:Synchronized val size: Int get() = items.size

    /** 判定しても分からなかった場所を覚える。より大きく写るまで、同じ場所は判定し直さない。 */
    @Synchronized
    fun reject(entry: Entry<T>) {
        rejected.removeAll { it.position.distanceTo(entry.position) < ShelfMap.SAME_SPOT_M }
        rejected.addLast(Rejected(entry.position, entry.area))
        if (rejected.size > MAX_REJECTED) rejected.removeFirst()
    }

    /** 同じ場所の切り抜きは大きく写った方だけ残す。@return 不要になった切り抜き（呼ぶ側で解放する） */
    @Synchronized
    fun add(entry: Entry<T>): List<T> {
        val failed = rejected.firstOrNull { it.position.distanceTo(entry.position) < ShelfMap.SAME_SPOT_M }
        if (failed != null && entry.area < failed.area * RETRY_AREA) return listOf(entry.crop)
        val same = items.firstOrNull { it.position.distanceTo(entry.position) < ShelfMap.SAME_SPOT_M }
        if (same != null) {
            if (same.area >= entry.area) return listOf(entry.crop)
            items.remove(same)
            items += entry
            return listOf(same.crop)
        }
        items += entry
        if (items.size <= capacity) return emptyList()
        val oldest = items.minBy { it.time }
        items.remove(oldest)
        return listOf(oldest.crop)
    }

    /** いちばん優先度の高い切り抜きを取り出す。空なら null。 */
    @Synchronized
    fun take(user: Vec3?, now: Long): Entry<T>? {
        val best = items.minByOrNull { score(it, user, now) } ?: return null
        items.remove(best)
        return best
    }

    private fun score(e: Entry<T>, user: Vec3?, now: Long): Float =
        (user?.distanceTo(e.position) ?: 0f) + (now - e.time) / 1000f * AGE_PER_SECOND - e.area * AREA_WEIGHT - e.confidence * CONFIDENCE_WEIGHT

    private companion object {
        const val AGE_PER_SECOND = 0.2f   // 5 秒古いと 1 m 遠いのと同じ
        const val AREA_WEIGHT = 2f
        const val CONFIDENCE_WEIGHT = 1f  // 確からしさ 0.5 の差は 0.5 m 近いのと同じ
        const val RETRY_AREA = 1.5f
        const val MAX_REJECTED = 1000
    }
}
