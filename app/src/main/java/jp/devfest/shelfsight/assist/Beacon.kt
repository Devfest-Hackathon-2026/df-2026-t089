package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import kotlin.math.hypot

/**
 * 「〇〇はどこ？」と聞かれた場所から、立体音響のジングルを [INTERVAL_MS] ごとに鳴らして方向を伝える。
 * 利用者が [ARRIVE_M] 以内（水平の距離）に着いたら止める。毎フレーム [update] を呼ぶ。
 */
class Beacon(val name: String, val at: Vec3) {
    enum class Event { PLAY, ARRIVED }

    private var lastPlay: Long? = null

    fun update(l: Listener, now: Long): Event? {
        if (hypot(at.x - l.position.x, at.z - l.position.z) <= ARRIVE_M) return Event.ARRIVED
        val last = lastPlay
        if (last != null && now - last < INTERVAL_MS) return null
        lastPlay = now
        return Event.PLAY
    }

    companion object {
        const val INTERVAL_MS = 3000L
        const val ARRIVE_M = 1.0f

        /** 「〇〇はどこ」の〇〇。場所を聞く質問でなければ null。 */
        fun whereQuery(text: String): String? {
            val t = text.replace(" ", "").replace("　", "").trimEnd('？', '?', '。')
            val m = WHERE.find(t) ?: return null
            return m.groupValues[1].removeSuffix("の場所").removeSuffix("売り場").removeSuffix("の棚").takeIf { it.length >= 2 }
        }

        private val WHERE = Regex("^(.+?)(?:は|って|が|の場所は)?(?:どこ|どっち|どのへん|どの辺)")

        /**
         * 地図の中で、〇〇にいちばん合う場所。商品（名前・細かい種類・ジャンル・話す名前）が合えばその中でいちばん近いもの、
         * なければ主な商品が合う棚の面。見つからなければ null。
         */
        fun find(query: String, items: List<MapItem>, view: StoreView?, l: Listener): Pair<String, Vec3>? {
            fun matches(key: String?) = key != null && key.length >= 2 && (query in key || key in query)
            fun distance(p: Vec3) = hypot(p.x - l.position.x, p.z - l.position.z)
            items.filter { listOf(it.name, it.fine, it.genre, ShelfDescriber.spoken(it.name)).any(::matches) }
                .minByOrNull { distance(it.position) }
                ?.let { return ShelfDescriber.spoken(it.name) to it.position }
            if (view == null) return null
            val cs = kotlin.math.cos(view.yaw).toFloat()
            val sn = kotlin.math.sin(view.yaw).toFloat()
            var best: Pair<String, Vec3>? = null
            for (f in view.fixtures) for ((sign, modules) in f.faces) {
                val s = if (sign == "+") 1f else -1f
                val face = (f.c + s * f.d / 2).toFloat()
                for ((i, goods) in modules.withIndex()) {
                    val hit = goods.firstOrNull { matches(it.first) || matches(ShelfDescriber.spoken(it.first)) } ?: continue
                    val along = f.moduleCenter(i).toFloat()
                    val pu = if (f.axis == 0) face else along
                    val pv = if (f.axis == 0) along else face
                    val at = Vec3(pu * cs - pv * sn, l.position.y, pu * sn + pv * cs)
                    if (best == null || distance(at) < distance(best.second)) best = ShelfDescriber.spoken(hit.first) to at
                }
            }
            return best
        }

    }
}
