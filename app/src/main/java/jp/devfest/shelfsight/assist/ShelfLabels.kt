package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.sqrt

/**
 * 1 枚の画像の中の 1 つの棚。歩行中の判別（画面を 8 列に分け、列ごとに 7 分類）で、同じ分類が続く列のまとまり。
 * [left]..[right] は画像の横位置（0..1）。[fromMap] は地図に貯めた棚のラベルで分類を直した列を含むとき true。
 */
data class ShelfSegment(val category: String, val confidence: Float, val left: Float, val right: Float, val fromMap: Boolean = false) {
    companion object {
        /** 歩行中の判別の「棚ではない」分類（coarse の head.json の classes）。 */
        const val NO_PRODUCT = "商品なし"

        /** 列ごとの分類（左から右）を、同じ分類が続くまとまりに分ける。「商品なし」は棚にしない。 */
        fun from(columns: List<String>, confidence: FloatArray, fromMap: BooleanArray = BooleanArray(columns.size)): List<ShelfSegment> {
            val n = columns.size
            val out = mutableListOf<ShelfSegment>()
            var start = 0
            for (i in 1..n) {
                if (i < n && columns[i] == columns[start]) continue
                if (columns[start] != NO_PRODUCT) {
                    val conf = (start until i).map { confidence[it] }.average().toFloat()
                    out += ShelfSegment(columns[start], conf, start.toFloat() / n, i.toFloat() / n, (start until i).any { fromMap[it] })
                }
                start = i
            }
            return out
        }
    }
}

/** 7 分類の色（画面の帯・棚の名前・地図）。 */
val SHELF_COLORS: Map<String, Int> = mapOf(
    "飲み物・お酒" to 0xFF1565C0.toInt(), "冷凍・アイス" to 0xFF00838F.toInt(), "弁当・パン" to 0xFFBF360C.toInt(),
    "お菓子・おつまみ" to 0xFFC2185B.toInt(), "食品" to 0xFF6D4C41.toInt(), "日用品・雑貨" to 0xFF455A64.toInt(),
    "商品なし" to 0xFF212121.toInt(),
)

/** 地図に置いた棚のラベル 1 つ。同じ場所で判別するたびに、分類ごとの票を足す。 */
class ShelfMark(var position: Vec3, var facing: Vec3, var lastSeen: Long) {
    val votes = HashMap<String, Int>()
    val total: Int get() = votes.values.sum()
    val label: String get() = votes.maxBy { it.value }.key
    val share: Float get() = votes.getValue(label) / total.toFloat()
}

/** ある場所の棚のラベル（地図から）。[groups] は 7 分類の票の割合（合計 1）。 */
data class ShelfPrior(val label: String, val share: Float, val votes: Int, val groups: Map<String, Float>)

/** 画面に描く、地図の棚のラベル。[u], [v] は正立画像の 0..1。 */
data class VisibleShelfLabel(val u: Float, val v: Float, val label: String, val share: Float, val votes: Int)

/**
 * 棚の種類（7 分類）を AR 空間の棚の上に貯める（アプリを閉じると消える）。
 *
 * 1 秒ごとに、画面の 8 列それぞれの分類を、その列の棚の位置（深度か ARCore の壁）に票として足す。
 * 同じ棚をもう一度向いたとき、カメラの位置と向きからその場所の票を引き、
 * - 1 枚ごとにぶれる列の分類を直す（[correct]）
 * - 商品の名前の候補を、その棚の分類に合うものへ寄せる（[MapBias]）
 * のに使う。背中合わせの棚の裏側（反対の向きから見た票）は使わない。
 */
class ShelfLabelMap {
    private val marks = ArrayList<ShelfMark>()

    @get:Synchronized val size: Int get() = marks.size

    /** 地図を描くための写し。 */
    @Synchronized
    fun snapshot(): List<ShelfMark> = marks.map { m -> ShelfMark(m.position, m.facing, m.lastSeen).also { it.votes.putAll(m.votes) } }

    /** [category] を [position] の棚に 1 票足す。[camera] は見たときのカメラの位置。 */
    @Synchronized
    fun add(category: String, position: Vec3, camera: Vec3, now: Long) {
        val facing = ShelfMap.towards(position, camera)
        val near = marks.filter { it.position.distanceTo(position) < MERGE_M && ShelfMap.facesSameWay(it.facing, facing) }
            .minByOrNull { it.position.distanceTo(position) }
        val mark = near ?: ShelfMark(position, facing, now).also {
            marks += it
            if (marks.size > MAX_MARKS) marks.remove(marks.minBy { m -> m.lastSeen })
        }
        if (near != null) {
            // 位置と向きは、票の数で重みを付けてならす
            val k = 1f / (mark.total + 1)
            mark.position = Vec3(
                mark.position.x + (position.x - mark.position.x) * k,
                mark.position.y + (position.y - mark.position.y) * k,
                mark.position.z + (position.z - mark.position.z) * k,
            )
            mark.facing = unit(Vec3(mark.facing.x * mark.total + facing.x, 0f, mark.facing.z * mark.total + facing.z)) ?: mark.facing
        }
        mark.votes[category] = (mark.votes[category] ?: 0) + 1
        mark.lastSeen = now
    }

    /** [position] の棚のラベル。[camera] から見て表側の、いちばん近い票のまとまり。なければ null。 */
    @Synchronized
    fun at(position: Vec3, camera: Vec3): ShelfPrior? {
        val view = ShelfMap.towards(position, camera)
        val m = marks.filter { it.position.distanceTo(position) < LOOKUP_M && ShelfMap.facesSameWay(it.facing, view) }
            .minByOrNull { it.position.distanceTo(position) } ?: return null
        val total = m.total.toFloat()
        return ShelfPrior(m.label, m.share, m.total, m.votes.mapValues { it.value / total })
    }

    /** 今の画面に入る棚のラベル。 */
    fun visible(g: CaptureGeometry): List<VisibleShelfLabel> = synchronized(this) {
        marks.map { Triple(it.position, it.label, it.share to it.total) }
    }.mapNotNull { (pos, label, st) ->
        ShelfMap.project(pos, g)?.takeIf { it[2] in ShelfMap.MIN_DEPTH_M..ShelfMap.MAX_DEPTH_M }
            ?.let { VisibleShelfLabel(it[0], it[1], label, st.first, st.second) }
    }

    companion object {
        /** これより近い票は同じ棚の場所としてまとめる（8 列の 1 列は 1m 先で 10cm ほど）。 */
        const val MERGE_M = 0.3f
        /** 引くときに探す範囲。 */
        const val LOOKUP_M = 0.5f
        /** 地図のラベルで列の分類を直すのに要る票の数と、いちばん多い分類の割合。 */
        const val MIN_VOTES = 3
        const val MIN_SHARE = 0.6f
        const val MAX_MARKS = 3000

        /**
         * 列の分類を地図のラベルで直す。地図に十分な票（[MIN_VOTES] 以上、いちばん多い分類が [MIN_SHARE] 以上）があり、
         * 今の判別と違えば地図の方を使う。直した列は true。
         */
        fun correct(columns: List<String>, priors: List<ShelfPrior?>): Pair<List<String>, BooleanArray> {
            val fixed = BooleanArray(columns.size)
            val out = columns.mapIndexed { i, c ->
                val p = priors.getOrNull(i)
                if (p != null && p.votes >= MIN_VOTES && p.share >= MIN_SHARE && p.label != c) { fixed[i] = true; p.label } else c
            }
            return out to fixed
        }

        private fun unit(v: Vec3): Vec3? {
            val l = sqrt(v.x * v.x + v.z * v.z)
            return if (l < 1e-4f) null else Vec3(v.x / l, 0f, v.z / l)
        }
    }
}
