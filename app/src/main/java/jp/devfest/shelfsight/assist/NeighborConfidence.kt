package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.ln

/**
 * ダッシュボードの「信頼度調整」（products/sku26n/neighbor_confidence.py）。カテゴリは 1 位のまま、周りの判定との一致で確からしさを付け直す。
 * 特徴は [1 位の logit, 周りがあるか, 周りの 1 位が同じ割合, 周りでのそのカテゴリの平均確率] で、重みは検証写真で求めたロジスティック回帰。
 *
 * PC では同じ写真の周りの枠を使うが、アプリは切り抜きを 1 枚ずつ優先度順に判定するので同じ写真の枠はそろわない。
 * 代わりに、地図で近く（枠の長辺の [radius] 倍以内）にある判定済みの切り抜きを周りとする。
 */
class NeighborConfidence(private val coef: FloatArray?, private val intercept: Float, private val k: Int, private val radius: Float) {
    private class Judged(val position: Vec3, val top: Map<String, Float>, val first: String)

    private val judged = ArrayDeque<Judged>()

    /** [top] は確率の高い順の (カテゴリ, 確率)。調整した確からしさを返し、この判定を次の周りとして覚える。重みがなければ 1 位の確率のまま。 */
    @Synchronized
    fun adjust(position: Vec3, sizeM: Float, top: List<Pair<String, Float>>): Float {
        val cat = top[0].first
        val near = judged.asSequence().map { it to it.position.distanceTo(position) }
            // 同じ場所の判定は、同じ商品をもう一度見ただけなので周りに入れない
            .filter { (_, d) -> d >= ShelfMap.SAME_SPOT_M && d <= radius * sizeM }
            .sortedBy { it.second }.take(k).map { it.first }.toList()
        judged.addLast(Judged(position, top.toMap(), cat))
        if (judged.size > MAX_JUDGED) judged.removeFirst()
        return score(top, near.map { it.first }, near.map { it.top })
    }

    /**
     * 同じ写真の枠どうしで調整する（PC の neighbor_confidence.py とまったく同じ計算）。
     * [centers] と [longSides] はピクセル、[tops] は枠ごとの確率の高い順の (カテゴリ, 確率)。
     */
    fun inImage(centers: List<Pair<Float, Float>>, longSides: List<Float>, tops: List<List<Pair<String, Float>>>): FloatArray = FloatArray(tops.size) { i ->
        val near = tops.indices.filter { it != i }
            .map { it to kotlin.math.hypot(centers[it].first - centers[i].first, centers[it].second - centers[i].second) }
            .filter { it.second <= radius * longSides[i] }.sortedBy { it.second }.take(k).map { it.first }
        score(tops[i], near.map { tops[it][0].first }, near.map { tops[it].toMap() })
    }

    private fun score(top: List<Pair<String, Float>>, firsts: List<String>, nearTops: List<Map<String, Float>>): Float {
        val (cat, prob) = top[0]
        val w = coef ?: return prob
        val p = prob.coerceIn(1e-4f, 1 - 1e-4f)
        val f = floatArrayOf(
            ln(p / (1 - p)),
            if (firsts.isEmpty()) 0f else 1f,
            if (firsts.isEmpty()) 0f else firsts.count { it == cat }.toFloat() / firsts.size,
            if (nearTops.isEmpty()) 0f else nearTops.map { it[cat] ?: 0f }.average().toFloat(),
        )
        var z = intercept
        for (i in f.indices) z += w[i] * f[i]
        return 1f / (1f + exp(-z))
    }

    companion object {
        private const val MAX_JUDGED = 3000

        /** category_head.json の "confidence"（tools/export_category_head.py が書き出す）。なければ調整しない。 */
        fun parse(o: JSONObject?): NeighborConfidence {
            val coef = o?.optJSONArray("coef")?.takeIf { it.length() == 4 } ?: return NeighborConfidence(null, 0f, 0, 0f)
            return NeighborConfidence(FloatArray(4) { coef.getDouble(it).toFloat() }, o.getDouble("intercept").toFloat(), o.getInt("k"), o.getDouble("radius").toFloat())
        }
    }
}
