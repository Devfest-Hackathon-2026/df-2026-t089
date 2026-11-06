package jp.devfest.shelfsight.assist

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * 周りの商品のカテゴリで判定を補正する（products/context.py の移植）。同じカテゴリの商品は棚で固まっているので、
 * 周りと違うカテゴリの候補にペナルティをかける。パン棚のブラックサンダーのような、似ているだけの候補が選ばれにくくなる。
 */
object ContextAdjust {
    private const val NB = 8        // 周りとみなす近い商品の数
    private const val ROW = 3.0     // 縦方向の距離の重み（別の段の商品は遠い扱い）
    private const val TEMP = 20.0   // 類似度 → 確率の鋭さ
    private const val EPS = 0.02    // 周りに一度も出ないカテゴリでも確率 0 にしない

    /** 候補（名前のキーと類似度、高い順）。 */
    data class Candidate(val key: String, val sim: Float)

    /** 選んだ候補。key が null なら候補なし。 */
    data class Pick(val key: String?, val score: Double, val sim: Float)

    /** 1 段の補正。[categoryOf] がキーをカテゴリに変える（不明なら null）。 */
    class Level(val categoryOf: (String) -> String?, val lambda: Double)

    /** [centers] は画像内の位置（0..1 の x, y）。 */
    fun adjust(centers: List<Pair<Float, Float>>, cands: List<List<Candidate>>, levels: List<Level>): List<Pick> {
        val n = cands.size
        val first = cands.map { c -> c.firstOrNull()?.let { Pick(it.key, it.sim.toDouble(), it.sim) } ?: Pick(null, 0.0, 0f) }
        if (n == 0 || levels.none { it.lambda != 0.0 }) return first

        // 候補ごとの確率（類似度の softmax）
        val w = cands.map { c ->
            if (c.isEmpty()) DoubleArray(0) else {
                val top = c.maxOf { it.sim }.toDouble()
                val e = DoubleArray(c.size) { exp(TEMP * (c[it].sim - top)) }
                val s = e.sum().coerceAtLeast(1e-12)
                DoubleArray(e.size) { e[it] / s }
            }
        }
        val nb = neighbors(centers)
        val score = cands.map { c -> DoubleArray(c.size) { c[it].sim.toDouble() } }
        for (level in levels) {
            if (level.lambda == 0.0) continue
            val ids = HashMap<String, Int>()
            val cat = cands.map { c -> IntArray(c.size) { k -> level.categoryOf(c[k].key)?.let { ids.getOrPut(it) { ids.size } } ?: -1 } }
            if (ids.isEmpty()) continue
            val m = ids.size
            val p = Array(n) { DoubleArray(m) }
            for (i in 0 until n) for (k in cat[i].indices) if (cat[i][k] >= 0) p[i][cat[i][k]] += w[i][k]
            for (i in 0 until n) {
                val q = DoubleArray(m)
                for ((j, wt) in nb[i]) for (c in 0 until m) q[c] += wt * p[j][c]
                val total = q.sum() + EPS * m
                for (k in cat[i].indices) if (cat[i][k] >= 0) score[i][k] += level.lambda * ln((q[cat[i][k]] + EPS) / total)
            }
        }
        return cands.mapIndexed { i, c ->
            if (c.isEmpty()) Pick(null, 0.0, 0f) else {
                val b = score[i].indices.maxBy { score[i][it] }
                Pick(c[b].key, score[i][b], c[b].sim)
            }
        }
    }

    /** 各商品の近い [NB] 個と重み（1/(距離+0.02) を合計 1 に）。 */
    private fun neighbors(centers: List<Pair<Float, Float>>): List<List<Pair<Int, Double>>> {
        val n = centers.size
        if (n < 2) return List(n) { emptyList() }
        val k = minOf(NB, n - 1)
        return List(n) { i ->
            val near = (0 until n).filter { it != i }.map { j ->
                val dx = (centers[i].first - centers[j].first).toDouble()
                val dy = (centers[i].second - centers[j].second).toDouble() * ROW
                j to sqrt(dx * dx + dy * dy)
            }.sortedBy { it.second }.take(k)
            val ws = near.map { it.first to 1.0 / (it.second + 0.02) }
            val s = ws.sumOf { it.second }
            ws.map { it.first to it.second / s }
        }
    }
}
