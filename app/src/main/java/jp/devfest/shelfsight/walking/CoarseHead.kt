package jp.devfest.shelfsight.walking

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import org.json.JSONObject

/** 画像の中の切り出し範囲。右端と下端は含まない。 */
data class CropRect(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width get() = right - left
    val height get() = bottom - top
}

/**
 * 歩行中の大まかなカテゴリを、画像の列ごとに決める部分。画像の特徴を出すモデルとは分けてある。
 *
 * tools/coarse_experiment.py の --export-head で書き出した JSON を読む。
 * 切り出し方と平滑化は PC の学習と同じにしてある（変えると精度が保証できない）。
 */
class CoarseHead(
    val modelName: String,
    val classes: List<String>,
    private val weights: Array<FloatArray>,
    private val bias: FloatArray,
    val columns: Int,
    private val windowColumns: Int,
    private val switchCost: Float,
) {
    init {
        require(weights.size == classes.size && bias.size == classes.size) { "weights do not match the classes" }
    }

    val embeddingSize get() = weights[0].size

    /**
     * 列 [column] を判別するための切り出し範囲。列を中心に [windowColumns] 列分の幅をとり、画像の端で切る。
     * モデルは正方形しか受け取らないので、PC の前処理（短辺を合わせて中央を切り出す）と同じく中央の正方形を返す。
     */
    fun cropFor(column: Int, imageWidth: Int, imageHeight: Int): CropRect {
        val center = (column + 0.5) / columns * imageWidth
        val half = windowColumns / 2.0 / columns * imageWidth
        val left = max(0.0, center - half).toInt()
        val right = min(imageWidth.toDouble(), center + half).toInt()
        val width = right - left
        return if (width <= imageHeight) {
            val top = (imageHeight - width) / 2
            CropRect(left, top, right, top + width)
        } else {
            val inset = (width - imageHeight) / 2
            CropRect(left + inset, 0, left + inset + imageHeight, imageHeight)
        }
    }

    /** モデルが出した特徴から、カテゴリごとの確率を返す。 */
    fun probabilities(embedding: FloatArray): FloatArray {
        require(embedding.size == embeddingSize) { "embedding has ${embedding.size} values, expected $embeddingSize" }
        var norm = 0.0
        for (v in embedding) norm += v * v
        val scale = 1.0 / max(sqrt(norm), 1e-12)
        val logits = DoubleArray(classes.size) { k ->
            var s = bias[k].toDouble()
            val w = weights[k]
            for (i in embedding.indices) s += w[i] * embedding[i] * scale
            s
        }
        val top = logits.max()
        val e = logits.map { exp(it - top) }
        val sum = e.sum()
        return FloatArray(classes.size) { (e[it] / sum).toFloat() }
    }

    /**
     * 列を左から右へたどり、最もありそうなカテゴリの並びを返す。カテゴリが変わるたびに [switchCost] の罰を与えるので、
     * 1 列だけ別のカテゴリになるような揺れが消える。
     */
    fun smooth(probs: List<FloatArray>): IntArray {
        val n = probs.size
        if (n == 0) return IntArray(0)
        val k = classes.size
        val logp = probs.map { p -> DoubleArray(k) { ln(p[it].toDouble().coerceIn(1e-6, 1.0)) } }
        var score = logp[0].copyOf()
        val back = Array(n) { IntArray(k) }
        for (i in 1 until n) {
            val next = DoubleArray(k)
            for (to in 0 until k) {
                var best = Double.NEGATIVE_INFINITY
                var from = 0
                for (j in 0 until k) {
                    val s = score[j] - if (j == to) 0.0 else switchCost.toDouble()
                    if (s > best) {
                        best = s
                        from = j
                    }
                }
                back[i][to] = from
                next[to] = best + logp[i][to]
            }
            score = next
        }
        val path = IntArray(n)
        path[n - 1] = score.indices.maxBy { score[it] }
        for (i in n - 1 downTo 1) path[i - 1] = back[i][path[i]]
        return path
    }

    companion object {
        fun parse(json: String): CoarseHead {
            val o = JSONObject(json)
            val classes = o.getJSONArray("classes").let { a -> List(a.length()) { a.getString(it) } }
            val w = o.getJSONArray("weights")
            val weights = Array(w.length()) { r ->
                val row = w.getJSONArray(r)
                FloatArray(row.length()) { row.getDouble(it).toFloat() }
            }
            val b = o.getJSONArray("bias")
            return CoarseHead(
                modelName = o.getString("model"),
                classes = classes,
                weights = weights,
                bias = FloatArray(b.length()) { b.getDouble(it).toFloat() },
                columns = o.getInt("columns"),
                windowColumns = o.getInt("window_columns"),
                switchCost = o.getDouble("switch_cost").toFloat(),
            )
        }
    }
}
