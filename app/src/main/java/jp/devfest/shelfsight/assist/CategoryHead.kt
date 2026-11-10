package jp.devfest.shelfsight.assist

import kotlin.math.exp

/**
 * MobileCLIP の埋め込みから 122 カテゴリの確率を出す分類層（全結合 → ReLU → 全結合）。
 * [weights] は tools/export_category_head.py の順（1 層目の重み・バイアス、2 層目の重み・バイアス。重みは出力ごとの行）。
 */
internal class CategoryHead(private val dim: Int, private val hidden: Int, private val classes: Int, private val weights: FloatArray) {
    init {
        require(weights.size == (dim + 1) * hidden + (hidden + 1) * classes) { "category head size ${weights.size}" }
    }

    /** いちばん確率の高いカテゴリの番号と、その確率。[embedding] は長さ 1 に正規化したもの。 */
    fun predict(embedding: FloatArray): Pair<Int, Float> {
        val p = probabilities(embedding)
        val best = p.indices.maxBy { p[it] }
        return best to p[best]
    }

    /** カテゴリごとの確率（softmax）。 */
    fun probabilities(embedding: FloatArray): FloatArray {
        val middle = layer(embedding, dim, hidden, 0)
        for (i in middle.indices) if (middle[i] < 0f) middle[i] = 0f
        val logits = layer(middle, hidden, classes, (dim + 1) * hidden)
        val top = logits.max()
        var sum = 0f
        for (v in logits) sum += exp(v - top)
        return FloatArray(classes) { exp(logits[it] - top) / sum }
    }

    private fun layer(input: FloatArray, inputs: Int, outputs: Int, offset: Int): FloatArray {
        val bias = offset + inputs * outputs
        return FloatArray(outputs) { row ->
            var value = weights[bias + row]
            val start = offset + row * inputs
            for (k in 0 until inputs) value += weights[start + k] * input[k]
            value
        }
    }
}
