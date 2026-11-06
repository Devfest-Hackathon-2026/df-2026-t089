package jp.devfest.shelfsight.assist

/**
 * 棚の 8 分類。棚の分類モデル（shelf_classifier.tflite、labels.txt の 26 種類）の出力を、
 * 歩きながら聞いて分かる粗さにまとめる。分類ごとの確率は、含まれる種類の確率の最大値（多ラベル・sigmoid の出力なので和ではなく最大）。
 */
enum class Shelf8(val spoken: String, val members: List<String>) {
    DRINK("飲み物", listOf("ソフトドリンク", "栄養・エナジードリンク", "紙パック飲料・乳製品")),
    ALCOHOL("お酒", listOf("酒")),
    FROZEN("冷凍食品とアイス", listOf("冷凍食品", "アイス")),
    MEAL("お弁当とお惣菜", listOf("弁当・おにぎり", "サンドイッチ・サラダ・惣菜")),
    BREAD("パンとデザート", listOf("パン", "チルドデザート")),
    SNACK("お菓子", listOf("スナック菓子", "チョコ・ビスケット", "グミ・キャンディ", "和洋菓子", "駄菓子・食玩", "おつまみ・ナッツ")),
    FOOD("カップ麺と食品", listOf("カップ麺", "加工食品")),
    DAILY("日用品", listOf("日用品・衛生用品", "化粧品", "文房具", "マスク", "スマホ用品・雑貨", "靴下・衣料", "ギフトカード", "その他")),
}

/** 26 種類の確率を 8 分類にまとめる。[labels] は labels.txt の並び。 */
class ShelfCategoryMap(labels: List<String>) {
    private val groupOf: IntArray = IntArray(labels.size) { i ->
        Shelf8.entries.indexOfFirst { labels[i].trim() in it.members }
    }

    init {
        val missing = labels.filterIndexed { i, _ -> groupOf[i] < 0 }
        require(missing.isEmpty()) { "8 分類に入っていない棚の種類: $missing" }
    }

    fun probabilities(probs26: FloatArray): FloatArray {
        val out = FloatArray(Shelf8.entries.size)
        for (i in probs26.indices) out[groupOf[i]] = maxOf(out[groupOf[i]], probs26[i])
        return out
    }

    /** いちばん確からしい分類。どれも [minProb] 未満なら null（棚でない・分からない）。 */
    fun best(probs26: FloatArray, minProb: Float = MIN_PROB): Pair<Shelf8, Float>? {
        val p = probabilities(probs26)
        val i = p.indices.maxBy { p[it] }
        return if (p[i] >= minProb) Shelf8.entries[i] to p[i] else null
    }

    companion object {
        const val MIN_PROB = 0.2f
    }
}
