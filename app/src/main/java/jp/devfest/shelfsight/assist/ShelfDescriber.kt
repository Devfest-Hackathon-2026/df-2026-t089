package jp.devfest.shelfsight.assist

/** 認識した 1 つの商品。位置は正立画像の 0..1。 */
data class RecognizedProduct(
    val left: Float, val top: Float, val right: Float, val bottom: Float,
    val name: String,
    /** 棚の種類（labels.txt の 26 種類）。 */
    val genre: String?,
    /** 細分類（緑茶・コーヒー・菓子パンなど）。 */
    val fine: String?,
    val sim: Float,
) {
    val cx get() = (left + right) / 2f
    val cy get() = (top + bottom) / 2f
    val height get() = bottom - top
}

/**
 * 棚 1 つ分の認識結果を、段ごとに短い文で説明する。
 * 例: 「上の段には緑茶や麦茶、真ん中の段にはウーロン茶とハーブティー、下の段には果汁ジュースと炭酸飲料があります。」
 *
 * 場違いな商品（パン棚のチョコ菓子など）は言わない。周りの商品による補正（[ContextAdjust]）に加えて、
 * 棚全体で少数しかないジャンルの商品をここで外す。
 */
object ShelfDescriber {
    /** 棚全体でこの割合未満のジャンルは、場違いな商品として外す。 */
    const val OUTLIER_SHARE = 0.15f
    /** これより多くあるジャンルは、割合が小さくても外さない（本当に混ざった棚）。 */
    const val OUTLIER_MAX_COUNT = 2
    /** 1 段で言う種類の数。 */
    const val TYPES_PER_ROW = 3
    const val MIN_PRODUCTS = 3

    /** 場違いな商品を除いた一覧。 */
    fun withoutOutliers(products: List<RecognizedProduct>): List<RecognizedProduct> {
        val withGenre = products.filter { it.genre != null }
        if (withGenre.size < 4) return products
        val counts = withGenre.groupingBy { it.genre!! }.eachCount()
        val dominant = counts.maxBy { it.value }.key
        val odd = counts.filter { (g, c) -> g != dominant && c <= OUTLIER_MAX_COUNT && c < withGenre.size * OUTLIER_SHARE }.keys
        return products.filterNot { it.genre in odd }
    }

    /** 中心の高さで段に分ける（上から順）。段の切れ目は、商品の高さの中央値の半分より大きいすき間。 */
    fun rows(products: List<RecognizedProduct>): List<List<RecognizedProduct>> {
        if (products.isEmpty()) return emptyList()
        val sorted = products.sortedBy { it.cy }
        val gap = sorted.map { it.height }.sorted()[sorted.size / 2] * 0.5f
        val out = mutableListOf(mutableListOf(sorted.first()))
        for (p in sorted.drop(1)) {
            val row = out.last()
            if (p.cy - row.map { it.cy }.average() > gap) out += mutableListOf(p) else row += p
        }
        return out
    }

    /** 説明の文。商品が少なすぎて分からないときは null。 */
    fun describe(products: List<RecognizedProduct>): String? {
        val kept = withoutOutliers(products)
        if (kept.size < MIN_PRODUCTS) return null
        val rows = rows(kept).map { typesOf(it) }.filter { it.isNotEmpty() }
        // 同じ種類が続く段はまとめる
        val merged = mutableListOf<List<String>>()
        for (r in rows) if (merged.lastOrNull() != r) merged += r
        if (merged.isEmpty()) return null
        if (merged.size == 1) return "この棚には${join(merged[0])}があります。"
        val names = rowNames(merged.size)
        return merged.indices.joinToString("、") { "${names[it]}には${join(merged[it])}" } + "があります。"
    }

    /** 段の中の種類（多い順）。同じ数なら左にあるものを先に。 */
    private fun typesOf(row: List<RecognizedProduct>): List<String> =
        row.filter { it.fine != null || it.genre != null }
            .groupBy { spoken(it.fine ?: it.genre!!) }
            .entries.sortedWith(compareBy({ -it.value.size }, { it.value.minOf { p -> p.cx } }))
            .map { it.key }.take(TYPES_PER_ROW)

    private fun join(types: List<String>): String = when (types.size) {
        1 -> types[0]
        2 -> "${types[0]}や${types[1]}"
        else -> types.dropLast(1).joinToString("、") + "、" + types.last()
    }

    fun rowNames(n: Int): List<String> = when (n) {
        1 -> listOf("この棚")
        2 -> listOf("上の段", "下の段")
        3 -> listOf("上の段", "真ん中の段", "下の段")
        else -> List(n) { i ->
            when (i) {
                0 -> "いちばん上の段"
                n - 1 -> "いちばん下の段"
                else -> "上から${i + 1}段目"
            }
        }
    }

    /** 細分類・ジャンルの名前を、声で聞いて分かりやすい言い方にする。 */
    fun spoken(category: String): String = SPOKEN[category] ?: category.substringBefore("・")

    private val SPOKEN = mapOf(
        "麦茶・ほうじ茶・その他のお茶" to "麦茶などのお茶",
        "果汁・野菜ジュース" to "ジュース",
        "スポーツ・機能性飲料" to "スポーツドリンク",
        "乳飲料・ヨーグルト飲料" to "乳飲料",
        "チューハイ・サワー・ハイボール" to "チューハイ",
        "日本酒・焼酎" to "日本酒や焼酎",
        "ワイン・洋酒" to "ワインや洋酒",
        "食パン・調理パン" to "食パン",
        "惣菜・サラダ" to "お惣菜",
        "レトルト・缶詰・インスタント" to "レトルト食品",
        "チルドデザート・ヨーグルト" to "デザート",
        "洋菓子・ケーキ" to "洋菓子",
        "ビスケット・クッキー" to "クッキー",
        "ポテトチップス・スナック" to "スナック菓子",
        "せんべい・米菓" to "おせんべい",
        "キャンディ・ガム・タブレット" to "飴やガム",
        "ナッツ・おつまみ・珍味" to "おつまみ",
        "駄菓子・食玩" to "駄菓子",
        "シリアル・栄養補助食品" to "栄養補助食品",
        "メイク化粧品" to "化粧品",
        "ヘアケア・ボディケア" to "ヘアケア用品",
        "オーラルケア" to "歯ブラシなど",
        "ティッシュ・生理用品・紙製品" to "ティッシュなど",
        "医薬品・救急用品" to "救急用品",
        "洗剤・日用雑貨" to "日用雑貨",
        "電池・スマホ用品・雑貨" to "電池やスマホ用品",
        "衣料・靴下" to "靴下など",
        "ギフトカード・その他" to "ギフトカード",
        "ソフトドリンク" to "飲み物",
        "栄養・エナジードリンク" to "エナジードリンク",
        "紙パック飲料・乳製品" to "紙パックの飲み物",
        "酒" to "お酒",
        "弁当・おにぎり" to "お弁当やおにぎり",
        "サンドイッチ・サラダ・惣菜" to "サンドイッチやお惣菜",
        "チョコ・ビスケット" to "チョコやビスケット",
        "グミ・キャンディ" to "グミや飴",
        "和洋菓子" to "お菓子",
        "おつまみ・ナッツ" to "おつまみ",
        "日用品・衛生用品" to "日用品",
        "スマホ用品・雑貨" to "スマホ用品",
        "靴下・衣料" to "靴下など",
    )
}
