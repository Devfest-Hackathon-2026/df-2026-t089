package jp.devfest.shelfsight.assist

/**
 * 棚の種類（歩行中の判別の 7 分類を、同じ分類が続くまとまりにしたもの）を、話す文にする。
 * 画面の真ん中にかかるまとまりを「正面」、それより左・右を「左」「右」として言う。
 */
object ShelfSpeech {
    /** [segments] は左から右の順（[ShelfSegment.from]、「商品なし」は入っていない）。 */
    fun describe(segments: List<ShelfSegment>): String {
        if (segments.isEmpty()) return "棚が見えません。棚の方にスマホを向けてください。"
        val front = segments.firstOrNull { it.left <= 0.5f && it.right > 0.5f }
        val left = segments.filter { it.right <= 0.5f && it !== front }.map { spoken(it.category) }.distinct()
        val right = segments.filter { it.left > 0.5f && it !== front }.map { spoken(it.category) }.distinct()
        return buildString {
            if (front != null) append("正面は、${spoken(front.category)}の棚です。")
            else append("正面には棚が見えません。")
            if (left.isNotEmpty()) append("左は、${left.joinToString("、")}。")
            if (right.isNotEmpty()) append("右は、${right.joinToString("、")}。")
        }
    }

    /** 分類の名前を読みやすくする（「お菓子・おつまみ」→「お菓子やおつまみ」）。 */
    fun spoken(category: String): String = category.replace("・", "や")
}
