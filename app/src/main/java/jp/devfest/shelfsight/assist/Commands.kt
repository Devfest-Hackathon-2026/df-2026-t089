package jp.devfest.shelfsight.assist

/** 声で言えること（それ以外はすべて AI への質問）。 */
enum class Command { REPEAT, HELP }

/** 聞き取った文をコマンドにする。言い方は自由なので、含まれる言葉で決める。どれでもなければ質問として AI に答えてもらう。 */
object Commands {
    private val RULES: List<Pair<Command, List<String>>> = listOf(
        Command.REPEAT to listOf("もう一度", "もう1回", "もう一回", "繰り返して"),
        Command.HELP to listOf("ヘルプ", "使い方", "何ができる", "なにができる"),
    )

    fun parse(text: String): Command? {
        val t = text.replace(" ", "").replace("　", "")
        return RULES.firstOrNull { (_, words) -> words.any { it in t } }?.first
    }

    const val HELP_TEXT = "1回たたくと、近くの棚や、向かっている棚の商品をお知らせします。" +
        "2回たたくと、手前の商品の名前を調べます。長押ししてから話すと、商品の場所などを質問できます。人が近づくと、その方向から音が鳴ります。"
}
