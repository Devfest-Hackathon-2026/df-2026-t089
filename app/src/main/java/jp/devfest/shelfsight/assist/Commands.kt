package jp.devfest.shelfsight.assist

/** 声で言えること。 */
enum class Command { SHELVES, DESCRIBE, LOOKUP, PAUSE, RESUME, REPEAT, HELP }

/** 聞き取った文をコマンドにする。言い方は自由なので、含まれる言葉で決める（上から順に優先）。どれでもなければ質問として LLM に答えてもらう。 */
object Commands {
    private val RULES: List<Pair<Command, List<String>>> = listOf(
        Command.PAUSE to listOf("止めて", "とめて", "静かに", "黙って", "ストップ", "うるさい"),
        Command.RESUME to listOf("再開", "お知らせを始めて", "案内を始めて"),
        Command.REPEAT to listOf("もう一度", "もう1回", "もう一回", "繰り返して"),
        Command.HELP to listOf("ヘルプ", "使い方", "何ができる", "なにができる"),
        Command.LOOKUP to listOf("これ何", "これは何", "これなに", "これはなに", "調べて"),
        Command.SHELVES to listOf("何の棚", "なんの棚", "棚の種類", "どんな棚"),
        Command.DESCRIBE to listOf("何がある", "なにがある", "何が並んで", "なにが並んで", "棚を説明"),
    )

    fun parse(text: String): Command? {
        val t = text.replace(" ", "").replace("　", "")
        return RULES.firstOrNull { (_, words) -> words.any { it in t } }?.first
    }

    const val HELP_TEXT = "画面の下のボタンを押すと、正面の棚の種類を言います。画面を長押しすると、画面の真ん中の商品の名前を調べます。" +
        "2回たたくと、質問できます。人が近づくと、その方向から音が鳴ります。"
}
