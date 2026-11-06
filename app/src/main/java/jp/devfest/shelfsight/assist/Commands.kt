package jp.devfest.shelfsight.assist

/** 声で言えること。 */
enum class Command { SHELVES, DESCRIBE, LOOKUP, PAUSE, RESUME, REPEAT, HELP }

/** 聞き取った文をコマンドにする。言い方は自由なので、含まれる言葉で決める（上から順に優先）。 */
object Commands {
    private val RULES: List<Pair<Command, List<String>>> = listOf(
        Command.PAUSE to listOf("止めて", "とめて", "止まって", "静かに", "黙って", "ストップ", "うるさい", "オフ"),
        Command.RESUME to listOf("再開", "始めて", "はじめて", "オン", "続けて"),
        Command.REPEAT to listOf("もう一度", "もう1回", "もう一回", "繰り返", "なんて"),
        Command.HELP to listOf("ヘルプ", "使い方", "何ができる", "なにができる"),
        Command.LOOKUP to listOf("これ何", "これは何", "これなに", "これはなに", "調べて", "商品名", "どこの", "何の商品"),
        Command.SHELVES to listOf("何の棚", "なんの棚", "どこが何", "周り", "まわり", "棚の種類", "どんな棚"),
        Command.DESCRIBE to listOf("何がある", "なにがある", "説明", "教えて", "商品", "並んで", "段"),
    )

    fun parse(text: String): Command? {
        val t = text.replace(" ", "").replace("　", "")
        return RULES.firstOrNull { (_, words) -> words.any { it in t } }?.first
    }

    const val HELP_TEXT = "棚の前では、何の棚かを、その方向からお知らせします。棚に近づくと、並んでいる商品を説明します。" +
        "商品を一つ手前に近づけて、画面を長押しすると、商品名を調べます。人が近づくと、その方向から音が鳴ります。" +
        "画面を軽く叩くと、話しかけられます。"
}
