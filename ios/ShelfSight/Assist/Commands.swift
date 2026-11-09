import Foundation

/// 声で言えること。
enum Command: String {
    case shelves = "SHELVES", describe = "DESCRIBE", lookup = "LOOKUP", pause = "PAUSE", resume = "RESUME", repeatLast = "REPEAT", help = "HELP"
}

/// 聞き取った文をコマンドにする。含まれる言葉で決める（上から順に優先）。どれでもなければ質問として LLM に答えてもらう。
enum Commands {
    private static let rules: [(Command, [String])] = [
        (.pause, ["止めて", "とめて", "静かに", "黙って", "ストップ", "うるさい"]),
        (.resume, ["再開", "お知らせを始めて", "案内を始めて"]),
        (.repeatLast, ["もう一度", "もう1回", "もう一回", "繰り返して"]),
        (.help, ["ヘルプ", "使い方", "何ができる", "なにができる"]),
        (.lookup, ["これ何", "これは何", "これなに", "これはなに", "調べて"]),
        (.shelves, ["何の棚", "なんの棚", "棚の種類", "どんな棚"]),
        (.describe, ["何がある", "なにがある", "何が並んで", "なにが並んで", "棚を説明"]),
    ]

    static func parse(_ text: String) -> Command? {
        let t = text.replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "　", with: "")
        return rules.first { _, words in words.contains { t.contains($0) } }?.0
    }

    static let helpText = "長押しすると、手前の商品の名前か、棚に並んでいる商品を調べます。" +
        "2回たたくと、質問できます。棚の前では、棚の種類をその方向からお知らせします。人が近づくと、その方向から音が鳴ります。"
}
