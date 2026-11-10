import Foundation

/// 声で言えること（それ以外はすべて AI への質問）。
enum Command: String {
    case repeatLast = "REPEAT", help = "HELP"
}

/// 聞き取った文をコマンドにする。含まれる言葉で決める。どれでもなければ質問として LLM に答えてもらう。
enum Commands {
    private static let rules: [(Command, [String])] = [
        (.repeatLast, ["もう一度", "もう1回", "もう一回", "繰り返して"]),
        (.help, ["ヘルプ", "使い方", "何ができる", "なにができる"]),
    ]

    static func parse(_ text: String) -> Command? {
        let t = text.replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "　", with: "")
        return rules.first { _, words in words.contains { t.contains($0) } }?.0
    }

    static let helpText = "1回たたくと、近くの棚や、向かっている棚の商品をお知らせします。" +
        "2回たたくと、手前の商品の名前を調べます。長押ししてから話すと、商品の場所などを質問できます。人が近づくと、その方向から音が鳴ります。"
}
