import Foundation

/// 認識した 1 つの商品。位置は正立画像の 0..1。
struct RecognizedProduct: Equatable {
    let left: Float
    let top: Float
    let right: Float
    let bottom: Float
    let name: String
    /// 棚の種類（labels.txt の 26 種類）。
    let genre: String?
    /// 細分類（緑茶・コーヒー・菓子パンなど）。
    let fine: String?
    let sim: Float

    var cx: Float { (left + right) / 2 }
    var cy: Float { (top + bottom) / 2 }
    var height: Float { bottom - top }
}

/**
 * 棚 1 つ分の認識結果を、段ごとに短い文で説明する。
 * 場違いな商品（パン棚のチョコ菓子など）は言わない。7 つのまとまり（categories.json）と細分類で、棚の中で少ない種類の商品を split で外す。
 */
enum ShelfDescriber {
    /// 棚の中でこの割合に届かない分類（7 分類・細分類）の商品は、場違いな商品として外す。
    static let outlierShare: Float = 0.25
    /// 1 段で言う種類の数。
    static let typesPerRow = 3
    static let minProducts = 3

    /// 商品のジャンル（26 種類）→ 7 分類。
    final class Groups {
        private let groupOf: [String: String]

        init(_ groupOf: [String: String]) { self.groupOf = groupOf }

        func of(_ p: RecognizedProduct) -> String? { of(p.genre) }
        func of(_ genre: String?) -> String? { genre.flatMap { groupOf[$0] } }

        /// coarse/categories.json（tools/coarse_categories.json と同じ）を読む。
        static func parse(_ data: Data) -> Groups {
            var map: [String: String] = [:]
            for (group, value) in JSON.object(data) ?? [:] where !group.hasPrefix("_") {
                for genre in (value as? [String]) ?? [] { map[genre] = group }
            }
            return Groups(map)
        }

        /// アプリに入れた categories.json（app/src/main/assets/coarse/categories.json と同じもの）を読む。
        static func fromBundle() -> Groups {
            guard let url = Bundle.main.url(forResource: "categories", withExtension: "json"),
                  let data = try? Data(contentsOf: url) else { return Groups([:]) }
            return parse(data)
        }
    }

    /// 残した商品と、場違いとして外した商品。mainGroup は棚の主な分類、mainKinds は残した細分類。
    struct Split {
        let kept: [RecognizedProduct]
        let removed: [RecognizedProduct]
        let mainGroup: String?
        var mainKinds: [String] = []
    }

    /// 細分類（緑茶・グミ・ポテトチップスなど）。なければジャンル。
    private static func kindOf(_ p: RecognizedProduct) -> String? { p.fine ?? p.genre }

    /**
     * 場違いな商品を 2 段階で外す。
     * 1. 7 分類: 主な分類（shelfGroup が商品にも十分あればそれ、なければいちばん多い分類）と、outlierShare 以上ある分類だけ残す
     * 2. 細分類: 残った商品の中で、いちばん多い細分類と outlierShare 以上ある細分類だけ残す
     * 分類の分からない商品も外す。
     */
    static func split(_ products: [RecognizedProduct], _ groups: Groups, shelfGroup: String? = nil) -> Split {
        let withGroup = products.filter { groups.of($0) != nil }
        if withGroup.isEmpty { return Split(kept: [], removed: products, mainGroup: nil) }
        let counts = withGroup.orderedGroups { groups.of($0)! }.map { (key: $0.key, count: $0.values.count) }
        let countOf = Dictionary(uniqueKeysWithValues: counts.map { ($0.key, $0.count) })
        let main: String
        if let s = shelfGroup, (countOf[s] ?? 0) >= minProducts {
            main = s
        } else {
            main = counts.max { $0.count < $1.count }!.key
        }
        let keepGroups = Set(counts.filter { $0.key == main || Float($0.count) >= Float(withGroup.count) * outlierShare }.map(\.key))
        let inGroup = products.filter { groups.of($0).map { keepGroups.contains($0) } ?? false }

        let kinds = inGroup.orderedGroups { kindOf($0)! }.map { (key: $0.key, count: $0.values.count) }
        let mainKind = kinds.max { $0.count < $1.count }!.key
        let keepKinds = kinds.filter { $0.key == mainKind || Float($0.count) >= Float(inGroup.count) * outlierShare }
            .stableSorted { $0.count > $1.count }.map(\.key)
        var kept: [RecognizedProduct] = []
        var removed: [RecognizedProduct] = []
        for p in products {
            if inGroup.contains(p), let k = kindOf(p), keepKinds.contains(k) { kept.append(p) } else { removed.append(p) }
        }
        return Split(kept: kept, removed: removed, mainGroup: main, mainKinds: keepKinds)
    }

    /// 中心の高さで段に分ける（上から順）。段の切れ目は、商品の高さの中央値の半分より大きいすき間。
    static func rows(_ products: [RecognizedProduct]) -> [[RecognizedProduct]] {
        if products.isEmpty { return [] }
        let sorted = products.stableSorted { $0.cy < $1.cy }
        let gap = sorted.map(\.height).sorted()[sorted.count / 2] * 0.5
        var out: [[RecognizedProduct]] = [[sorted[0]]]
        for p in sorted.dropFirst() {
            let row = out[out.count - 1]
            let average = row.map(\.cy).reduce(0, +) / Float(row.count)
            if p.cy - average > gap { out.append([p]) } else { out[out.count - 1].append(p) }
        }
        return out
    }

    /// 説明の文（場違いな商品は split で外してから渡す）。商品が少なすぎて分からないときは nil。
    static func describe(_ kept: [RecognizedProduct]) -> String? {
        if kept.count < minProducts { return nil }
        let rowTypes = rows(kept).map { typesOf($0) }.filter { !$0.isEmpty }
        // 同じ種類が続く段はまとめる
        var merged: [[String]] = []
        for r in rowTypes where merged.last != r { merged.append(r) }
        if merged.isEmpty { return nil }
        if merged.count == 1 { return "この棚には\(join(merged[0]))があります。" }
        let names = rowNames(merged.count)
        return merged.indices.map { "\(names[$0])には\(join(merged[$0]))" }.joined(separator: "、") + "があります。"
    }

    /// 段の中の種類（多い順）。同じ数なら左にあるものを先に。
    private static func typesOf(_ row: [RecognizedProduct]) -> [String] {
        row.filter { $0.fine != nil || $0.genre != nil }
            .orderedGroups { spoken($0.fine ?? $0.genre!) }
            .stableSorted { a, b in
                if a.values.count != b.values.count { return a.values.count > b.values.count }
                return a.values.map(\.cx).min()! < b.values.map(\.cx).min()!
            }
            .prefix(typesPerRow).map(\.key)
    }

    private static func join(_ types: [String]) -> String {
        switch types.count {
        case 1: return types[0]
        case 2: return "\(types[0])や\(types[1])"
        default: return types.dropLast().joined(separator: "、") + "、" + types.last!
        }
    }

    static func rowNames(_ n: Int) -> [String] {
        switch n {
        case 1: return ["この棚"]
        case 2: return ["上の段", "下の段"]
        case 3: return ["上の段", "真ん中の段", "下の段"]
        default:
            return (0..<n).map { i in
                if i == 0 { return "いちばん上の段" }
                if i == n - 1 { return "いちばん下の段" }
                return "上から\(i + 1)段目"
            }
        }
    }

    /// 細分類・ジャンルの名前を、声で聞いて分かりやすい言い方にする。
    static func spoken(_ category: String) -> String {
        if let s = spokenNames[category] { return s }
        return String(category.split(separator: "・", maxSplits: 1, omittingEmptySubsequences: false).first ?? Substring(category))
    }

    private static let spokenNames: [String: String] = [
        "麦茶・ほうじ茶・その他のお茶": "麦茶などのお茶",
        "果汁・野菜ジュース": "ジュース",
        "スポーツ・機能性飲料": "スポーツドリンク",
        "乳飲料・ヨーグルト飲料": "乳飲料",
        "チューハイ・サワー・ハイボール": "チューハイ",
        "日本酒・焼酎": "日本酒や焼酎",
        "ワイン・洋酒": "ワインや洋酒",
        "食パン・調理パン": "食パン",
        "惣菜・サラダ": "お惣菜",
        "レトルト・缶詰・インスタント": "レトルト食品",
        "チルドデザート・ヨーグルト": "デザート",
        "洋菓子・ケーキ": "洋菓子",
        "ビスケット・クッキー": "クッキー",
        "ポテトチップス・スナック": "スナック菓子",
        "せんべい・米菓": "おせんべい",
        "キャンディ・ガム・タブレット": "飴やガム",
        "ナッツ・おつまみ・珍味": "おつまみ",
        "駄菓子・食玩": "駄菓子",
        "シリアル・栄養補助食品": "栄養補助食品",
        "メイク化粧品": "化粧品",
        "ヘアケア・ボディケア": "ヘアケア用品",
        "オーラルケア": "歯ブラシなど",
        "ティッシュ・生理用品・紙製品": "ティッシュなど",
        "医薬品・救急用品": "救急用品",
        "洗剤・日用雑貨": "日用雑貨",
        "電池・スマホ用品・雑貨": "電池やスマホ用品",
        "衣料・靴下": "靴下など",
        "ギフトカード・その他": "ギフトカード",
        "ソフトドリンク": "飲み物",
        "栄養・エナジードリンク": "エナジードリンク",
        "紙パック飲料・乳製品": "紙パックの飲み物",
        "酒": "お酒",
        "弁当・おにぎり": "お弁当やおにぎり",
        "サンドイッチ・サラダ・惣菜": "サンドイッチやお惣菜",
        "チョコ・ビスケット": "チョコやビスケット",
        "グミ・キャンディ": "グミや飴",
        "和洋菓子": "お菓子",
        "おつまみ・ナッツ": "おつまみ",
        "日用品・衛生用品": "日用品",
        "スマホ用品・雑貨": "スマホ用品",
        "靴下・衣料": "靴下など",
    ]
}
