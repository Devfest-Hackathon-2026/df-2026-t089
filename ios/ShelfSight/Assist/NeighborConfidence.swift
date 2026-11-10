import Foundation

/**
 * ダッシュボードの「信頼度調整」（products/sku26n/neighbor_confidence.py、Android 版の NeighborConfidence）。
 * カテゴリは 1 位のまま、周りの判定との一致で確からしさを付け直す。
 * 特徴は [1 位の logit, 周りがあるか, 周りの 1 位が同じ割合, 周りでのそのカテゴリの平均確率] で、重みは検証写真で求めたロジスティック回帰。
 * アプリは切り抜きを 1 枚ずつ優先度順に判定するので、同じ写真の周りの枠の代わりに、地図で近くにある判定済みの切り抜きを周りとする。
 */
final class NeighborConfidence {
    private struct Judged {
        let position: Vec3
        let top: [String: Float]
        let first: String
    }

    private static let maxJudged = 3000
    /// 同じ商品をもう一度見ただけとみなす距離（ShelfMap.sameSpotM と同じ。テストでカメラまわりを持ち込まないよう、ここに置く）。
    private static let sameSpotM: Float = 0.12

    private let coef: [Float]?
    private let intercept: Float
    private let k: Int
    private let radius: Float
    private let lock = NSLock()
    private var judged: [Judged] = []

    init(coef: [Float]?, intercept: Float, k: Int, radius: Float) {
        self.coef = coef
        self.intercept = intercept
        self.k = k
        self.radius = radius
    }

    /// top は確率の高い順の (カテゴリ, 確率)。調整した確からしさを返し、この判定を次の周りとして覚える。重みがなければ 1 位の確率のまま。
    func adjust(_ position: Vec3, sizeM: Float, top: [(String, Float)]) -> Float {
        lock.lock(); defer { lock.unlock() }
        let (cat, prob) = top[0]
        let near = judged.map { ($0, $0.position.distanceTo(position)) }
            // 同じ場所の判定は、同じ商品をもう一度見ただけなので周りに入れない
            .filter { $0.1 >= Self.sameSpotM && $0.1 <= radius * sizeM }
            .stableSorted { $0.1 < $1.1 }.prefix(k).map(\.0)
        judged.append(Judged(position: position, top: Dictionary(top, uniquingKeysWith: { a, _ in a }), first: cat))
        if judged.count > Self.maxJudged { judged.removeFirst() }
        guard let w = coef else { return prob }
        let p = prob.clamped(1e-4, 1 - 1e-4)
        let n = Float(near.count)
        let f: [Float] = [
            log(p / (1 - p)),
            near.isEmpty ? 0 : 1,
            near.isEmpty ? 0 : Float(near.filter { $0.first == cat }.count) / n,
            near.isEmpty ? 0 : near.reduce(0) { $0 + ($1.top[cat] ?? 0) } / n,
        ]
        var z = intercept
        for i in f.indices { z += w[i] * f[i] }
        return 1 / (1 + exp(-z))
    }

    /// category_head.json の "confidence"（tools/export_category_head.py が書き出す）。なければ調整しない。
    static func parse(_ o: [String: Any]?) -> NeighborConfidence {
        guard let coef = (o?["coef"] as? [NSNumber])?.map(\.floatValue), coef.count == 4,
              let intercept = JSON.double(o, "intercept"), let k = JSON.int(o, "k"), let radius = JSON.double(o, "radius") else {
            return NeighborConfidence(coef: nil, intercept: 0, k: 0, radius: 0)
        }
        return NeighborConfidence(coef: coef, intercept: Float(intercept), k: k, radius: Float(radius))
    }

    static func fromBundle() -> NeighborConfidence {
        let url = Bundle.main.url(forResource: "category_head", withExtension: "json", subdirectory: "products")
        return parse(url.flatMap { try? Data(contentsOf: $0) }.flatMap(JSON.object)?["confidence"] as? [String: Any])
    }
}
