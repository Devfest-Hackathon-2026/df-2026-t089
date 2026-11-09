import Foundation

/**
 * 周りの商品のカテゴリで判定を補正する（products/context.py の移植）。同じカテゴリの商品は棚で固まっているので、
 * 周りと違うカテゴリの候補にペナルティをかける。パン棚のブラックサンダーのような、似ているだけの候補が選ばれにくくなる。
 */
enum ContextAdjust {
    private static let nb = 8          // 周りとみなす近い商品の数
    private static let rowWeight = 3.0 // 縦方向の距離の重み（別の段の商品は遠い扱い）
    private static let temp = 20.0     // 類似度 → 確率の鋭さ
    private static let eps = 0.02      // 周りに一度も出ないカテゴリでも確率 0 にしない

    /// 候補（名前のキーと類似度、高い順）。
    struct Candidate {
        let key: String
        let sim: Float
    }

    /// 選んだ候補。key が nil なら候補なし。
    struct Pick {
        let key: String?
        let score: Double
        let sim: Float
    }

    /// 1 段の補正。categoryOf がキーをカテゴリに変える（不明なら nil）。
    struct Level {
        let categoryOf: (String) -> String?
        let lambda: Double
    }

    /// centers は画像内の位置（0..1 の x, y）。
    static func adjust(_ centers: [(Float, Float)], _ cands: [[Candidate]], _ levels: [Level]) -> [Pick] {
        let n = cands.count
        let first = cands.map { c in c.first.map { Pick(key: $0.key, score: Double($0.sim), sim: $0.sim) } ?? Pick(key: nil, score: 0, sim: 0) }
        if n == 0 || !levels.contains(where: { $0.lambda != 0 }) { return first }

        // 候補ごとの確率（類似度の softmax）
        let w: [[Double]] = cands.map { c in
            if c.isEmpty { return [] }
            let top = Double(c.map(\.sim).max()!)
            let e = c.map { exp(temp * (Double($0.sim) - top)) }
            let s = max(e.reduce(0, +), 1e-12)
            return e.map { $0 / s }
        }
        let near = neighbors(centers)
        var score: [[Double]] = cands.map { c in c.map { Double($0.sim) } }
        for level in levels where level.lambda != 0 {
            var ids: [String: Int] = [:]
            let cat: [[Int]] = cands.map { c in
                c.map { cand in
                    guard let g = level.categoryOf(cand.key) else { return -1 }
                    if let id = ids[g] { return id }
                    ids[g] = ids.count
                    return ids.count - 1
                }
            }
            if ids.isEmpty { continue }
            let m = ids.count
            var p = [[Double]](repeating: [Double](repeating: 0, count: m), count: n)
            for i in 0..<n { for k in cat[i].indices where cat[i][k] >= 0 { p[i][cat[i][k]] += w[i][k] } }
            for i in 0..<n {
                var q = [Double](repeating: 0, count: m)
                for (j, wt) in near[i] { for c in 0..<m { q[c] += wt * p[j][c] } }
                let total = q.reduce(0, +) + eps * Double(m)
                for k in cat[i].indices where cat[i][k] >= 0 {
                    score[i][k] += level.lambda * log((q[cat[i][k]] + eps) / total)
                }
            }
        }
        return cands.enumerated().map { i, c in
            if c.isEmpty { return Pick(key: nil, score: 0, sim: 0) }
            let b = score[i].indices.max { score[i][$0] < score[i][$1] }!
            return Pick(key: c[b].key, score: score[i][b], sim: c[b].sim)
        }
    }

    /// 各商品の近い nb 個と重み（1/(距離+0.02) を合計 1 に）。
    private static func neighbors(_ centers: [(Float, Float)]) -> [[(Int, Double)]] {
        let n = centers.count
        if n < 2 { return [[(Int, Double)]](repeating: [], count: n) }
        let k = min(nb, n - 1)
        return (0..<n).map { i in
            let near = (0..<n).filter { $0 != i }.map { j -> (Int, Double) in
                let dx = Double(centers[i].0 - centers[j].0)
                let dy = Double(centers[i].1 - centers[j].1) * rowWeight
                return (j, (dx * dx + dy * dy).squareRoot())
            }.stableSorted { $0.1 < $1.1 }.prefix(k)
            let ws = near.map { ($0.0, 1.0 / ($0.1 + 0.02)) }
            let s = ws.reduce(0) { $0 + $1.1 }
            return ws.map { ($0.0, $0.1 / s) }
        }
    }
}
