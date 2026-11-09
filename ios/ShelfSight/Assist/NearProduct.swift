import Foundation

/**
 * スマホのすぐ前（画面の中央）に商品があるかを、深度だけで決める。商品の枠は使わない
 * （近づくと枠が出なかったり、商品の中の絵や文字ごとにばらばらの枠になったりして、枠では決められないことが多かったため）。
 *
 * 1. 画面の中央（centerFraction 四方）の深度の中央値を、スマホから商品までの距離とする。
 *    中央の点の minValidFraction 以上で深度が取れ、距離が maxDistanceM 以内なら、近い商品がある
 * 2. 調べる範囲（Cloud Vision に送る切り抜き）は、中央から、距離が近い商品と同じくらい（差が sameTolM 以内）の点を
 *    つないだ範囲。ただし棚の面は全部同じくらいの距離なので、実際の大きさで maxSizeM 四方までにする
 */
enum NearProduct {
    /** 選んだ商品。box は調べる範囲（正立画像の 0..1、[left, top, right, bottom]）。 */
    struct Pick {
        let box: [Float]
        let distanceM: Float

        func describe() -> String { "近い商品 \(String(format: "%.2fm", distanceM))" }
    }

    /** 1 回の判定の結果。近い商品がなくても、中央の距離と深度の取れた割合は分かる（画面の表示とログ用）。 */
    struct Reading {
        let pick: Pick?
        /// 画面の中央の距離（深度が足りなければ nil）。
        let centerM: Float?
        /// 画面の中央で深度が取れた点の割合。
        let validFraction: Float

        func describe() -> String {
            if let pick { return pick.describe() }
            let center = centerM.map { String(format: "%.2fm", $0) } ?? "-"
            return "なし（中央 \(center)、深度の取れた点 \(Int(validFraction * 100))%）"
        }
    }

    static let maxDistanceM: Float = 0.5
    static let sameTolM: Float = 0.10
    static let maxSizeM: Float = 0.30
    static let centerFraction: Float = 0.2
    static let minValidFraction: Float = 0.5
    /// 画面全体を grid × grid の点で調べる。
    static let grid = 32

    static func measure(_ g: CaptureGeometry) -> Reading {
        guard g.depth != nil else { return Reading(pick: nil, centerM: nil, validFraction: 0) }
        // 正立画像の上の点ごとの深度（取れなければ nil）
        let n = grid
        var depth = [Float?](repeating: nil, count: n * n)
        for row in 0..<n {
            for col in 0..<n {
                depth[row * n + col] = Unprojector.depthAt((Float(col) + 0.5) / Float(n), (Float(row) + 0.5) / Float(n), g)
            }
        }
        // 1. 画面の中央の距離
        let lo = Int((0.5 - centerFraction / 2) * Float(n)), hi = Int((0.5 + centerFraction / 2) * Float(n))
        var center: [Float] = []
        for row in lo..<hi { for col in lo..<hi { if let d = depth[row * n + col] { center.append(d) } } }
        let valid = Float(center.count) / Float((hi - lo) * (hi - lo))
        if center.isEmpty || valid < minValidFraction { return Reading(pick: nil, centerM: nil, validFraction: valid) }
        center.sort()
        let d = center[center.count / 2]
        if d > maxDistanceM { return Reading(pick: nil, centerM: d, validFraction: valid) }

        // 2. 中央から、同じくらいの距離の点をつなぐ（上下左右の隣へ）
        var seen = [Bool](repeating: false, count: n * n)
        var stack: [Int] = []
        for row in lo..<hi {
            for col in lo..<hi {
                let i = row * n + col
                if let di = depth[i], abs(di - d) <= sameTolM { seen[i] = true; stack.append(i) }
            }
        }
        var minCol = n, minRow = n, maxCol = -1, maxRow = -1
        while let i = stack.popLast() {
            let row = i / n, col = i % n
            minCol = min(minCol, col); maxCol = max(maxCol, col)
            minRow = min(minRow, row); maxRow = max(maxRow, row)
            for (r, c) in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)] where r >= 0 && r < n && c >= 0 && c < n {
                let j = r * n + c
                if !seen[j], let dj = depth[j], abs(dj - d) <= sameTolM { seen[j] = true; stack.append(j) }
            }
        }
        var box = [Float(minCol) / Float(n), Float(minRow) / Float(n), Float(maxCol + 1) / Float(n), Float(maxRow + 1) / Float(n)]

        // 実際の大きさで maxSizeM 四方まで（中央を真ん中にして切る）
        let (hu, hv) = halfSize(maxSizeM / 2, at: d, g)
        box = [max(box[0], 0.5 - hu), max(box[1], 0.5 - hv), min(box[2], 0.5 + hu), min(box[3], 0.5 + hv)]
        return Reading(pick: Pick(box: box, distanceM: d), centerM: d, validFraction: valid)
    }

    /// カメラから distanceM 離れたところの halfM が、正立画像の正規化座標でどれだけの幅・高さになるか。
    static func halfSize(_ halfM: Float, at distanceM: Float, _ g: CaptureGeometry) -> (Float, Float) {
        let rotated = g.uprightRotationDegrees == 90 || g.uprightRotationDegrees == 270
        // 正立画像の横・縦の焦点距離と大きさ（センサー画像を 90 度回すと横と縦が入れ替わる）
        let fu = rotated ? g.focalY : g.focalX, fv = rotated ? g.focalX : g.focalY
        let wu = Float(rotated ? g.imageHeight : g.imageWidth), wv = Float(rotated ? g.imageWidth : g.imageHeight)
        return (halfM / distanceM * fu / wu, halfM / distanceM * fv / wv)
    }
}
