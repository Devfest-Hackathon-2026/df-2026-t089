import Foundation

/**
 * いくつも見えている商品のうち、1 つだけ特にスマホに近い商品を選ぶ。決め手はスマホから商品までの距離（深度）だけで、
 * 枠の大きさは使わない（枠は、どこに商品があるかと、その場所の距離を測るためだけに使う）。
 *
 * 1. 距離の分かる枠のうち、いちばん近いものが maxDistanceM 以内
 * 2. それと同じくらいの距離（差が sameTolM 以内）の枠は、重なっていれば同じ商品（近づくと商品の中の絵や文字も枠になる）、
 *    重なっていなければ、同じくらい近い別の商品があるので選ばない
 * 3. それ以外の枠は、どれも minGapM 以上遠い
 * 他に距離の分かる枠がなければ、1 だけで選ぶ。
 */
enum NearProduct {
    /** 選んだ商品。box は同じ商品の枠を合わせた範囲（正立画像の 0..1）、gapM は次に近い商品との距離の差（他になければ nil）。 */
    struct Pick {
        let box: [Float]
        let distanceM: Float
        let gapM: Float?

        func describe() -> String {
            "近い商品 \(String(format: "%.2fm", distanceM))（次の商品と \(gapM.map { String(format: "%.2fm", $0) } ?? "-") 差）"
        }
    }

    static let maxDistanceM: Float = 0.8
    static let minGapM: Float = 0.15
    static let sameTolM: Float = 0.08

    /// boxes と distances（同じ順、分からなければ nil）から選ぶ。なければ nil。
    static func pick(_ boxes: [[Float]], _ distances: [Float?]) -> Pick? {
        precondition(boxes.count == distances.count)
        let known = boxes.indices.filter { distances[$0] != nil }.stableSorted { distances[$0]! < distances[$1]! }
        guard let first = known.first else { return nil }
        let d = distances[first]!
        if d > maxDistanceM { return nil }
        var box = boxes[first]
        var gap: Float?
        for i in known.dropFirst() {
            let di = distances[i]!
            if di - d <= sameTolM {
                if !overlaps(box, boxes[i]) { return nil }   // 同じくらい近い別の商品がある
                box = union(box, boxes[i])
            } else {
                gap = di - d
                break   // 近い順に見ているので、残りはもっと遠い
            }
        }
        if let gap, gap < minGapM { return nil }
        return Pick(box: box, distanceM: d, gapM: gap)
    }

    /// 枠ごとの距離を distanceOf で測り、pick する。距離の一覧も返す（画面の表示用）。
    static func measure(_ boxes: [[Float]], _ distanceOf: ([Float]) -> Float?) -> (Pick?, [Float?]) {
        let distances = boxes.map(distanceOf)
        return (pick(boxes, distances), distances)
    }

    private static func overlaps(_ a: [Float], _ b: [Float]) -> Bool { a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3] }

    private static func union(_ a: [Float], _ b: [Float]) -> [Float] {
        [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    }
}
