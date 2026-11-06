import Foundation

/// カメラに何が主に写っているか。
enum Scene: String {
    case none = "NONE", shelves = "SHELVES", oneShelf = "ONE_SHELF", oneProduct = "ONE_PRODUCT"
}

/// 長押しで調べるもの。
enum Focus: String {
    case product = "PRODUCT", shelf = "SHELF", stepBack = "STEP_BACK", nothing = "NOTHING"
}

/// 1 回の解析で分かったこと。
struct SceneEvidence {
    /// 棚の境界で区切った区間のうち、分類のどれかになったもの。
    let shelves: Int
    /// 商品の枠（正立画像の 0..1、[left, top, right, bottom]）。
    let productBoxes: [[Float]]
}

/**
 * 写っているものを決める。毎回の結果はぶれるので、同じ判定が STABLE 回続いたときだけ切り替える。
 * - 商品が 1 つ大きく写っている（近づいている）→ oneProduct
 * - 棚が 2 つ以上 → shelves
 * - 棚が 1 つで、商品がいくつも見える → oneShelf
 */
final class SceneJudge {
    private(set) var scene = Scene.none
    private var candidate = Scene.none
    private var count = 0

    /// 判定を足し、切り替わったら true。
    func update(_ e: SceneEvidence) -> Bool {
        let now = Self.classify(e)
        if now == candidate { count += 1 } else { candidate = now; count = 1 }
        if candidate != scene && count >= Self.stable {
            scene = candidate
            return true
        }
        return false
    }

    static let stable = 2
    /// 画面に占める割合がこれ以上の商品は「1 つ大きく写っている」。
    static let oneProductArea: Float = 0.12
    /// 1 つ大きく写っている商品は、2 番目の商品よりこれだけ大きい。
    static let oneProductRatio: Float = 2.5
    static let minProductsOnShelf = 4

    static func area(_ b: [Float]) -> Float { max(b[2] - b[0], 0) * max(b[3] - b[1], 0) }

    /// 画面の真ん中にかかる、大きく写った 1 つの商品。なければ nil。
    static func mainProduct(_ boxes: [[Float]]) -> [Float]? {
        let sorted = boxes.stableSorted { area($0) > area($1) }
        guard let first = sorted.first else { return nil }
        let second = sorted.count > 1 ? area(sorted[1]) : 0
        let centered = first[0] < 0.75 && first[2] > 0.25 && first[1] < 0.75 && first[3] > 0.25
        return (area(first) >= oneProductArea && area(first) >= second * oneProductRatio && centered) ? first : nil
    }

    /// 長押しで棚を説明するのに必要な商品の数と、商品が広がっている範囲（画面に対する幅・高さ）。
    static let shelfMinProducts = 6
    static let shelfMinWidth: Float = 0.5
    static let shelfMinHeight: Float = 0.35
    /// 画面のこれ以上を占める枠は、他の枠があっても手前の商品 1 つ（近づくと商品の中の文字や絵も枠になる）。
    static let closeProductArea: Float = 0.25
    /// 枠が少なく、合わせて画面のこれ以上を占めるときも、商品 1 つを近くで写している。
    static let closeUnionArea: Float = 0.6
    static let closeMaxBoxes = 4

    /**
     * 長押しのとき、何を調べるか。商品 1 つなら調べる範囲（正立画像の 0..1）も返す。
     * - 商品 1 つが大きく写っている、または画面いっぱいに写っている → product
     * - 棚の広い範囲に小さな商品が並んでいる → shelf
     * - その間（中くらいの商品がいくつか、または商品が少ない） → stepBack
     */
    static func focus(_ boxes: [[Float]]) -> (Focus, [Float]?) {
        if boxes.isEmpty { return (.nothing, nil) }
        if let main = mainProduct(boxes) { return (.product, main) }
        let largest = boxes.max { area($0) < area($1) }!
        let union: [Float] = [
            boxes.map { $0[0] }.min()!, boxes.map { $0[1] }.min()!,
            boxes.map { $0[2] }.max()!, boxes.map { $0[3] }.max()!,
        ]
        // 調べる範囲は、枠が少なければ全部を合わせた範囲（商品の部分ごとに枠が出ることがある）
        if area(largest) >= closeProductArea { return (.product, boxes.count <= closeMaxBoxes ? union : largest) }
        if boxes.count <= closeMaxBoxes && area(union) >= closeUnionArea && area(largest) >= oneProductArea / 2 {
            return (.product, union)
        }
        let spread = boxes.count >= shelfMinProducts && union[2] - union[0] >= shelfMinWidth && union[3] - union[1] >= shelfMinHeight
        let noneLarge = !boxes.contains { area($0) >= oneProductArea }
        return (spread && noneLarge ? .shelf : .stepBack, nil)
    }

    static func classify(_ e: SceneEvidence) -> Scene {
        if mainProduct(e.productBoxes) != nil { return .oneProduct }
        if e.shelves >= 2 { return .shelves }
        if e.productBoxes.count >= minProductsOnShelf { return .oneShelf }
        if e.shelves == 1 { return .oneShelf }
        return .none
    }
}
