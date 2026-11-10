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
    /// スマホのすぐ前に商品がある（NearProduct、画面の中央の深度だけで決める）。
    var near = false
}

/**
 * 写っているものを決める。毎回の結果はぶれるので、同じ判定が STABLE 回続いたときだけ切り替える。
 * - スマホのすぐ前に商品がある（NearProduct、画面の中央の深度だけで決める）→ oneProduct
 * - 棚が 2 つ以上 → shelves
 * - 棚が 1 つで、商品がいくつも見える → oneShelf
 */
final class SceneJudge {
    private(set) var scene = Scene.none
    private var candidate = Scene.none
    private var count = 0

    /// 判定を足し、切り替わったら true。
    @discardableResult
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
    static let minProductsOnShelf = 4

    static func area(_ b: [Float]) -> Float { max(b[2] - b[0], 0) * max(b[3] - b[1], 0) }

    /// 長押しで棚を説明するのに必要な商品の数と、商品が広がっている範囲（画面に対する幅・高さ）。
    static let shelfMinProducts = 6
    static let shelfMinWidth: Float = 0.5
    static let shelfMinHeight: Float = 0.35

    /**
     * 長押しのとき、何を調べるか。商品 1 つなら調べる範囲（正立画像の 0..1）も返す。
     * - スマホのすぐ前に商品がある（nearBox は NearProduct で決めた調べる範囲）→ product
     * - 棚の広い範囲に小さな商品が並んでいる → shelf
     * - それ以外（商品が少ない、広がっていない）→ stepBack。枠の大きさでは決めない
     */
    static func focus(_ boxes: [[Float]], nearBox: [Float]? = nil) -> (Focus, [Float]?) {
        if let nearBox { return (.product, nearBox) }
        if boxes.isEmpty { return (.nothing, nil) }
        let union: [Float] = [
            boxes.map { $0[0] }.min()!, boxes.map { $0[1] }.min()!,
            boxes.map { $0[2] }.max()!, boxes.map { $0[3] }.max()!,
        ]
        let spread = boxes.count >= shelfMinProducts && union[2] - union[0] >= shelfMinWidth && union[3] - union[1] >= shelfMinHeight
        return (spread ? .shelf : .stepBack, nil)
    }

    private static let centerRadius: Float = 0.15
    private static let oneProductArea: Float = 0.12
    private static let oneProductRatio: Float = 2.5

    /// 画面の中央の商品の枠。中央の点を含む枠のうち中心がいちばん中央に近いもの。含む枠がなければ、中心が centerRadius 以内でいちばん近いもの。
    static func centerProduct(_ boxes: [[Float]]) -> [Float]? {
        func dist(_ b: [Float]) -> Float { hypot((b[0] + b[2]) / 2 - 0.5, (b[1] + b[3]) / 2 - 0.5) }
        let containing = boxes.filter { $0[0] <= 0.5 && $0[2] >= 0.5 && $0[1] <= 0.5 && $0[3] >= 0.5 }
        return (containing.isEmpty ? boxes.filter { dist($0) <= centerRadius } : containing).min { dist($0) < dist($1) }
    }

    /// 2 回たたいたときに LiDAR で近い商品を測れなければ、中央の商品を調べる。枠がなくても中央を文字読みに渡す。
    static func productTapBox(_ boxes: [[Float]]) -> [Float] { centerProduct(boxes) ?? [0.2, 0.2, 0.8, 0.8] }

    /// 画面いっぱいに大きく 1 つだけ写っている商品の枠。
    static func mainProduct(_ boxes: [[Float]]) -> [Float]? {
        let sorted = boxes.stableSorted { area($0) > area($1) }
        guard let first = sorted.first else { return nil }
        let second = sorted.count > 1 ? area(sorted[1]) : 0
        let centered = first[0] < 0.75 && first[2] > 0.25 && first[1] < 0.75 && first[3] > 0.25
        return area(first) >= oneProductArea && area(first) >= second * oneProductRatio && centered ? first : nil
    }

    static func classify(_ e: SceneEvidence) -> Scene {
        if e.near { return .oneProduct }
        if e.shelves >= 2 { return .shelves }
        if e.productBoxes.count >= minProductsOnShelf { return .oneShelf }
        if e.shelves == 1 { return .oneShelf }
        return .none
    }
}
