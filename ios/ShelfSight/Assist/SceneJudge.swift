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
    /// 1 つだけ特に近い商品がある（NearProduct、スマホからの距離で決める）。
    var near = false
}

/**
 * 写っているものを決める。毎回の結果はぶれるので、同じ判定が STABLE 回続いたときだけ切り替える。
 * - 1 つだけ特に近い商品がある（NearProduct の距離の判定。枠の大きさでは決めない）→ oneProduct
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
    /// 棚の説明で、大きすぎる枠（手前の商品や棚の区画）があれば棚とはみなさない。画面に占める割合。
    static let largeBoxArea: Float = 0.12
    static let minProductsOnShelf = 4

    static func area(_ b: [Float]) -> Float { max(b[2] - b[0], 0) * max(b[3] - b[1], 0) }

    /// 長押しで棚を説明するのに必要な商品の数と、商品が広がっている範囲（画面に対する幅・高さ）。
    static let shelfMinProducts = 6
    static let shelfMinWidth: Float = 0.5
    static let shelfMinHeight: Float = 0.35

    /**
     * 長押しのとき、何を調べるか。商品 1 つなら調べる範囲（正立画像の 0..1）も返す。
     * - 1 つだけ特に近い商品がある（nearBox は NearProduct で選んだ商品の範囲）→ product
     * - 棚の広い範囲に小さな商品が並んでいる → shelf
     * - それ以外（商品が少ない、大きめの物が写っているが近づいてはいない）→ stepBack
     */
    static func focus(_ boxes: [[Float]], nearBox: [Float]? = nil) -> (Focus, [Float]?) {
        if let nearBox { return (.product, nearBox) }
        if boxes.isEmpty { return (.nothing, nil) }
        let union: [Float] = [
            boxes.map { $0[0] }.min()!, boxes.map { $0[1] }.min()!,
            boxes.map { $0[2] }.max()!, boxes.map { $0[3] }.max()!,
        ]
        let spread = boxes.count >= shelfMinProducts && union[2] - union[0] >= shelfMinWidth && union[3] - union[1] >= shelfMinHeight
        let noneLarge = !boxes.contains { area($0) >= largeBoxArea }
        return (spread && noneLarge ? .shelf : .stepBack, nil)
    }

    static func classify(_ e: SceneEvidence) -> Scene {
        if e.near { return .oneProduct }
        if e.shelves >= 2 { return .shelves }
        if e.productBoxes.count >= minProductsOnShelf { return .oneShelf }
        if e.shelves == 1 { return .oneShelf }
        return .none
    }
}
