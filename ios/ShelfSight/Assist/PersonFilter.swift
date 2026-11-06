import Foundation

/**
 * 人として検出された枠のうち、自分の手や腕（商品を持った手など）らしいものを外す。手の検出は使わず、
 * 枠の位置・形と深度だけで決める。胸の高さのカメラで見た人は、近ければ画面の上の方まで写り、
 * 全身が写るなら縦に長いことを使う。
 */
enum PersonFilter {
    /// 全身が写った人の枠の縦横比（ピクセルで、高さ / 幅）の下限。手や持った商品はこれより横に広い。
    static let minStandingAspect: Float = 1.4
    /// これより近い物は、手を伸ばして届く範囲。
    static let nearM: Float = 1.2
    /// 近くの人なら、頭か胴が画面の上の方まで写る。
    static let nearMaxTop: Float = 0.25
    static let nearMinHeight: Float = 0.55
    /// 写った高さから出した距離に比べて、深度がこれより小さく近いなら、人ではなく手前の小さな物。
    static let depthToHeightRatio: Float = 0.5
    private static let edge: Float = 0.02

    /// - Parameters:
    ///   - box: 正立画像の枠
    ///   - aspect: 枠の高さ / 幅（ピクセル）
    ///   - depthM: 枠の中央の深度（なければ nil）
    static func isOwnHand(_ box: NormalizedBox, aspect: Float, depthM: Float?) -> Bool {
        // 画面の下端から立ち上がり、上半分に届かない枠
        if OwnHandFilter.isLowerOnly(box) { return true }
        let height = box.bottom - box.top
        // 手の届く距離にあるのに、画面の上の方まで写っていない
        if let d = depthM, d < nearM, box.top > nearMaxTop || height < nearMinHeight { return true }
        let fullBody = box.top > edge && box.bottom < 1 - edge
        if fullBody {
            // 全身が写っているなら縦に長いはず
            if aspect < minStandingAspect { return true }
            // 写った高さからは遠くの人に見えるのに、深度では手前にある（顔の前にかざした手など）
            if let d = depthM, d < nearM * 2, d < heightDistanceRatio(height) * depthToHeightRatio { return true }
        }
        return false
    }

    /// 枠の高さが画面の height のとき、身長 1.65m の人までの距離のおおよその値（縦の画角 65° くらいのカメラ）。
    private static func heightDistanceRatio(_ height: Float) -> Float { 1.65 * 0.78 / max(height, 0.05) }
}
