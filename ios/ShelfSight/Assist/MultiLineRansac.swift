import Foundation

/**
 * 局所 RANSAC。各点の周囲だけを見て 1 m 未満の短い線分を当て、内点を取り除いて次の点へ進む。
 * 店の棚は互いに平行か垂直なので、線の向きは店の 2 つの軸だけを試す。
 */
enum MultiLineRansac {
    static let maxLines = 256
    private static let iterations = 30
    private static let radiusM = 0.5   // 直径が線分の上限の 1 m になる
    private static let inlierM = 0.1
    private static let minInliers = 6
    private static let minInlierShare = 0.35   // 周囲の点の大半が線に乗らないなら、ただの散らばり
    private static let gapM = 0.3
    private static let minLengthM = 0.25
    private static let maxLengthM = 1.0

    private struct Lcg: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (state ^ (state >> 22)) &* 0x9E37_79B9_7F4A_7C15
        }
    }

    /// points は (x, z)、yaw は世界 → 店の座標の回転。戻り値は線分 (x0, z0, x1, z1)。線は店の u 軸か v 軸に平行になる。
    static func fit(_ points: [[Double]], yaw: Double, seed: UInt64 = 1) -> [[Double]] {
        var rng = Lcg(state: seed)
        let axes = [(cos(yaw), sin(yaw)), (-sin(yaw), cos(yaw))]
        var out: [[Double]] = []
        var used = [Bool](repeating: false, count: points.count)
        for c in Array(points.indices).shuffled(using: &rng) {
            if out.count == maxLines { break }
            if used[c] { continue }
            let center = points[c]
            let local = points.indices.filter { !used[$0] && hypot(points[$0][0] - center[0], points[$0][1] - center[1]) <= radiusM }
            if local.count < minInliers { continue }
            var best: [Int] = []
            var bestAxis = axes[0]
            for _ in 0..<iterations {
                let a = points[local[Int.random(in: 0..<local.count, using: &rng)]]
                for axis in axes {
                    let inliers = local.filter { abs((points[$0][0] - a[0]) * -axis.1 + (points[$0][1] - a[1]) * axis.0) <= inlierM }
                    let run = longestRun(inliers, points, a, axis.0, axis.1)
                    if run.count > best.count { best = run; bestAxis = axis }
                }
            }
            if best.count < minInliers || Double(best.count) < Double(local.count) * minInlierShare { continue }
            if let s = segment(best.map { points[$0] }, bestAxis.0, bestAxis.1) { out.append(s) }
            for i in best { used[i] = true }
        }
        return out
    }

    /// 線に沿って途切れない、いちばん点の多い区間。
    private static func longestRun(_ inliers: [Int], _ points: [[Double]], _ origin: [Double], _ dx: Double, _ dz: Double) -> [Int] {
        if inliers.isEmpty { return [] }
        let sorted = inliers.stableSorted {
            (points[$0][0] - origin[0]) * dx + (points[$0][1] - origin[1]) * dz < (points[$1][0] - origin[0]) * dx + (points[$1][1] - origin[1]) * dz
        }
        var best: ArraySlice<Int> = []
        var start = 0
        for i in 1...sorted.count {
            let gap = i == sorted.count ||
                (points[sorted[i]][0] - points[sorted[i - 1]][0]) * dx + (points[sorted[i]][1] - points[sorted[i - 1]][1]) * dz > gapM
            if gap {
                if i - start > best.count { best = sorted[start..<i] }
                start = i
            }
        }
        return Array(best)
    }

    /// 内点の重心を通り軸に平行な線分にする。長さが範囲の外なら nil。
    private static func segment(_ points: [[Double]], _ dx: Double, _ dz: Double) -> [Double]? {
        let n = Double(points.count)
        let mx = points.reduce(0) { $0 + $1[0] } / n
        let mz = points.reduce(0) { $0 + $1[1] } / n
        let t = points.map { ($0[0] - mx) * dx + ($0[1] - mz) * dz }
        guard let lo = t.min(), let hi = t.max(), hi - lo >= minLengthM, hi - lo < maxLengthM else { return nil }
        return [mx + lo * dx, mz + lo * dz, mx + hi * dx, mz + hi * dz]
    }
}
