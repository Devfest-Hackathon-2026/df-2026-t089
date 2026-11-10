import Foundation

/**
 * RANSAC の前に、ガラス越しに奥行きを誤った点を直す。同じ視点から同じ方位を見ているのに奥行きが大きくばらつくなら、
 * そこはガラス面なので、各点を視線に沿ってばらつきの手前の端へ戻す。斜めに見た棚は方位につれて奥行きが素直に変わるだけなので、
 * 方位に対する傾きを除いた残差でばらつきを測る。
 */
enum MirrorCorrection {
    private static let viewpointM = 0.35
    private static let sectorRad = 10.0 * Double.pi / 180
    private static let minRays = 6
    private static let stdLimitM = 0.7
    private static let nearQuantile = 0.15
    private static let minDepthM = 0.2

    /// points は補正後の全ての点 (x, z)、moved は動かした点の (元 x, 元 z, 新 x, 新 z)。
    struct Result {
        let points: [[Double]]
        let moved: [[Double]]
    }

    private struct Ray {
        let index: Int
        let bearing: Double
        let depth: Double
    }

    /// rays は観測ごとの (カメラ x, カメラ z, 点 x, 点 z)。
    static func correct(_ rays: [[Double]]) -> Result {
        var points = rays.map { [$0[2], $0[3]] }
        var moved: [[Double]] = []
        let viewpoints = Array(rays.indices).orderedGroups { IntPair(Int(floor(rays[$0][0] / viewpointM)), Int(floor(rays[$0][1] / viewpointM))) }
        for viewpoint in viewpoints {
            let sectors = viewpoint.values.map { i -> Ray in
                let r = rays[i]
                return Ray(index: i, bearing: atan2(r[3] - r[1], r[2] - r[0]), depth: hypot(r[2] - r[0], r[3] - r[1]))
            }.orderedGroups { Int(floor($0.bearing / sectorRad)) }
            for group in sectors {
                let sector = group.values
                if sector.count < minRays { continue }
                let count = Double(sector.count)
                let meanBearing = sector.reduce(0) { $0 + $1.bearing } / count
                let meanDepth = sector.reduce(0) { $0 + $1.depth } / count
                let spreadB = sector.reduce(0) { $0 + ($1.bearing - meanBearing) * ($1.bearing - meanBearing) }
                let slope = spreadB > 1e-9 ? sector.reduce(0) { $0 + ($1.bearing - meanBearing) * ($1.depth - meanDepth) } / spreadB : 0
                func trend(_ r: Ray) -> Double { meanDepth + slope * (r.bearing - meanBearing) }
                let residuals = sector.map { $0.depth - trend($0) }.sorted()
                if (residuals.reduce(0) { $0 + $1 * $1 } / count).squareRoot() < stdLimitM { continue }
                let near = residuals[min(residuals.count - 1, Int(nearQuantile * count))]
                for r in sector {
                    let depth = trend(r) + near
                    if r.depth - trend(r) <= near || depth <= minDepthM { continue }
                    let cam = rays[r.index]
                    let x = cam[0] + depth * cos(r.bearing)
                    let z = cam[1] + depth * sin(r.bearing)
                    moved.append([cam[2], cam[3], x, z])
                    points[r.index] = [x, z]
                }
            }
        }
        return Result(points: points, moved: moved)
    }

    /// RANSAC の計算量を抑えるため、10 cm の格子ごとの平均にする。
    static func cellMeans(_ points: [[Double]]) -> [[Double]] {
        points.orderedGroups { IntPair(Int(floor($0[0] / 0.1)), Int(floor($0[1] / 0.1))) }.map { g in
            let n = Double(g.values.count)
            return [g.values.reduce(0) { $0 + $1[0] } / n, g.values.reduce(0) { $0 + $1[1] } / n]
        }
    }
}
