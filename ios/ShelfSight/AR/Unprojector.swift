import Foundation

/// 深度画像のコピー。値はメートルで、0 や NaN は無効（ARKit の sceneDepth）。
final class DepthSnapshot {
    let width: Int
    let height: Int
    let meters: [Float]

    init(width: Int, height: Int, meters: [Float]) {
        self.width = width
        self.height = height
        self.meters = meters
    }

    func at(_ x: Int, _ y: Int) -> Float { meters[y * width + x] }
}

/**
 * 撮影した瞬間のカメラの状態。解析には時間がかかり、その間にカメラが動くので、撮影時の値で 3D の位置へ戻す。
 * 座標の決まりは Android 版（ARCore）と同じにしてある。
 */
struct CaptureGeometry {
    /// センサー画像（横向き、縮めた後）の大きさ。
    let imageWidth: Int
    let imageHeight: Int
    let focalX: Float
    let focalY: Float
    let principalX: Float
    let principalY: Float
    /// 列優先の 4x4 行列。カメラ座標は +X が画像の右、+Y が画像の上、-Z が前方（ARKit の camera.transform と同じ）。
    let cameraToWorld: [Float]
    /// センサー画像を正立させるために時計回りに回す角度（0, 90, 180, 270）。
    let uprightRotationDegrees: Int
    let depth: DepthSnapshot?
    /// センサー画像のピクセル座標から深度画像の正規化座標への変換。
    let depthU0: Float
    let depthUPerPixel: Float
    let depthV0: Float
    let depthVPerPixel: Float
}

enum Unprojector {
    private static let minDepthM: Float = 0.2
    private static let maxDepthM: Float = 8.0
    private static let sampleGrid = 9
    private static let sampledBoxFraction: Float = 0.4

    /// 正立画像の正規化座標を、回転前のセンサー画像の正規化座標へ戻す。
    static func uprightToSensor(_ u: Float, _ v: Float, _ rotationDegrees: Int) -> (Float, Float) {
        switch rotationDegrees {
        case 90: return (v, 1 - u)
        case 180: return (1 - u, 1 - v)
        case 270: return (1 - v, u)
        default: return (u, v)
        }
    }

    /// 正立画像の座標 (u, v) にある、カメラから depth メートルの点のワールド座標。
    static func worldAt(_ u: Float, _ v: Float, _ depth: Float, _ g: CaptureGeometry) -> Vec3 {
        let (sx, sy) = uprightToSensor(u, v, g.uprightRotationDegrees)
        return pixelToWorld(sx * Float(g.imageWidth), sy * Float(g.imageHeight), depth, g)
    }

    /// 正立画像の座標 (u, v) の深度。無効または範囲外なら nil。
    static func depthAt(_ u: Float, _ v: Float, _ g: CaptureGeometry) -> Float? {
        guard let depth = g.depth else { return nil }
        let (sx, sy) = uprightToSensor(u, v, g.uprightRotationDegrees)
        let du = g.depthU0 + sx * Float(g.imageWidth) * g.depthUPerPixel
        let dv = g.depthV0 + sy * Float(g.imageHeight) * g.depthVPerPixel
        let fx = du * Float(depth.width)
        let fy = dv * Float(depth.height)
        guard fx.isFinite, fy.isFinite else { return nil }
        let dx = Int(fx)
        let dy = Int(fy)
        guard dx >= 0, dx < depth.width, dy >= 0, dy < depth.height else { return nil }
        let m = depth.at(dx, dy)
        return (m.isFinite && m >= minDepthM && m <= maxDepthM) ? m : nil
    }

    /// 矩形の端は棚の背景を拾いやすいので、中央部分だけを見て中央値を取る。
    static func medianDepthMeters(_ box: NormalizedBox, _ g: CaptureGeometry) -> Float? {
        if g.depth == nil { return nil }
        let cu = (box.left + box.right) / 2
        let cv = (box.top + box.bottom) / 2
        let hw = (box.right - box.left) * sampledBoxFraction / 2
        let hh = (box.bottom - box.top) * sampledBoxFraction / 2
        var samples: [Float] = []
        samples.reserveCapacity(sampleGrid * sampleGrid)
        for row in 0..<sampleGrid {
            for col in 0..<sampleGrid {
                let u = cu - hw + 2 * hw * Float(col) / Float(sampleGrid - 1)
                let v = cv - hh + 2 * hh * Float(row) / Float(sampleGrid - 1)
                if let d = depthAt(u, v, g) { samples.append(d) }
            }
        }
        if samples.isEmpty { return nil }
        samples.sort()
        return samples[samples.count / 2]
    }

    private static func pixelToWorld(_ px: Float, _ py: Float, _ depth: Float, _ g: CaptureGeometry) -> Vec3 {
        let x = (px - g.principalX) / g.focalX * depth
        let y = -(py - g.principalY) / g.focalY * depth
        let z = -depth
        let m = g.cameraToWorld
        return Vec3(
            m[0] * x + m[4] * y + m[8] * z + m[12],
            m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14]
        )
    }
}
