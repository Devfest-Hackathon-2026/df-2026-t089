import Foundation

/// 正立画像上の矩形。各値は左上原点で 0.0〜1.0 に正規化されている。
struct NormalizedBox: Equatable {
    var left: Float
    var top: Float
    var right: Float
    var bottom: Float
}

/// ARKit のワールド座標（y が上、メートル）。
struct Vec3: Equatable {
    var x: Float
    var y: Float
    var z: Float

    init(_ x: Float, _ y: Float, _ z: Float) {
        self.x = x
        self.y = y
        self.z = z
    }

    static let zero = Vec3(0, 0, 0)

    /// NaN や無限大を含まない。距離の推定が外れると入ることがあり、そのまま音の計算に渡すと落ちる。
    var isFinite: Bool { x.isFinite && y.isFinite && z.isFinite }

    var length: Float { (x * x + y * y + z * z).squareRoot() }
    var normalized: Vec3 { self * (1 / max(length, 1e-6)) }

    func distanceTo(_ o: Vec3) -> Float { (self - o).length }

    static func + (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x + b.x, a.y + b.y, a.z + b.z) }
    static func - (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x - b.x, a.y - b.y, a.z - b.z) }
    static func * (a: Vec3, k: Float) -> Vec3 { Vec3(a.x * k, a.y * k, a.z * k) }
    static func dot(_ a: Vec3, _ b: Vec3) -> Float { a.x * b.x + a.y * b.y + a.z * b.z }
}
