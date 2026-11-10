import Foundation

/// 聞く人の位置と、水平面での正面の向き（単位ベクトル x, z）。
struct Listener {
    let position: Vec3
    let forwardX: Float
    let forwardZ: Float
}

/// 聞く人から見た音の方向と距離。耳へ届く音そのものは Hrtf で作る。
enum Binaural {
    /// 右が正の方位角（ラジアン、-π..π）。0 が正面。
    static func azimuth(_ l: Listener, _ source: Vec3) -> Float {
        let dx = source.x - l.position.x
        let dz = source.z - l.position.z
        // 右向きのベクトルは正面を時計回りに 90° 回したもの（y が上、正面が -z のとき右は +x）
        let rightX = -l.forwardZ
        let rightZ = l.forwardX
        return atan2(dx * rightX + dz * rightZ, dx * l.forwardX + dz * l.forwardZ)
    }

    static func horizontalDistance(_ l: Listener, _ source: Vec3) -> Float {
        let dx = source.x - l.position.x
        let dz = source.z - l.position.z
        return (dx * dx + dz * dz).squareRoot()
    }

    /// 距離による音量。1m までは 1、それより遠いと距離に反比例（遠くても小さくなりすぎない）。
    static func distanceGain(_ d: Float) -> Float { max(1 / max(d, 1), 0.25) }
}
