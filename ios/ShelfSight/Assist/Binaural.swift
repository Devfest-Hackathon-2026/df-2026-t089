import Foundation

/// 聞く人の位置と、水平面での正面の向き（単位ベクトル x, z）。
struct Listener {
    let position: Vec3
    let forwardX: Float
    let forwardZ: Float
}

/// 片耳ずつの鳴らし方。
struct EarParams {
    let gainL: Float
    let gainR: Float
    /// 遅らせる量（秒）。
    let delayL: Float
    let delayR: Float
    /// 1 次ローパスの係数（1 で素通し）。頭の陰になる耳ほど高音を落とす。
    let lowpassL: Float
    let lowpassR: Float
}

/**
 * イヤホン向けの簡単な立体音響。左右の耳に届く時間の差（ITD）と大きさ・音色の差（ILD、頭の陰）、距離による減衰で、
 * 音の方向を作る。後ろの音は少しこもらせて、前と区別しやすくする。
 */
enum Binaural {
    private static let headRadiusM: Float = 0.0875
    private static let speedOfSound: Float = 343
    private static let shadowHz: Float = 1200
    private static let behindHz: Float = 5000

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

    static func params(_ az: Float, _ distance: Float, _ sampleRate: Int) -> EarParams {
        let s = sin(az)
        // 横から来る角度（前後どちらでも同じ）。Woodworth の式で時間差を出す
        let lateral = abs(atan2(s, abs(cos(az))))
        let itd = headRadiusM / speedOfSound * (lateral + sin(lateral))
        let g = distanceGain(distance)
        // 等パワーのパン。正面で左右 0.71、真横で 1 と 0.2 くらい
        let gl = (0.5 * (1 - 0.92 * s)).squareRoot() * g
        let gr = (0.5 * (1 + 0.92 * s)).squareRoot() * g
        let behind = cos(az) < 0
        func shadow(_ amount: Float) -> Float { Binaural.lowpassCoeff(20000 - (20000 - Binaural.shadowHz) * amount, sampleRate) }
        let back: Float = behind ? lowpassCoeff(behindHz, sampleRate) : 1
        return EarParams(
            gainL: gl, gainR: gr,
            delayL: s > 0 ? itd : 0, delayR: s < 0 ? itd : 0,
            lowpassL: min(back, shadow(max(0, s))), lowpassR: min(back, shadow(max(0, -s)))
        )
    }

    /// 1 次ローパス y += a(x - y) の a。
    static func lowpassCoeff(_ cutoffHz: Float, _ sampleRate: Int) -> Float {
        (1 - exp(-2 * Float.pi * cutoffHz / Float(sampleRate))).clamped(0, 1)
    }

    /// 時計の文字盤の向き（12 時が正面）。
    static func clock(_ az: Float) -> Int {
        let h = ((Int((az / (2 * Float.pi) * 12).rounded()) % 12) + 12) % 12
        return h == 0 ? 12 : h
    }
}
