import Foundation

/// 測定した頭部インパルス応答（SADIE II の KU100）。方向ごとに左右の耳の FIR を持つ。tools/export_hrtf.py で作る。
final class Hrtf {
    static let taps = 256

    private let count: Int
    private var dirX: [Float]
    private var dirY: [Float]
    private var dirZ: [Float]
    private var ir: [Float]

    init?(_ data: Data) {
        let raw = [UInt8](data)
        func int32(_ at: Int) -> Int { raw.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: at, as: Int32.self).littleEndian) } }
        guard raw.count >= 8, int32(4) == Hrtf.taps else { return nil }
        count = int32(0)
        guard count > 0, raw.count == 8 + count * 8 + count * 2 * Hrtf.taps * 2 else { return nil }
        dirX = [Float](repeating: 0, count: count)
        dirY = dirX
        dirZ = dirX
        ir = [Float](repeating: 0, count: count * 2 * Hrtf.taps)
        raw.withUnsafeBytes { b in
            for i in 0..<count {
                let az = b.loadUnaligned(fromByteOffset: 8 + i * 8, as: Float.self) * .pi / 180
                let el = b.loadUnaligned(fromByteOffset: 12 + i * 8, as: Float.self) * .pi / 180
                // ファイルの方位は左回りなので、右が正の向きに直す
                dirX[i] = -cos(el) * sin(az)
                dirY[i] = sin(el)
                dirZ[i] = cos(el) * cos(az)
            }
            let start = 8 + count * 8
            for i in 0..<ir.count {
                ir[i] = Float(b.loadUnaligned(fromByteOffset: start + i * 2, as: Int16.self).littleEndian) / 32767
            }
        }
    }

    /// いちばん近い測定方向。azimuth は右が正、elevation は上が正（ラジアン）。
    func nearest(azimuth: Float, elevation: Float) -> Int {
        let x = cos(elevation) * sin(azimuth), y = sin(elevation), z = cos(elevation) * cos(azimuth)
        var best = 0
        var bestDot: Float = -2
        for i in 0..<count {
            let d = dirX[i] * x + dirY[i] * y + dirZ[i] * z
            if d > bestDot { bestDot = d; best = i }
        }
        return best
    }

    /// pcm の frame 番目の出力。ear は 0 が左、1 が右。音の終わりの余韻のため、frame は pcm を taps 超えてよい。
    func convolve(_ pcm: UnsafeBufferPointer<Float>, frame: Int, direction: Int, ear: Int) -> Float {
        let start = (direction * 2 + ear) * Hrtf.taps
        let first = max(0, frame - pcm.count + 1)
        let last = min(Hrtf.taps - 1, frame)
        if first > last { return 0 }
        return ir.withUnsafeBufferPointer { h in
            var sum: Float = 0
            for i in first...last { sum += h[start + i] * pcm[frame - i] }
            return sum
        }
    }

    static func load() -> Hrtf? {
        guard let url = Bundle.main.url(forResource: "sadie_d1_ku100", withExtension: "bin", subdirectory: "hrtf"),
              let data = try? Data(contentsOf: url), let hrtf = Hrtf(data) else {
            logInfo("Hrtf", "HRTF を読めないので、音は頭の中（正面）で鳴らす")
            return nil
        }
        return hrtf
    }
}
