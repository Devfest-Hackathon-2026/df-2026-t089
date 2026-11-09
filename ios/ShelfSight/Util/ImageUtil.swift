import CoreGraphics
import UIKit

/**
 * 正立させたカメラの画像。RGBA のバイト列（A は使わない、上の行から）のまま持つ。
 *
 * 解析（1 秒に 9 回ほど）の縮小・切り抜き・モデルへの入力は、Core Graphics を使わずにこのバイト列から直接作る。
 * Core Graphics で描くと、形式を変えた写しが作られて、すぐには捨てられずに取っておかれ（malloc で 3GB ほどまで溜まり）、
 * 写る場所が変わるたびに増えて、iOS のアプリごとの上限を超えて止められた。CGImage は JPEG にするときだけ作る。
 */
struct RGBAImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]
}

/// Android 版の Bitmap 操作（縮小・切り抜き・JPEG・モデルの入力）にあたるもの。
enum ImageUtil {
    private static let colorSpace = CGColorSpaceCreateDeviceRGB()

    // ------------------------------------------------------------ バイト列のまま扱う（解析で使う）

    /// 長辺を longSide 以下に縮める（大きくはしない）。双一次補間。
    static func scaled(_ image: RGBAImage, longSide: Int) -> RGBAImage {
        let k = Float(longSide) / Float(max(image.width, image.height))
        if k >= 1 { return image }
        let w = max(1, Int((Float(image.width) * k).rounded()))
        let h = max(1, Int((Float(image.height) * k).rounded()))
        var out = [UInt8](repeating: 255, count: w * h * 4)
        image.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<h {
                    let fy = ((Float(y) + 0.5) / k - 0.5).clamped(0, Float(image.height - 1))
                    for x in 0..<w {
                        let fx = ((Float(x) + 0.5) / k - 0.5).clamped(0, Float(image.width - 1))
                        let o = (y * w + x) * 4
                        for c in 0..<3 {
                            dst[o + c] = UInt8(clamping: Int(sample(src, image.width, fx, fy, c, 0, 0, image.width - 1, image.height - 1) + 0.5))
                        }
                    }
                }
            }
        }
        return RGBAImage(width: w, height: h, pixels: out)
    }

    /// ピクセルで切り抜く（左上原点、右端と下端は含まない）。
    static func crop(_ image: RGBAImage, _ l: Int, _ t: Int, _ r: Int, _ b: Int) -> RGBAImage? {
        guard l >= 0, t >= 0, r <= image.width, b <= image.height, r > l, b > t else { return nil }
        let w = r - l
        var out = [UInt8](repeating: 0, count: w * (b - t) * 4)
        image.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in t..<b {
                    let from = (y * image.width + l) * 4
                    let to = (y - t) * w * 4
                    for i in 0..<(w * 4) { dst[to + i] = src[from + i] }
                }
            }
        }
        return RGBAImage(width: w, height: b - t, pixels: out)
    }

    /**
     * モデルの入力（CHW、0〜1。平均や分散では割らない、PC と同じ）を out[base...] に直接書く。
     * size × size の正方形を fill で塗り、image の source の範囲（ピクセル）を dest（正方形の中のピクセル）に双一次補間で描いたものと同じ。
     */
    static func chw(
        _ image: RGBAImage,
        source: (x: Int, y: Int, w: Int, h: Int),
        size: Int,
        dest: (x: Float, y: Float, w: Float, h: Float),
        fill: (UInt8, UInt8, UInt8),
        into out: inout [Float],
        at base: Int
    ) {
        let plane = size * size
        let fills = [Float(fill.0) / 255, Float(fill.1) / 255, Float(fill.2) / 255]
        let x0 = source.x, y0 = source.y
        let x1 = source.x + source.w - 1, y1 = source.y + source.h - 1
        image.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for oy in 0..<size {
                    let cy = Float(oy) + 0.5
                    let inY = cy >= dest.y && cy < dest.y + dest.h
                    let fy = (Float(source.y) + (cy - dest.y) / dest.h * Float(source.h) - 0.5).clamped(Float(y0), Float(y1))
                    for ox in 0..<size {
                        let i = oy * size + ox
                        let cx = Float(ox) + 0.5
                        if !inY || cx < dest.x || cx >= dest.x + dest.w {
                            for c in 0..<3 { dst[base + c * plane + i] = fills[c] }
                            continue
                        }
                        let fx = (Float(source.x) + (cx - dest.x) / dest.w * Float(source.w) - 0.5).clamped(Float(x0), Float(x1))
                        for c in 0..<3 {
                            dst[base + c * plane + i] = sample(src, image.width, fx, fy, c, x0, y0, x1, y1) / 255
                        }
                    }
                }
            }
        }
    }

    /// (fx, fy) の色 c を双一次補間で。範囲は x0...x1, y0...y1 の中。
    @inline(__always)
    private static func sample(_ p: UnsafeBufferPointer<UInt8>, _ width: Int, _ fx: Float, _ fy: Float, _ c: Int,
                               _ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) -> Float {
        let ix = Int(fx), iy = Int(fy)
        let jx = min(ix + 1, x1), jy = min(iy + 1, y1)
        let ax = fx - Float(ix), ay = fy - Float(iy)
        let a = Float(p[(iy * width + ix) * 4 + c]), b = Float(p[(iy * width + jx) * 4 + c])
        let d = Float(p[(jy * width + ix) * 4 + c]), e = Float(p[(jy * width + jx) * 4 + c])
        let top = a + (b - a) * ax
        let bottom = d + (e - d) * ax
        return top + (bottom - top) * ay
    }

    // ------------------------------------------------------------ CGImage（JPEG にするとき、人の検出、記録だけ）

    static func cgImage(_ image: RGBAImage) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(image.pixels) as CFData) else { return nil }
        return CGImage(
            width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: image.width * 4,
            space: colorSpace, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
        )
    }

    static func jpeg(_ image: RGBAImage, longSide: Int = Int.max, quality: CGFloat = 0.85) -> Data {
        autoreleasepool {
            guard let cg = cgImage(scaled(image, longSide: longSide)) else { return Data() }
            return UIImage(cgImage: cg).jpegData(compressionQuality: quality) ?? Data()
        }
    }
}
