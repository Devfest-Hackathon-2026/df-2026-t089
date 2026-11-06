import CoreGraphics
import UIKit

/// Android 版の Bitmap 操作（縮小・切り抜き・JPEG・モデルの入力）にあたるもの。
enum ImageUtil {
    private static let colorSpace = CGColorSpaceCreateDeviceRGB()

    static func context(_ w: Int, _ h: Int) -> CGContext? {
        CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
    }

    /// 長辺を longSide 以下に縮める（大きくはしない）。
    static func scaled(_ image: CGImage, longSide: Int) -> CGImage {
        let k = min(1, Double(longSide) / Double(max(image.width, image.height)))
        if k >= 1 { return image }
        let w = max(1, Int((Double(image.width) * k).rounded()))
        let h = max(1, Int((Double(image.height) * k).rounded()))
        guard let ctx = context(w, h) else { return image }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage() ?? image
    }

    static func jpeg(_ image: CGImage, longSide: Int = Int.max, quality: CGFloat = 0.85) -> Data {
        UIImage(cgImage: scaled(image, longSide: longSide)).jpegData(compressionQuality: quality) ?? Data()
    }

    /// ピクセルで切り抜く（左上原点、右端と下端は含まない）。
    static func crop(_ image: CGImage, _ l: Int, _ t: Int, _ r: Int, _ b: Int) -> CGImage? {
        image.cropping(to: CGRect(x: l, y: t, width: r - l, height: b - t))
    }

    /**
     * width x height の画像を fill で塗り、image（source があればその範囲）を dest（左上原点のピクセル）に描いて、
     * RGBA のバイト列（上の行から）を返す。
     */
    static func pixels(
        width: Int, height: Int, fill: (UInt8, UInt8, UInt8),
        image: CGImage, source: CGRect? = nil, dest: CGRect
    ) -> [UInt8] {
        let count = width * height * 4
        guard let ctx = context(width, height) else { return [UInt8](repeating: 0, count: count) }
        ctx.setFillColor(red: CGFloat(fill.0) / 255, green: CGFloat(fill.1) / 255, blue: CGFloat(fill.2) / 255, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.interpolationQuality = .medium
        let src = source.flatMap { image.cropping(to: $0) } ?? image
        // CGContext は左下原点なので、上下を入れ替えた位置に描く
        ctx.draw(src, in: CGRect(x: dest.minX, y: CGFloat(height) - dest.maxY, width: dest.width, height: dest.height))
        guard let data = ctx.data else { return [UInt8](repeating: 0, count: count) }
        return Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: UInt8.self), count: count))
    }

    /// RGBA を、モデルの入力（CHW、0〜1）の out[base...] に書く。平均や分散では割らない（PC と同じ）。
    static func putCHW(_ rgba: [UInt8], plane: Int, into out: inout [Float], at base: Int) {
        rgba.withUnsafeBufferPointer { p in
            out.withUnsafeMutableBufferPointer { o in
                for i in 0..<plane {
                    o[base + i] = Float(p[4 * i]) / 255
                    o[base + plane + i] = Float(p[4 * i + 1]) / 255
                    o[base + 2 * plane + i] = Float(p[4 * i + 2]) / 255
                }
            }
        }
    }
}
