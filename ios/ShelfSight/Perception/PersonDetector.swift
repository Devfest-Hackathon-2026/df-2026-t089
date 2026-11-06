import CoreGraphics
import Vision

/// 正立画像から人の範囲を見つける。
protocol PersonDetector {
    func detect(_ image: CGImage) -> [NormalizedBox]
}

/**
 * Vision の人の検出による実装（Android 版の MediaPipe EfficientDet-Lite0 の代わり）。OS に入っているモデルを端末内で使う。
 */
final class VisionPersonDetector: PersonDetector {
    private static let minScore: Float = 0.5

    func detect(_ image: CGImage) -> [NormalizedBox] {
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = false   // 全身の枠（全身が写れば縦に長い、を使うため）
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            logInfo("Person", "detection failed: \(error)")
            return []
        }
        return (request.results ?? []).filter { $0.confidence >= Self.minScore }.map { o in
            // Vision は左下原点
            let r = o.boundingBox
            return NormalizedBox(
                left: Float(r.minX).clamped(0, 1),
                top: Float(1 - r.maxY).clamped(0, 1),
                right: Float(r.maxX).clamped(0, 1),
                bottom: Float(1 - r.minY).clamped(0, 1)
            )
        }
    }
}

/// 画面の下端から立ち上がり、上半分に届かない枠（自分の手や腕）。Android 版の OwnHandFilter.isLowerOnly。
enum OwnHandFilter {
    private static let minTopOfLowerOnlyBox: Float = 0.55
    private static let bottomEdge: Float = 0.98

    static func isLowerOnly(_ person: NormalizedBox) -> Bool {
        person.bottom >= bottomEdge && person.top >= minTopOfLowerOnlyBox
    }
}
