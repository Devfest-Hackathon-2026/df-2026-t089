import Foundation

/// 正立画像から人の範囲を見つける。
protocol PersonDetector {
    func detect(_ image: RGBAImage) -> [NormalizedBox]
}

/// COCO で学習した YOLO の person クラスで人を見つける（Android 版と同じモデル。tools/export_person.py で作る）。
final class YoloPersonDetector: PersonDetector {
    private static let size = 416
    private static let minScore: Float = 0.4
    private static let nmsIou: Float = 0.5
    private static let personScore = 4   // 枠の 4 列のあとに、クラス 0 (person) の確からしさが続く

    /// 読み込みに時間がかかるので、最初の検出のときに作る。
    private lazy var model: Onnx.Model? = {
        guard let path = Bundle.main.path(forResource: "yolo26n_416", ofType: "onnx", inDirectory: "person") else {
            logInfo("Person", "person/yolo26n_416.onnx がアプリに入っていません")
            return nil
        }
        do {
            return try Onnx.Model(path: path, threads: 2)   // 商品の認識と CPU を分け合う
        } catch {
            logInfo("Person", "model: \(error)")
            return nil
        }
    }()

    func detect(_ image: RGBAImage) -> [NormalizedBox] {
        guard let model else { return [] }
        let size = Self.size
        let scale = Float(size) / Float(max(image.width, image.height))
        let width = max(1, Int((Float(image.width) * scale).rounded()))
        let height = max(1, Int((Float(image.height) * scale).rounded()))
        var input = [Float](repeating: 0, count: 3 * size * size)
        ImageUtil.chw(
            image, source: (0, 0, image.width, image.height), size: size,
            dest: (0, 0, Float(width), Float(height)), fill: (114, 114, 114), into: &input, at: 0
        )
        do {
            let (out, shape) = try model.run(input, shape: [1, 3, size, size])
            guard shape.count >= 3 else { return [] }
            return Self.decode(out, candidates: shape[2], width: width, height: height)
        } catch {
            logInfo("Person", "detection failed: \(error)")
            return []
        }
    }

    /// YOLO の出力（[cx, cy, w, h, person, ...] の列、ピクセル）を枠にする。width, height は余白を除いた画像の大きさ。
    static func decode(_ out: [Float], candidates n: Int, width: Int, height: Int) -> [NormalizedBox] {
        func o(_ r: Int, _ i: Int) -> Float { out[r * n + i] }
        var cand: [([Float], Float)] = []
        for i in 0..<n where o(personScore, i) >= minScore {
            let cx = o(0, i), cy = o(1, i), w = o(2, i), h = o(3, i)
            cand.append(([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], o(personScore, i)))
        }
        let fw = Float(width), fh = Float(height)
        return ProductRecognizer.nms(cand, nmsIou).map { b, _ in
            NormalizedBox(left: (b[0] / fw).clamped(0, 1), top: (b[1] / fh).clamped(0, 1), right: (b[2] / fw).clamped(0, 1), bottom: (b[3] / fh).clamped(0, 1))
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
