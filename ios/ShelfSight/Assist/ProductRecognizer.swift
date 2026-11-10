import Foundation

/**
 * 商品らしいものの枠を YOLO（SKU-110K で学習）で出し、切り抜きを MobileCLIP-S0 の埋め込みと分類層で 122 カテゴリに分ける。
 * モデルは tools/export_products.py などで app/src/main/assets/products/ に書き出したものを、そのまま使う。
 */
final class ProductRecognizer {
    private static let directory = "products"
    /// 地図に置く確からしさ（NeighborConfidence で調整したもの）。検証写真で、調整後 0.7 以上は枠の 53% を残して 91% が正解だった（調整前は 52% で 87%）。
    static let knownProb: Float = 0.7
    private static let topCategories = 5

    private struct Category {
        let name: String
        let genre: String
    }

    private let detector: Onnx.Model
    private let embedder: Onnx.Model
    private let detSize: Int
    private let embSize: Int
    private let detThreshold: Float
    private let nmsIou: Float
    private let categories: [Category]
    private let head: CategoryHead

    init() throws {
        func url(_ name: String, _ ext: String) throws -> URL {
            guard let u = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: ProductRecognizer.directory) else {
                throw Onnx.Failure(description: "\(ProductRecognizer.directory)/\(name).\(ext) がアプリに入っていません")
            }
            return u
        }
        guard let meta = JSON.object(try Data(contentsOf: url("gallery", "json"))),
              let headMeta = JSON.object(try Data(contentsOf: url("category_head", "json"))) else {
            throw Onnx.Failure(description: "gallery.json か category_head.json を読めません")
        }
        detector = try Onnx.Model(path: try url("detector", "onnx").path, threads: 4)
        embedder = try Onnx.Model(path: try url("mobileclip_s0", "onnx").path, threads: 4)
        detSize = JSON.int(meta, "detector_size") ?? 640
        embSize = JSON.int(meta, "embed_size") ?? 256
        detThreshold = Float(JSON.double(meta, "det_threshold") ?? 0.15)
        nmsIou = Float(JSON.double(meta, "nms_iou") ?? 0.5)
        categories = JSON.objects(headMeta, "categories").map { Category(name: JSON.string($0, "name"), genre: JSON.string($0, "genre")) }
        let bytes = try Data(contentsOf: url("category_head", "bin"))
        let weights = bytes.withUnsafeBytes { raw in
            (0..<(raw.count / 4)).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
        }
        guard let h = CategoryHead(dim: JSON.int(headMeta, "dim") ?? 0, hidden: JSON.int(headMeta, "hidden") ?? 0, classes: categories.count, weights: weights) else {
            throw Onnx.Failure(description: "category_head.bin の大きさが合いません")
        }
        head = h
    }

    /// 商品の枠（正立画像の 0..1、[left, top, right, bottom]）と確からしさ。
    func detect(_ image: RGBAImage) throws -> [([Float], Float)] {
        let scale = Float(detSize) / Float(max(image.width, image.height))
        let nw = max(1, Int((Float(image.width) * scale).rounded()))
        let nh = max(1, Int((Float(image.height) * scale).rounded()))
        let plane = detSize * detSize
        var input = [Float](repeating: 0, count: 3 * plane)
        ImageUtil.chw(
            image, source: (0, 0, image.width, image.height), size: detSize,
            dest: (0, 0, Float(nw), Float(nh)), fill: (114, 114, 114), into: &input, at: 0
        )
        let (out, shape) = try detector.run(input, shape: [1, 3, detSize, detSize])
        // 出力は [1, 4 + クラス数, 候補の数]（cx, cy, w, h, 確からしさ）
        let n = shape.count >= 3 ? shape[2] : out.count / 5
        func o(_ r: Int, _ i: Int) -> Float { out[r * n + i] }
        var cand: [([Float], Float)] = []
        for i in 0..<n where o(4, i) >= detThreshold {
            let cx = o(0, i), cy = o(1, i), w = o(2, i), h = o(3, i)
            cand.append(([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], o(4, i)))
        }
        let fw = Float(nw), fh = Float(nh)
        return ProductRecognizer.nms(cand, nmsIou).map { b, s in
            ([(b[0] / fw).clamped(0, 1), (b[1] / fh).clamped(0, 1), (b[2] / fw).clamped(0, 1), (b[3] / fh).clamped(0, 1)], s)
        }
    }

    /// 枠の切り抜き（分類層の学習と同じく広げない）。
    func cropOf(_ image: RGBAImage, _ b: [Float]) -> RGBAImage? {
        let W = image.width, H = image.height
        let l = Int(b[0] * Float(W)).clamped(0, W - 1)
        let t = Int(b[1] * Float(H)).clamped(0, H - 1)
        let r = Int(b[2] * Float(W)).clamped(l + 1, W)
        let btm = Int(b[3] * Float(H)).clamped(t + 1, H)
        return ImageUtil.crop(image, l, t, r, btm)
    }

    /// 切り抜き 1 枚を 122 カテゴリに分類する。1 位の商品と、確率の高い順の上位 5 つ（NeighborConfidence に渡す）を返す。
    func judge(_ crop: RGBAImage, _ box: [Float]) throws -> (RecognizedProduct, [(String, Float)]) {
        var input = [Float](repeating: 0, count: 3 * embSize * embSize)
        // 正方形に引き伸ばす（学習と同じ作り方）
        ImageUtil.chw(
            crop, source: (0, 0, crop.width, crop.height), size: embSize,
            dest: (0, 0, Float(embSize), Float(embSize)), fill: (0, 0, 0), into: &input, at: 0
        )
        let (out, _) = try embedder.run(input, shape: [1, 3, embSize, embSize])
        let p = head.probabilities(Self.normalize(out))
        let order = Array(Array(p.indices).stableSorted { p[$0] > p[$1] }.prefix(Self.topCategories))
        let c = categories[order[0]]
        return (RecognizedProduct(left: box[0], top: box[1], right: box[2], bottom: box[3], name: c.name, genre: c.genre, fine: c.name, sim: p[order[0]]),
                order.map { (categories[$0].name, p[$0]) })
    }

    private static func normalize(_ v: [Float]) -> [Float] {
        let n = max(v.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-6)
        return v.map { $0 / n }
    }

    static func iou(_ a: [Float], _ b: [Float]) -> Float {
        let w = max(min(a[2], b[2]) - max(a[0], b[0]), 0)
        let h = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
        let inter = w * h
        let union = SceneJudge.area(a) + SceneJudge.area(b) - inter
        return union > 0 ? inter / union : 0
    }

    static func nms(_ boxes: [([Float], Float)], _ threshold: Float) -> [([Float], Float)] {
        var out: [([Float], Float)] = []
        for c in boxes.stableSorted(by: { $0.1 > $1.1 }) where !out.contains(where: { iou($0.0, c.0) > threshold }) {
            out.append(c)
        }
        return out
    }
}

/// MobileCLIP の埋め込みから 122 カテゴリの確率を出す分類層（全結合 → ReLU → 全結合）。weights は tools/export_category_head.py の順。
final class CategoryHead {
    private let dim: Int, hidden: Int, classes: Int
    private let weights: [Float]

    init?(dim: Int, hidden: Int, classes: Int, weights: [Float]) {
        guard dim > 0, hidden > 0, classes > 0, weights.count == (dim + 1) * hidden + (hidden + 1) * classes else { return nil }
        self.dim = dim; self.hidden = hidden; self.classes = classes; self.weights = weights
    }

    /// カテゴリごとの確率（softmax）。embedding は長さ 1 に正規化したもの。
    func probabilities(_ embedding: [Float]) -> [Float] {
        let middle = layer(embedding, dim, hidden, 0).map { max($0, 0) }
        let logits = layer(middle, hidden, classes, (dim + 1) * hidden)
        let top = logits.max() ?? 0
        let e = logits.map { exp($0 - top) }
        let sum = e.reduce(0, +)
        return e.map { $0 / sum }
    }

    private func layer(_ input: [Float], _ inputs: Int, _ outputs: Int, _ offset: Int) -> [Float] {
        let bias = offset + inputs * outputs
        return weights.withUnsafeBufferPointer { w in
            input.withUnsafeBufferPointer { x in
                (0..<outputs).map { row in
                    var value = w[bias + row]
                    let start = offset + row * inputs
                    for k in 0..<inputs { value += w[start + k] * x[k] }
                    return value
                }
            }
        }
    }
}
