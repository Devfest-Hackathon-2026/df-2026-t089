import CoreGraphics
import Foundation

/**
 * 商品認識ダッシュボード（products/）で作ったモデルで、写っている商品を見つけて名前を付ける（products/infer.py の移植）。
 *
 * 1. YOLO で「商品らしいもの」の枠を全部出す（detect。写っているものの判定にも使う）
 * 2. 各枠を切り抜いて埋め込み（MobileNetV3-small 128px）を出し、照合先（命名済みの切り抜き）と比べて名前の候補を出す
 * 3. 周りの商品のジャンル・細分類で選び直す（ContextAdjust）
 *
 * モデルと照合先は tools/export_products.py で app/src/main/assets/products/ に書き出したものを、そのまま使う。
 */
final class ProductRecognizer {
    private static let directory = "products"
    private static let topK = 10
    private static let minCropPx = 14
    /// 名前を付ける切り抜きの数の目安。照合が切り抜きの数に比例して遅くなるので抑える（区画が多く埋まっていればこれを超える）。
    static let maxCrops = 10
    /// 散らして選ぶための画像の区画（横 × 縦）。名前を付ける数は多くてもこの区画の数まで。
    static let spreadCols = 4
    static let spreadRows = 4

    private struct Product {
        let key: String
        let name: String
        let genre: String?
        let fine: String?
        let start: Int
    }

    private let detector: Onnx.Model
    private let embedder: Onnx.Model
    private let detSize: Int
    private let embSize: Int
    private let dim: Int
    private let detThreshold: Float
    private let nmsIou: Float
    private let matchSim: Float
    private let lambdas: (Double, Double)
    private let pad: (UInt8, UInt8, UInt8)
    private let products: [Product]
    private let gallery: [Float]
    private let byKey: [String: Product]

    init() throws {
        func url(_ name: String, _ ext: String) throws -> URL {
            guard let u = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: ProductRecognizer.directory) else {
                throw Onnx.Failure(description: "\(ProductRecognizer.directory)/\(name).\(ext) がアプリに入っていません")
            }
            return u
        }
        let metaData = try Data(contentsOf: url("gallery", "json"))
        guard let meta = JSON.object(metaData) else {
            throw Onnx.Failure(description: "gallery.json を読めません")
        }
        let detectorPath = try url("detector", "onnx").path
        let embedPath = try url("embed", "onnx").path
        let galleryURL = try url("gallery", "bin")
        detector = try Onnx.Model(path: detectorPath, threads: 4)
        embedder = try Onnx.Model(path: embedPath, threads: 4)
        detSize = JSON.int(meta, "detector_size") ?? 640
        embSize = JSON.int(meta, "embed_size") ?? 128
        dim = JSON.int(meta, "dim") ?? 128
        detThreshold = Float(JSON.double(meta, "det_threshold") ?? 0.25)
        nmsIou = Float(JSON.double(meta, "nms_iou") ?? 0.5)
        matchSim = Float(JSON.double(meta, "match_sim") ?? 0.5)
        let l = (meta["context_lambda"] as? [NSNumber])?.map(\.doubleValue) ?? [0.05, 0.05]
        lambdas = (l.count > 0 ? l[0] : 0.05, l.count > 1 ? l[1] : 0.05)
        let p = (meta["pad_color"] as? [NSNumber])?.map(\.uint8Value) ?? [124, 116, 104]
        pad = (p[0], p[1], p[2])
        let list = JSON.objects(meta, "products").map {
            Product(key: JSON.string($0, "key"), name: JSON.string($0, "name"),
                    genre: JSON.text($0, "genre"), fine: JSON.text($0, "fine"), start: JSON.int($0, "start") ?? 0)
        }
        products = list
        var keyed: [String: Product] = [:]
        for p in list { keyed[p.key] = p }
        byKey = keyed
        // 照合先は float16（リトルエンディアン）
        let count = JSON.int(meta, "count") ?? 0
        let raw = try Data(contentsOf: galleryURL)
        let n = min(count * dim, raw.count / 2)
        var g = [Float](repeating: 0, count: n)
        raw.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
            for i in 0..<n {
                let bits = UInt16(b[2 * i]) | (UInt16(b[2 * i + 1]) << 8)
                g[i] = ProductRecognizer.halfToFloat(bits)
            }
        }
        gallery = g
    }

    /// 商品の枠（正立画像の 0..1、[left, top, right, bottom]）と確からしさ。
    func detect(_ image: CGImage) throws -> [([Float], Float)] {
        let scale = Float(detSize) / Float(max(image.width, image.height))
        let nw = max(1, Int((Float(image.width) * scale).rounded()))
        let nh = max(1, Int((Float(image.height) * scale).rounded()))
        let px = ImageUtil.pixels(
            width: detSize, height: detSize, fill: (114, 114, 114), image: image,
            dest: CGRect(x: 0, y: 0, width: nw, height: nh)
        )
        let plane = detSize * detSize
        var input = [Float](repeating: 0, count: 3 * plane)
        ImageUtil.putCHW(px, plane: plane, into: &input, at: 0)
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

    /**
     * 写っている商品に名前を付ける。照合先にない商品（似ていない）は外す。
     * boxes は確からしさの高い順（detect の順）で、画像の各所から散らして maxCrops 個ほどに名前を付ける（spread）。
     */
    func recognize(_ image: CGImage, _ boxes: [[Float]]) throws -> [RecognizedProduct] {
        let minW = Float(ProductRecognizer.minCropPx) / Float(image.width)
        let minH = Float(ProductRecognizer.minCropPx) / Float(image.height)
        // 大きい順に選ぶと、棚の区画や商品の一部などの大きな誤検出ばかりになるので、確からしさの順に選ぶ。
        // ただし確からしさの順だけだと一か所に固まるので、画像の区画ごとに散らして選ぶ（spread）
        let big = ProductRecognizer.spread(boxes.filter { $0[2] - $0[0] >= minW && $0[3] - $0[1] >= minH })
        if big.isEmpty { return [] }
        let plane = embSize * embSize
        var input = [Float](repeating: 0, count: big.count * 3 * plane)
        for (j, b) in big.enumerated() {
            ImageUtil.putCHW(square(image, b), plane: plane, into: &input, at: j * 3 * plane)
        }
        let (out, shape) = try embedder.run(input, shape: [big.count, 3, embSize, embSize])
        let d = shape.last ?? dim
        let embs = (0..<big.count).map { ProductRecognizer.normalize(Array(out[(($0) * d)..<(($0 + 1) * d)])) }

        // 照合先と比べた、そのままの類似度。名前を付けるかどうかは補正前のこれで決める
        let raw = embs.map { topProducts($0) }
        let rawSim: [[String: Float]] = raw.map { c in
            var m: [String: Float] = [:]
            for x in c { m[x.key] = x.sim }
            return m
        }
        let centers = big.map { (($0[0] + $0[2]) / 2, ($0[1] + $0[3]) / 2) }
        let keyed = byKey
        let picked = ContextAdjust.adjust(centers, raw, [
            ContextAdjust.Level(categoryOf: { keyed[$0]?.genre }, lambda: lambdas.0),
            ContextAdjust.Level(categoryOf: { keyed[$0]?.fine }, lambda: lambdas.1),
        ])
        return big.indices.compactMap { i in
            let p = picked[i]
            guard let key = p.key, let prod = byKey[key] else { return nil }
            let sim = rawSim[i][key] ?? 0
            if sim < matchSim { return nil }
            let b = big[i]
            return RecognizedProduct(left: b[0], top: b[1], right: b[2], bottom: b[3], name: prod.name, genre: prod.genre, fine: prod.fine, sim: sim)
        }
    }

    /// 商品ごとの最大類似度の上位 topK。
    private func topProducts(_ e: [Float]) -> [ContextAdjust.Candidate] {
        let rows = gallery.count / dim
        var best = [Float](repeating: -1, count: products.count)
        gallery.withUnsafeBufferPointer { g in
            e.withUnsafeBufferPointer { v in
                for pi in 0..<products.count {
                    let end = pi + 1 < products.count ? products[pi + 1].start : rows
                    var m: Float = -1
                    var row = products[pi].start
                    while row < min(end, rows) {
                        var s: Float = 0
                        let o = row * dim
                        for k in 0..<dim { s += g[o + k] * v[k] }
                        if s > m { m = s }
                        row += 1
                    }
                    best[pi] = m
                }
            }
        }
        return Array(best.indices).stableSorted { best[$0] > best[$1] }.prefix(ProductRecognizer.topK)
            .map { ContextAdjust.Candidate(key: products[$0].key, sim: best[$0]) }
    }

    /// 枠を切り抜き、灰色で正方形に埋めて縮める（縦長の商品の形を残す）。
    private func square(_ image: CGImage, _ b: [Float]) -> [UInt8] {
        let W = image.width, H = image.height
        let l = Int(b[0] * Float(W)).clamped(0, W - 1)
        let t = Int(b[1] * Float(H)).clamped(0, H - 1)
        let r = Int(b[2] * Float(W)).clamped(l + 1, W)
        let btm = Int(b[3] * Float(H)).clamped(t + 1, H)
        let w = r - l, h = btm - t
        let s = Float(max(w, h))
        let k = Float(embSize) / s
        let dx = (s - Float(w)) / 2 * k
        let dy = (s - Float(h)) / 2 * k
        return ImageUtil.pixels(
            width: embSize, height: embSize, fill: pad, image: image,
            source: CGRect(x: l, y: t, width: w, height: h),
            dest: CGRect(x: CGFloat(dx), y: CGFloat(dy), width: CGFloat(Float(w) * k), height: CGFloat(Float(h) * k))
        )
    }

    private static func normalize(_ v: [Float]) -> [Float] {
        let n = max(v.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-6)
        return v.map { $0 / n }
    }

    /**
     * 名前を付ける枠を、画像のいろいろな場所から選ぶ。boxes は確からしさの高い順。
     * 画像を spreadCols×spreadRows の区画に分け（枠の中心で決める）、区画ごとに確からしさの高い順に 1 つずつ順番に取る。
     * 1 巡目は枠のある区画を全部取る（limit を超えてもよい）。2 巡目からは limit 個になったら止める。
     * 返す順は確からしさの高い順のまま。
     */
    static func spread(_ boxes: [[Float]], limit: Int = ProductRecognizer.maxCrops) -> [[Float]] {
        // いちばん確からしい枠のある区画から
        let cells = Array(boxes.indices).orderedGroups { i -> Int in
            let b = boxes[i]
            let cx = Int((b[0] + b[2]) / 2 * Float(spreadCols)).clamped(0, spreadCols - 1)
            let cy = Int((b[1] + b[3]) / 2 * Float(spreadRows)).clamped(0, spreadRows - 1)
            return cy * spreadCols + cx
        }.map(\.values)
        var picked: [Int] = []
        var round = 0
        while true {
            let next = cells.compactMap { round < $0.count ? $0[round] : nil }
            if next.isEmpty { break }
            if round == 0 {
                picked += next
            } else {
                for i in next {
                    if picked.count >= limit { break }
                    picked.append(i)
                }
            }
            if picked.count >= limit { break }
            round += 1
        }
        return picked.sorted().map { boxes[$0] }
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

    /// IEEE 754 の半精度を単精度にする（Float16 型はシミュレータの x86 で使えないので自前で）。
    private static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        var exp = UInt32((h >> 10) & 0x1F)
        var frac = UInt32(h & 0x3FF)
        if exp == 0 {
            if frac == 0 { return Float(bitPattern: sign) }
            // 非正規化数: 仮数の先頭の 1 が出るまでずらし、その分だけ指数を下げる
            var shift: UInt32 = 0
            while frac & 0x400 == 0 { frac <<= 1; shift += 1 }
            frac &= 0x3FF
            exp = 113 - shift
            return Float(bitPattern: sign | (exp << 23) | (frac << 13))
        }
        if exp == 0x1F { return Float(bitPattern: sign | 0x7F80_0000 | (frac << 13)) }
        return Float(bitPattern: sign | ((exp + 112) << 23) | (frac << 13))
    }
}
