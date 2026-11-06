import CoreGraphics
import Foundation

/// 調べた商品。社名が分からないときは空。
struct ProductAnswer {
    let maker: String
    let product: String

    func spoken() -> String {
        if !maker.isEmpty && !product.isEmpty { return "\(maker)の、\(product)です。" }
        if !product.isEmpty { return "\(product)です。" }
        return "\(maker)の商品です。"
    }
}

/**
 * Cloud Vision API の応答（WEB_DETECTION・LOGO_DETECTION・TEXT_DETECTION を 1 回で頼んだもの）から、社名と商品名を決める。
 * Web 検索（逆画像検索）の判定は products/s3_name.py の judge を移したもの。Web で分からなければ、ロゴと印刷の文字で決める。
 */
enum VisionProductParser {
    static func request(_ jpegBase64: String) -> Data {
        let features: [[String: Any]] = [
            ["type": "WEB_DETECTION", "maxResults": 10],
            ["type": "LOGO_DETECTION", "maxResults": 3],
            ["type": "DOCUMENT_TEXT_DETECTION"],
        ]
        let imageContext: [String: Any] = [
            "languageHints": ["ja"],
            "webDetectionParams": ["includeGeoResults": false],
        ]
        let request: [String: Any] = [
            "image": ["content": jpegBase64],
            "features": features,
            "imageContext": imageContext,
        ]
        return JSON.serialize(["requests": [request]])
    }

    /// 分からなければ nil。imageHeight は送った画像の高さ（文字の大きさを比べるため）。
    static func parse(_ json: String, imageHeight: Int) throws -> ProductAnswer? {
        guard let r = JSON.objects(JSON.object(json), "responses").first else {
            throw CloudError(failure: .badResponse, detail: "Unexpected Cloud Vision response")
        }
        if let error = r["error"] as? [String: Any] {
            throw CloudError(failure: .badResponse, detail: "Cloud Vision error: \(JSON.string(error, "message"))")
        }
        let web = r["webDetection"] as? [String: Any]
        let best = JSON.objects(web, "bestGuessLabels").map { JSON.string($0, "label") }
            .first { !$0.trimmingCharacters(in: .whitespaces).isEmpty } ?? ""
        let entities = JSON.objects(web, "webEntities")
            .filter { !JSON.string($0, "description").trimmingCharacters(in: .whitespaces).isEmpty }
            .map { (JSON.string($0, "description"), JSON.double($0, "score") ?? 0) }
        let pages = JSON.objects(web, "pagesWithMatchingImages").map { JSON.string($0, "pageTitle") }
        let matches = JSON.objects(web, "fullMatchingImages").count + JSON.objects(web, "partialMatchingImages").count
        let specific = entities.filter { !generic.contains($0.0.lowercased()) && $0.1 >= 0.5 }.map(\.0)
        let webName = (!best.isEmpty && !generic.contains(best.lowercased())) ? best : (specific.first ?? "")
        let webGood = !webName.isEmpty && (matches > 0 || !pages.isEmpty || !specific.isEmpty)

        let logos = JSON.objects(r, "logoAnnotations").map { JSON.string($0, "description") }
        let text = ([best] + entities.map(\.0) + pages + logos).joined(separator: " ").lowercased()
        let maker = makerIn(best.lowercased()) ?? makerIn(text) ?? logos.first { !$0.trimmingCharacters(in: .whitespaces).isEmpty } ?? ""

        let printed = (try? CloudVisionParser.lines(json, imageHeight: imageHeight)).map { ProductReading.from($0) }
        let product: String
        if webGood {
            product = withoutMaker(webName, maker)
        } else if let p = printed, p.readable {
            product = withoutMaker(p.name, maker)
        } else {
            product = ""
        }
        if product.isEmpty && maker.isEmpty { return nil }
        return ProductAnswer(maker: maker, product: product)
    }

    static func makerIn(_ text: String) -> String? {
        makers.first { _, aliases in aliases.contains { text.contains($0) } }?.0
    }

    /// 商品名から社名の部分を取り除く。
    static func withoutMaker(_ name: String, _ maker: String) -> String {
        var out = name
        let aliases = (makers.first { $0.0 == maker }?.1 ?? []) + [maker]
        for a in aliases where !a.isEmpty { out = out.replacingOccurrences(of: a, with: "", options: .caseInsensitive) }
        out = out.trimmingCharacters(in: CharacterSet(charactersIn: " -・　"))
        return out.isEmpty ? name.trimmingCharacters(in: .whitespaces) : out
    }

    /// 商品名として意味のない一般語（best guess がこれだけなら Web の結果は使わない）。
    private static let generic: Set<String> = [
        "product", "products", "packaging and labeling", "packaging", "label", "brand", "bottle", "plastic bottle",
        "glass bottle", "convenience store", "store", "shelf", "retail", "supermarket", "cosmetics", "snack", "food",
        "drink", "box", "carton", "font", "logo", "advertising", "plastic", "paper", "personal care", "skin care",
        "beauty", "health", "liquid", "water", "juice", "tea", "beer", "sock", "socks", "textile", "material property",
        "rectangle", "japan", "japanese", "graphics", "design", "pattern", "illustration", "art", "image", "photograph",
        "medicine", "pharmaceutical drug", "household supply", "hygiene", "lipstick", "toothbrush", "cosmetic",
        "makeup", "comb", "hair care", "electric blue", "magenta", "pink", "red", "white", "black", "transparency",
        "convenience food", "flavor", "ingredient", "fast food", "candy", "confectionery", "pharmacy", "drugstore",
    ]

    /// 社名の辞書（表記ゆれ → 正式名）。products/s3_name.py と同じ。上から順に探す。
    static let makers: [(String, [String])] = [
        ("花王", ["花王", "kao"]), ("ライオン", ["ライオン", "lion corporation"]), ("資生堂", ["資生堂", "shiseido"]),
        ("P&G", ["p&g", "procter", "ピーアンドジー"]), ("ユニリーバ", ["ユニリーバ", "unilever"]),
        ("ユニ・チャーム", ["ユニ・チャーム", "ユニチャーム", "unicharm"]), ("小林製薬", ["小林製薬", "kobayashi"]),
        ("ロート製薬", ["ロート製薬", "rohto"]), ("大正製薬", ["大正製薬", "taisho"]),
        ("サントリー", ["サントリー", "suntory"]), ("アサヒ", ["アサヒ", "asahi"]), ("キリン", ["キリン", "kirin"]),
        ("サッポロ", ["サッポロ", "sapporo"]), ("伊藤園", ["伊藤園", "ito en"]),
        ("コカ・コーラ", ["コカ・コーラ", "coca-cola", "coca cola"]), ("ダイドー", ["ダイドー", "dydo"]),
        ("明治", ["明治", "meiji"]), ("森永", ["森永", "morinaga"]), ("ロッテ", ["ロッテ", "lotte"]),
        ("グリコ", ["グリコ", "glico"]), ("カルビー", ["カルビー", "calbee"]), ("湖池屋", ["湖池屋", "koikeya"]),
        ("日清", ["日清", "nissin"]), ("東洋水産", ["東洋水産", "マルちゃん", "maruchan"]),
        ("ブルボン", ["ブルボン", "bourbon"]), ("不二家", ["不二家", "fujiya"]),
        ("ハウス食品", ["ハウス食品", "house foods"]), ("味の素", ["味の素", "ajinomoto"]),
        ("キッコーマン", ["キッコーマン", "kikkoman"]), ("ヤクルト", ["ヤクルト", "yakult"]),
        ("雪印メグミルク", ["雪印", "megmilk"]), ("ファミリーマート", ["ファミリーマート", "familymart"]),
        ("セブン-イレブン", ["セブン-イレブン", "セブンイレブン", "7-eleven", "seven-eleven"]), ("ローソン", ["ローソン", "lawson"]),
        ("ジレット", ["gillette", "ジレット"]),
        ("ジョンソン・エンド・ジョンソン", ["johnson & johnson", "ジョンソン・エンド・ジョンソン", "band-aid"]),
        ("アース製薬", ["アース製薬", "earth corporation"]), ("エステー", ["エステー", "st corporation"]),
        ("コーセー", ["コーセー", "kose"]), ("マンダム", ["マンダム", "mandom"]), ("ピジョン", ["ピジョン", "pigeon"]),
        ("レッドブル", ["red bull", "レッドブル"]), ("モンスター", ["monster energy", "モンスターエナジー"]),
        ("ネスレ", ["ネスレ", "nestle", "nestlé"]), ("有楽製菓", ["有楽製菓", "yuraku", "ブラックサンダー"]),
    ]
}

/**
 * 手前の商品 1 つを Cloud Vision API で調べ、社名と商品名を返す。1 回の要求で Web 検索・ロゴ・文字を頼む。
 * 商品に近づいている間は結果を覚えておき API を呼び直さない。新しい商品に近づき直したら newApproach で忘れる。
 */
final class ProductLookup {
    private static let url = "https://vision.googleapis.com/v1/images:annotate"
    private static let jpegQuality: CGFloat = 0.9

    private let apiKey: String
    @Locked private var cached: Result<ProductAnswer?, Error>? = nil

    init(apiKey: String) { self.apiKey = apiKey }

    var configured: Bool { !apiKey.isEmpty }

    /// この接近ですでに調べた結果（まだなら nil）。
    var cachedResult: Result<ProductAnswer?, Error>? { cached }

    func newApproach() { cached = nil }

    /// 調べる。この接近ですでに調べていれば、その結果を返す。通信を待つのでメインスレッドでは呼ばない。
    func lookup(_ image: CGImage) -> Result<ProductAnswer?, Error> {
        if let c = cached { return c }
        let r = Result<ProductAnswer?, Error> {
            guard configured else { throw CloudError(failure: .auth, detail: "Cloud Vision API key is not set") }
            let jpeg = ImageUtil.jpeg(image, quality: Self.jpegQuality)
            let body = VisionProductParser.request(jpeg.base64EncodedString())
            let res: HttpResult
            do {
                res = try Http.post(Self.url, headers: ["x-goog-api-key": apiKey, "Content-Type": "application/json"], body: body)
            } catch {
                throw CloudError(failure: .network, detail: "Cloud Vision unreachable: \(error)")
            }
            guard (200...299).contains(res.code) else {
                throw CloudError(failure: .badResponse, detail: "Cloud Vision HTTP \(res.code): \(res.body.prefix(300))")
            }
            return try VisionProductParser.parse(res.body, imageHeight: image.height)
        }
        // 通信の失敗は覚えない（もう一度長押しすれば調べ直せる）
        if case .success = r { cached = r }
        return r
    }
}
