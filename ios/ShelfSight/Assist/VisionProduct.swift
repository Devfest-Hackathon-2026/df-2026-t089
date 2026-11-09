import CoreGraphics
import Foundation

/// 調べた商品。分かれば商品名を先に言い、推定したカテゴリを続ける。社名は言わない。
struct ProductAnswer {
    /// 商品名（分からなければ空）。
    let product: String
    /// categories の中の 1 つ（分からなければ nil）。
    let category: String?
    /// 声で言う種類（炭酸水、ビール、グミなど）。
    let kind: String?
    /// false なら「〜のようです」と控えめに言う。
    let sure: Bool
    let alcohol: Bool
    /// カテゴリはお酒でないが、お酒の手がかり（「ビール」の文字、年齢確認の掲示など）もあった。
    let maybeAlcohol: Bool
    /// 決めたところ（「端末」「Gemini」など。画面とログ用）。
    let source: String
    /// 端末で数えた手がかり（画面とログ用）。
    let detail: String

    /// 例:「黒ラベル。お酒、ビールです。」「ウィルキンソン。炭酸水です。」「グミのようです。」
    func spoken() -> String {
        var s = ""
        if !product.isEmpty { s += "\(product)。" }
        if let kind {
            let k = alcohol ? "お酒、\(kind)" : kind
            s += sure ? "\(k)です。" : "\(k)のようです。"
        } else {
            s += "種類は分かりませんでした。"
        }
        if maybeAlcohol { s += "お酒かもしれません。" }
        return s
    }

    func describe() -> String {
        "種類「\(kind ?? "-")」\(sure ? "" : "（推測）") 商品「\(product)」\(alcohol ? " お酒" : "")\(maybeAlcohol ? " お酒かも" : "")（\(source)、\(detail)）"
    }
}

/// Cloud Vision API に、印字の文字（DOCUMENT_TEXT_DETECTION）だけを頼む。
/// Web 検索（逆画像検索）とロゴは使わない（手に持った商品の写真ではネットの画像と一致せず、ビールを「Sparkling Water」と答えたこともあった）。
enum VisionProductParser {
    static func request(_ jpegBase64: String) -> Data {
        let request: [String: Any] = [
            "image": ["content": jpegBase64],
            "features": [["type": "DOCUMENT_TEXT_DETECTION"]],
            "imageContext": ["languageHints": ["ja"]],
        ]
        return JSON.serialize(["requests": [request]])
    }
}

/**
 * 手前の商品 1 つを Cloud Vision API で読み、カテゴリと商品名を返す（カテゴリは CategoryJudge が決める）。
 * 商品に近づいている間（NearProduct で 1 つだけ特に近い商品が見え続けている間）は、結果を覚えておき API を呼び直さない。
 * 新しい商品に近づき直したら newApproach で忘れる。これで、長押しを何度しても 1 回の接近につき 1 回しか課金されない。
 * 調べるたびに、送った画像・応答・答えを Documents/lookup に残す（LookupLog）。
 */
final class ProductLookup {
    private static let url = "https://vision.googleapis.com/v1/images:annotate"
    private static let jpegQuality: CGFloat = 0.9

    private let apiKey: String
    private let judge: CategoryJudge
    @Locked private var cached: Result<ProductAnswer?, Error>? = nil

    init(apiKey: String, judge: CategoryJudge) {
        self.apiKey = apiKey
        self.judge = judge
    }

    var configured: Bool { !apiKey.isEmpty }

    /// この接近ですでに調べた結果（まだなら nil）。
    var cachedResult: Result<ProductAnswer?, Error>? { cached }

    func newApproach() { cached = nil }

    /// 調べる。この接近ですでに調べていれば、その結果を返す。通信を待つのでメインスレッドでは呼ばない。
    func lookup(_ image: RGBAImage) -> Result<ProductAnswer?, Error> {
        if let c = cached { return c }
        let r = Result<ProductAnswer?, Error> {
            guard configured else { throw CloudError(failure: .auth, detail: "Cloud Vision API key is not set") }
            let t0 = nowMs()
            let jpeg = ImageUtil.jpeg(image, quality: Self.jpegQuality)
            let body = VisionProductParser.request(jpeg.base64EncodedString())
            let res: HttpResult
            do {
                res = try Http.post(Self.url, headers: ["x-goog-api-key": apiKey, "Content-Type": "application/json"], body: body)
            } catch {
                LookupLog.save(jpeg: jpeg, response: nil, fields: [("result", "network error: \(error)"), ("ms", "\(nowMs() - t0)")])
                throw CloudError(failure: .network, detail: "Cloud Vision unreachable: \(error)")
            }
            let visionMs = nowMs() - t0
            guard (200...299).contains(res.code) else {
                LookupLog.save(jpeg: jpeg, response: res.body, fields: [("result", "HTTP \(res.code)"), ("ms", "\(visionMs)")])
                throw CloudError(failure: .badResponse, detail: "Cloud Vision HTTP \(res.code): \(res.body.prefix(300))")
            }
            let lines: [OcrLine]
            do {
                lines = try CloudVisionParser.lines(res.body, imageWidth: image.width, imageHeight: image.height)
            } catch {
                LookupLog.save(jpeg: jpeg, response: res.body, fields: [("result", "parse error: \(error)"), ("ms", "\(visionMs)")])
                throw error
            }
            let reading = ProductReading.from(lines)
            let outcome = judge.judge(lines, ocrName: reading.readable ? reading.name : "")
            let a = outcome.answer
            logInfo("Assist", "lookup: \(a?.describe() ?? "分からない"), vision \(visionMs)ms, gemini \(outcome.geminiMs.map { "\($0)ms" } ?? "-")")
            LookupLog.save(jpeg: jpeg, response: res.body, fields: [
                ("result", a == nil ? "nil" : "ok"),
                ("spoken", a?.spoken() ?? "（分からない）"),
                ("category", a?.category ?? ""),
                ("kind", a?.kind ?? ""),
                ("product", a?.product ?? ""),
                ("sure", a.map { "\($0.sure)" } ?? ""),
                ("alcohol", a.map { "\($0.alcohol)" } ?? ""),
                ("maybe_alcohol", a.map { "\($0.maybeAlcohol)" } ?? ""),
                ("source", a?.source ?? ""),
                ("ocr_name", reading.name),
                ("local", outcome.local.describe()),
                ("hits", outcome.local.hits.joined(separator: " ")),
                ("gemini_ms", outcome.geminiMs.map { "\($0)" } ?? ""),
                ("gemini_error", outcome.geminiError ?? ""),
                ("vision_ms", "\(visionMs)"),
                ("ms", "\(nowMs() - t0)"),
                ("image", "\(image.width)x\(image.height)"),
                ("jpeg_bytes", "\(jpeg.count)"),
            ])
            return a
        }
        // 分かったときだけ覚える。通信の失敗や「分からない」は覚えない（もう一度長押しすれば、撮り直して調べ直せる）
        if case .success(let a) = r, a != nil { cached = r }
        return r
    }
}

/**
 * 調べた記録を Documents/lookup に残す（失敗したときも）。<時刻>.jpg（送った画像）、<時刻>.json（Cloud Vision の応答。通信できなかったときはない）、
 * <時刻>.answer.txt（答え・言った文・点数の内訳・かかった時間）。
 * 「ファイル」アプリの「このiPhone内 > ShelfSight > lookup」か、USB でつないだ PC から取り出して、判定の改善に使う。
 */
enum LookupLog {
    private static let queue = DispatchQueue(label: "lookup-log", qos: .utility)

    static func save(jpeg: Data, response: String?, fields: [(String, String)]) {
        queue.async {
            guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
            let dir = docs.appendingPathComponent("lookup", isDirectory: true)
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                let f = DateFormatter()
                f.locale = Locale(identifier: "en_US_POSIX")
                f.dateFormat = "yyyyMMdd_HHmmss_SSS"
                let stem = f.string(from: Date())
                try jpeg.write(to: dir.appendingPathComponent("\(stem).jpg"))
                if let response { try Data(response.utf8).write(to: dir.appendingPathComponent("\(stem).json")) }
                let text = fields.map { "\($0.0)=\($0.1.replacingOccurrences(of: "\n", with: " "))" }.joined(separator: "\n") + "\n"
                try Data(text.utf8).write(to: dir.appendingPathComponent("\(stem).answer.txt"))
            } catch {
                logInfo("Assist", "lookup log failed: \(error)")
            }
        }
    }
}
