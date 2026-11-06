import Foundation

/// OCR が読み取った 1 行。大きさと位置は画像の高さを 1 とした割合。
struct OcrLine {
    let text: String
    let height: Float
    let top: Float
}

/// 商品を読み取った結果。
struct ProductResult {
    /// 商品名らしい文字を読めたときだけ true。
    let readable: Bool
    let name: String
    let ocrText: [String]
}

/// Vision API の応答（fullTextAnnotation）を行に直す。
enum CloudVisionParser {
    static func lines(_ json: String, imageHeight: Int) throws -> [OcrLine] {
        guard let response = JSON.objects(JSON.object(json), "responses").first else {
            throw CloudError(failure: .badResponse, detail: "Unexpected Cloud Vision response")
        }
        if let error = response["error"] as? [String: Any] {
            throw CloudError(failure: .badResponse, detail: "Cloud Vision error: \(JSON.string(error, "message"))")
        }
        let annotation = response["fullTextAnnotation"] as? [String: Any]
        let pages = JSON.objects(annotation, "pages")
        if pages.isEmpty { return [] }
        let h = Float(max(imageHeight, 1))
        var out: [OcrLine] = []
        var text = ""
        var heights: [Int] = []
        var top = Int.max

        func endLine() {
            if !text.trimmingCharacters(in: .whitespaces).isEmpty && !heights.isEmpty {
                // 行全体の枠は、箱が少し傾くだけで高くなるので、1 文字ずつの高さの中央値を使う
                let median = heights.sorted()[heights.count / 2]
                out.append(OcrLine(text: text.trimmingCharacters(in: .whitespaces), height: Float(median) / h, top: Float(top) / h))
            }
            text = ""
            heights = []
            top = Int.max
        }

        for page in pages {
            for block in JSON.objects(page, "blocks") {
                for paragraph in JSON.objects(block, "paragraphs") {
                    for word in JSON.objects(paragraph, "words") {
                        for symbol in JSON.objects(word, "symbols") {
                            text += JSON.string(symbol, "text")
                            let ys = vertices(symbol).map(\.1)
                            if let lo = ys.min(), let hi = ys.max() {
                                heights.append(hi - lo)
                                top = min(top, lo)
                            }
                            let property = symbol["property"] as? [String: Any]
                            let brk = JSON.string(property?["detectedBreak"] as? [String: Any], "type")
                            switch brk {
                            case "SPACE", "SURE_SPACE": text += " "
                            case "EOL_SURE_SPACE", "LINE_BREAK": endLine()
                            default: break   // 改行のハイフンは単語の途中なので行を続ける
                            }
                        }
                    }
                }
            }
        }
        endLine()
        return out
    }

    /// 0 の座標は省略されて返る。
    private static func vertices(_ o: [String: Any]) -> [(Int, Int)] {
        JSON.objects(o["boundingBox"] as? [String: Any], "vertices").map {
            (max(0, JSON.int($0, "x") ?? 0), max(0, JSON.int($0, "y") ?? 0))
        }
    }
}

/// 読み取った行の中から商品名を選ぶ。パッケージでは商品名がいちばん大きく印刷されている。
enum ProductReading {
    /// これより小さい文字は、背景の棚や値札であることが多い。
    private static let minNameHeight: Float = 0.012
    /// 2 行に分けて印刷された商品名をつなげる範囲。
    private static let sameNameHeightRatio: Float = 0.8
    private static let sameNameMaxGapLines: Float = 2.5

    static func from(_ lines: [OcrLine]) -> ProductResult {
        let texts = lines.map { $0.text.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        let candidates = lines.filter { isNameLike($0.text) && $0.height >= minNameHeight }
        guard let largest = candidates.max(by: { $0.height < $1.height }) else {
            return ProductResult(readable: false, name: "", ocrText: texts)
        }
        let name = candidates
            .filter { $0.height >= largest.height * sameNameHeightRatio && abs($0.top - largest.top) <= largest.height * sameNameMaxGapLines }
            .stableSorted { $0.top < $1.top }
            .map { $0.text.trimmingCharacters(in: .whitespaces) }
            .joined(separator: " ")
        return ProductResult(readable: true, name: name, ocrText: texts)
    }

    /// ロゴの 1 文字や、数字だけの行を商品名にしない。「お茶」のように短い日本語の名前は認める。
    private static func isNameLike(_ text: String) -> Bool {
        let letters = text.filter(\.isLetter)
        let japanese = letters.filter { ($0.unicodeScalars.first?.value ?? 0) > 0x2E7F }.count
        return japanese >= 2 || letters.count >= 3
    }
}
