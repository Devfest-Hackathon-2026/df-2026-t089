import Foundation

/// クラウドへの 1 種類の依頼。
struct CloudTask {
    let prompt: String
    let schemaJson: String
    let imageResolution: String
}

enum CloudFailure: String {
    case auth = "AUTH", rateLimit = "RATE_LIMIT", network = "NETWORK", server = "SERVER", badResponse = "BAD_RESPONSE"

    var messageJa: String {
        switch self {
        case .auth: return "API キーを確認してください"
        case .rateLimit: return "利用上限に達しました。しばらく待ちます"
        case .network: return "通信できません"
        case .server: return "クラウド側のエラーです"
        case .badResponse: return "判別結果を読み取れませんでした"
        }
    }
}

struct CloudError: Error, CustomStringConvertible {
    let failure: CloudFailure
    let detail: String
    var description: String { "\(failure.rawValue): \(detail)" }
}

struct HttpResult {
    let code: Int
    let body: String
}

/// HTTP の POST。通信を待って止まるので、メインスレッドや AR のキューでは呼ばない。
enum Http {
    /// 立ち止まっている間に答えが欲しいので、待ち続けずに次のモデルへ切り替える。
    static let timeout: TimeInterval = 12

    static func post(_ url: String, headers: [String: String], body: Data, timeout: TimeInterval = Http.timeout) throws -> HttpResult {
        guard let u = URL(string: url) else { throw URLError(.badURL) }
        var request = URLRequest(url: u, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: timeout)
        request.httpMethod = "POST"
        for (k, v) in headers { request.setValue(v, forHTTPHeaderField: k) }
        request.httpBody = body
        let done = DispatchSemaphore(value: 0)
        var result: Result<HttpResult, Error> = .failure(URLError(.unknown))
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                result = .failure(error)
            } else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                result = .success(HttpResult(code: code, body: String(decoding: data ?? Data(), as: UTF8.self)))
            }
            done.signal()
        }.resume()
        done.wait()
        return try result.get()
    }
}

/// 指示（と画像）を渡して、スキーマに沿った JSON 文字列を受け取る。jpeg が nil なら文字だけで頼む。
protocol CloudVisionClient {
    func generate(_ task: CloudTask, _ jpeg: Data?) throws -> String
}

/**
 * Gemini の Interactions API を HTTP で直接呼ぶ。モデルは混雑で頻繁に 503 を返すので、失敗したら次のモデルに切り替える。
 */
final class GeminiClient: CloudVisionClient {
    private static let endpoint = "https://generativelanguage.googleapis.com/v1beta/interactions"
    private static let apiRevision = "2026-05-20"
    private static let maxErrorChars = 500
    static let retryDelays: [TimeInterval] = [0.3, 0.3, 1.0, 3.0]

    private let apiKey: String
    private let models: [String]
    private let retryDelays: [TimeInterval]
    private let timeout: TimeInterval

    /// models は試す順に並べたモデル。retryDelays が空ならやり直さない（すぐに答えが欲しい棚の言い直しなど）。
    init(apiKey: String, models: [String], retryDelays: [TimeInterval] = GeminiClient.retryDelays, timeout: TimeInterval = Http.timeout) {
        self.apiKey = apiKey
        self.models = models
        self.retryDelays = retryDelays
        self.timeout = timeout
    }

    func generate(_ task: CloudTask, _ jpeg: Data?) throws -> String {
        let headers = ["x-goog-api-key": apiKey, "Content-Type": "application/json", "Api-Revision": Self.apiRevision]
        var attempt = 0
        while true {
            let model = models[attempt % models.count]
            do {
                let response = try Http.post(Self.endpoint, headers: headers, body: requestBody(task, jpeg, model), timeout: timeout)
                return try handle(response)
            } catch let e as CloudError {
                let retryable = e.failure == .rateLimit || e.failure == .server
                if !retryable || attempt >= retryDelays.count { throw e }
            } catch {
                if attempt >= retryDelays.count { throw CloudError(failure: .network, detail: "request failed: \(error)") }
            }
            Thread.sleep(forTimeInterval: retryDelays[attempt])
            attempt += 1
        }
    }

    private func requestBody(_ task: CloudTask, _ jpeg: Data?, _ model: String) -> Data {
        let schema: Any = (try? JSONSerialization.jsonObject(with: Data(task.schemaJson.utf8))) ?? [String: Any]()
        var input: [[String: String]] = [["type": "text", "text": task.prompt]]
        if let jpeg {
            input.append([
                "type": "image",
                "data": jpeg.base64EncodedString(),
                "mime_type": "image/jpeg",
                "resolution": task.imageResolution,
            ])
        }
        let format: [String: Any] = ["type": "text", "mime_type": "application/json", "schema": schema]
        let body: [String: Any] = [
            "model": model,
            // 店内の写真には来店者が写ることがあるので、応答を返したら保存させない
            "store": false,
            "input": input,
            "response_format": format,
        ]
        return JSON.serialize(body)
    }

    private func handle(_ r: HttpResult) throws -> String {
        if r.code == 401 || r.code == 403 { throw CloudError(failure: .auth, detail: "HTTP \(r.code)") }
        if r.code == 429 { throw CloudError(failure: .rateLimit, detail: "HTTP 429") }
        if r.code >= 500 { throw CloudError(failure: .server, detail: "HTTP \(r.code)") }
        if !(200...299).contains(r.code) {
            throw CloudError(failure: .badResponse, detail: "HTTP \(r.code): \(r.body.prefix(Self.maxErrorChars))")
        }
        return try extractText(r.body)
    }

    /// 応答の steps から、モデルの出力の文章を取り出す。思考のステップは読み飛ばす。
    private func extractText(_ body: String) throws -> String {
        guard let root = JSON.object(body) else { throw CloudError(failure: .badResponse, detail: "not JSON") }
        let status = JSON.string(root, "status")
        if status != "completed" {
            throw CloudError(failure: .badResponse, detail: "status=\(status) errors=\(String(describing: root["errors"]))")
        }
        var text = ""
        for step in JSON.objects(root, "steps") where JSON.string(step, "type") == "model_output" {
            for part in JSON.objects(step, "content") where JSON.string(part, "type") == "text" {
                text += JSON.string(part, "text")
            }
        }
        if text.isEmpty { throw CloudError(failure: .badResponse, detail: "no text in response") }
        return text
    }
}
