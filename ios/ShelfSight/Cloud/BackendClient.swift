import Foundation

/// 店のモデルのバックエンド（backend/app.py）。通信を待つのでメインスレッドや AR のキューでは呼ばない。失敗したら nil か false を返す。
final class BackendClient {
    struct SavedMap {
        let name: String
        let view: StoreView?
    }

    private let baseUrl: String
    private let apiKey: String

    init(baseUrl: String, apiKey: String) {
        self.baseUrl = baseUrl
        self.apiKey = apiKey
    }

    var configured: Bool { !baseUrl.isEmpty && !apiKey.isEmpty }

    /// 観測を送る。サーバが OR-Tools で解いて店の地図として保存する。body は StoreModel.requestBody()。
    @discardableResult
    func upload(lat: Double, lng: Double, body: [String: Any]) -> Bool {
        var request = body
        request["lat"] = lat
        request["lng"] = lng
        return post("/v1/maps", request)?["saved"] as? Bool ?? false
    }

    func lookup(lat: Double, lng: Double) -> SavedMap? {
        guard let r = post("/v1/maps/lookup", ["lat": lat, "lng": lng]), let name = r["name"] as? String else { return nil }
        return SavedMap(name: name, view: (r["model"] as? [String: Any]).flatMap(Self.parseView))
    }

    private func post(_ path: String, _ body: [String: Any]) -> [String: Any]? {
        // 不正な値（NaN など）を渡すと JSONSerialization は例外で落ちる
        guard JSONSerialization.isValidJSONObject(body) else {
            logInfo("Backend", "\(path): 送れない値が入っています")
            return nil
        }
        let url = (baseUrl.hasSuffix("/") ? String(baseUrl.dropLast()) : baseUrl) + path
        do {
            let r = try Http.post(url, headers: ["X-Api-Key": apiKey, "Content-Type": "application/json"], body: JSON.serialize(body))
            guard (200...299).contains(r.code) else {
                logInfo("Backend", "\(path) failed: HTTP \(r.code)")
                return nil
            }
            return JSON.object(r.body)
        } catch {
            logInfo("Backend", "\(path) unreachable: \(error)")
            return nil
        }
    }

    static func parseView(_ m: [String: Any]) -> StoreView? {
        guard let yaw = JSON.double(m, "yaw") else { return nil }
        let room = (m["room"] as? [NSNumber])?.map(\.doubleValue)
        return StoreView(yaw, room?.count == 4 ? room : nil, JSON.objects(m, "fixtures").compactMap(parseFixture))
    }

    private static func parseFixture(_ f: [String: Any]) -> StoreFixture? {
        guard let kind = f["kind"] as? String, let axis = JSON.int(f, "axis"), let c = JSON.double(f, "c"), let a = JSON.double(f, "a"), let b = JSON.double(f, "b"),
              let d = JSON.double(f, "d"), let h = JSON.double(f, "h"), let n = JSON.int(f, "n"), n > 0, let w = JSON.double(f, "w") else { return nil }
        var faces: [String: [[(String, Double)]]] = [:]
        for (side, value) in (f["faces"] as? [String: [[[Any]]]]) ?? [:] {
            faces[side] = value.map { module in
                module.compactMap { pair in
                    guard pair.count == 2, let name = pair[0] as? String, let share = (pair[1] as? NSNumber)?.doubleValue else { return nil }
                    return (name, share)
                }
            }
        }
        var themes: [String: [String?]] = [:]
        for (side, value) in (f["themes"] as? [String: [Any]]) ?? [:] { themes[side] = value.map { $0 as? String } }
        let widths = (f["widths"] as? [NSNumber])?.map(\.doubleValue)
        return StoreFixture(kind, axis, c, a, b, d, h, n, w, faces, themes, widths?.count == n ? widths : nil)
    }
}
