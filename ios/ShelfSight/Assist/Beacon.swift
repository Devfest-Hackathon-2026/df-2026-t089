import Foundation

/**
 * 「〇〇はどこ？」と聞かれた場所から、立体音響のジングルを intervalMs ごとに鳴らして方向を伝える（Android 版の Beacon）。
 * 利用者が arriveM 以内（水平の距離）に着いたら止める。毎フレーム update を呼ぶ。
 */
final class Beacon {
    enum Event { case play, arrived }

    static let intervalMs: Int64 = 3000
    static let arriveM: Float = 1.0

    let name: String
    let at: Vec3
    private var lastPlay: Int64?

    init(name: String, at: Vec3) {
        self.name = name
        self.at = at
    }

    func update(_ l: Listener, now: Int64) -> Event? {
        if hypot(at.x - l.position.x, at.z - l.position.z) <= Self.arriveM { return .arrived }
        if let last = lastPlay, now - last < Self.intervalMs { return nil }
        lastPlay = now
        return .play
    }

    private static let wherePattern = try! NSRegularExpression(pattern: "^(.+?)(?:は|って|が|の場所は)?(?:どこ|どっち|どのへん|どの辺)")

    /// 「〇〇はどこ」の〇〇。場所を聞く質問でなければ nil。
    static func whereQuery(_ text: String) -> String? {
        var t = text.replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "　", with: "")
        while let last = t.last, "？?。".contains(last) { t.removeLast() }
        let ns = t as NSString
        guard let m = wherePattern.firstMatch(in: t, range: NSRange(location: 0, length: ns.length)) else { return nil }
        var q = ns.substring(with: m.range(at: 1))
        for suffix in ["の場所", "売り場", "の棚"] where q.hasSuffix(suffix) { q.removeLast(suffix.count) }
        return q.count >= 2 ? q : nil
    }

    /// 地図の中で、〇〇にいちばん合う場所。商品（名前・細かい種類・ジャンル・話す名前）が合えばその中でいちばん近いもの、
    /// なければ主な商品が合う棚の面。見つからなければ nil。
    static func find(_ query: String, _ items: [MapItem], _ view: StoreView?, _ l: Listener) -> (name: String, at: Vec3)? {
        func matches(_ key: String?) -> Bool {
            guard let key, key.count >= 2 else { return false }
            return key.contains(query) || query.contains(key)
        }
        func distance(_ p: Vec3) -> Float { hypot(p.x - l.position.x, p.z - l.position.z) }
        if let hit = items.filter({ [$0.name, $0.fine, $0.genre, ShelfDescriber.spoken($0.name)].contains(where: matches) })
            .min(by: { distance($0.position) < distance($1.position) }) {
            return (ShelfDescriber.spoken(hit.name), hit.position)
        }
        guard let view else { return nil }
        let cs = Float(cos(view.yaw)), sn = Float(sin(view.yaw))
        var best: (name: String, at: Vec3)?
        for f in view.fixtures {
            for (sign, modules) in f.orderedFaces {
                let s: Float = sign == "+" ? 1 : -1
                let face = Float(f.c) + s * Float(f.d) / 2
                for (i, goods) in modules.enumerated() {
                    guard let hit = goods.first(where: { matches($0.0) || matches(ShelfDescriber.spoken($0.0)) }) else { continue }
                    let along = Float(f.moduleCenter(i))
                    let pu = f.axis == 0 ? face : along
                    let pv = f.axis == 0 ? along : face
                    let at = Vec3(pu * cs - pv * sn, l.position.y, pu * sn + pv * cs)
                    if best == nil || distance(at) < distance(best!.at) { best = (ShelfDescriber.spoken(hit.0), at) }
                }
            }
        }
        return best
    }
}
