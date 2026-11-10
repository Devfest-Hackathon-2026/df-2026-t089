import Foundation

/// 店のモデルから見つけた、近くの棚の面 1 つ。
struct NearShelf {
    let main: String
    let at: Vec3
    let distance: Float
    let azimuthDeg: Float
    let facing: Bool
}

/// 商品名を調べた記録。at は調べたときにいた場所。
struct LookedUp {
    let name: String
    let at: Vec3
    let time: Int64
}

/// 1 回たたいたときの答え。
enum TapAnswer {
    case sentence(String)
    case around(clauses: [String], shelves: [NearShelf])
    case unknown
}

/// タップで話す文と、質問に答える AI に渡す文脈を、店のモデルと今の位置から作る（Android 版の StoreGuide）。
enum StoreGuide {
    /// ここまでの棚を話す（商品の位置が分かる距離と同じ）。
    static let rangeM = ShelfMap.maxDepthM
    /// 棚に「向かっている」: この距離以内で、棚の面と向き合う角度（面の法線との角度が facingDeg 以内）。
    static let facingM: Float = 1.8
    static let facingDeg: Float = 40
    private static let maxShelves = 4
    /// 棚の商品として見る範囲（正面からの角度と距離）。
    private static let frontDeg: Float = 35
    private static let frontM: Float = 2.2
    /// モデルがまだないときに、商品だけで棚に向かっていると言える範囲（通路を歩いているだけの横の棚を含めない）。
    private static let itemsOnlyDeg: Float = 20
    private static let itemsOnlyM: Float = 1.5
    private static let kindsPerBand = 3
    /// 床からの高さで分ける（棚は 1.8 m ほど）。
    private static let upperM: Float = 1.25
    private static let lowerM: Float = 0.75
    private static let maxMapLines = 30
    private static let maxItemLines = 60
    private static let maxLookedLines = 10
    private static let kindNames = ["glass": "ガラスのケース", "island": "島の棚", "wall": "壁ぎわの棚", "case": "冷蔵・冷凍ケース", "end": "島の端の棚", "register": "レジ"]

    private static func degrees(_ r: Float) -> Float { r * 180 / .pi }

    /// 右が正の方位角（度）を、話す方向にする。
    static func direction(_ azimuthDeg: Float) -> String {
        let a = abs(azimuthDeg)
        let side = azimuthDeg > 0 ? "右" : "左"
        if a <= 30 { return "前" }
        if a <= 60 { return "\(side)前" }
        if a <= 120 { return side }
        if a <= 150 { return "\(side)後ろ" }
        return "後ろ"
    }

    /// 近い順の、こちらを向いている棚の面。主な商品が分かっている面だけ。
    static func nearShelves(_ view: StoreView, _ l: Listener) -> [NearShelf] {
        let cs = Float(cos(view.yaw)), sn = Float(sin(view.yaw))
        let u = l.position.x * cs + l.position.z * sn
        let v = -l.position.x * sn + l.position.z * cs
        var out: [NearShelf] = []
        for f in view.fixtures {
            for (sign, modules) in f.orderedFaces {
                let s: Float = sign == "+" ? 1 : -1
                let face = Float(f.c) + s * Float(f.d) / 2
                let cross = f.axis == 0 ? u : v
                let along = f.axis == 0 ? v : u
                if (cross - face) * s <= 0 { continue }   // 棚の裏側にいる
                func centerOf(_ i: Int) -> Float { Float(f.moduleCenter(i)) }
                guard let i = modules.indices.filter({ !modules[$0].isEmpty }).min(by: { abs(centerOf($0) - along) < abs(centerOf($1) - along) }) else { continue }
                let pu = f.axis == 0 ? face : centerOf(i)
                let pv = f.axis == 0 ? centerOf(i) : face
                let at = Vec3(pu * cs - pv * sn, l.position.y, pu * sn + pv * cs)
                let distance = hypot(at.x - l.position.x, at.z - l.position.z)
                if distance > rangeM { continue }
                let nu: Float = f.axis == 0 ? s : 0
                let nv: Float = f.axis == 0 ? 0 : s
                let nx = nu * cs - nv * sn
                let nz = nu * sn + nv * cs
                let facing = distance <= facingM && -(nx * l.forwardX + nz * l.forwardZ) >= cos(facingDeg * .pi / 180)
                out.append(NearShelf(main: modules[i][0].0, at: at, distance: distance, azimuthDeg: degrees(Binaural.azimuth(l, at)), facing: facing))
            }
        }
        return out.stableSorted { $0.distance < $1.distance }
    }

    /// 方向ごとにいちばん近い棚を、話す分だけ。
    static func toSpeak(_ shelves: [NearShelf]) -> [NearShelf] {
        var seen = Set<String>()
        return Array(shelves.filter { seen.insert(direction($0.azimuthDeg)).inserted }.prefix(maxShelves))
    }

    /// 「右にはカップ麺の棚、」「前にはお菓子の棚があります。」のように、1 棚ごとの文。チャイムを挟むので分ける。
    static func clauses(_ shelves: [NearShelf]) -> [String] {
        shelves.enumerated().map { i, s in
            "\(direction(s.azimuthDeg))には\(ShelfDescriber.spoken(s.main))の棚" + (i == shelves.count - 1 ? "があります。" : "、")
        }
    }

    private static func aimedAt(_ item: MapItem, _ l: Listener, _ maxDeg: Float, _ maxM: Float) -> Bool {
        hypot(item.position.x - l.position.x, item.position.z - l.position.z) <= maxM && abs(degrees(Binaural.azimuth(l, item.position))) <= maxDeg
    }

    /// 1 回たたいたときに話すことを決める。棚に向かっていればその棚の説明、近くに棚があれば方向ごとの棚。
    static func tapAnswer(_ view: StoreView?, _ items: [MapItem], _ l: Listener, _ groups: ShelfDescriber.Groups, floorY: Float) -> TapAnswer {
        let shelves = view.map { nearShelves($0, l) } ?? []
        let facing = shelves.first { $0.facing }
        // モデルができる前でも、正面の棚の商品が十分見えていれば答える
        let front: [MapItem]
        if facing != nil {
            front = items.filter { aimedAt($0, l, frontDeg, frontM) }
        } else if shelves.isEmpty {
            front = items.filter { aimedAt($0, l, itemsOnlyDeg, itemsOnlyM) }
        } else {
            front = []
        }
        if facing != nil || front.count >= ShelfDescriber.minProducts {
            let split = ShelfDescriber.split(front.map { RecognizedProduct(left: 0, top: 0, right: 0, bottom: 0, name: $0.name, genre: $0.genre, fine: $0.fine, sim: Float($0.seen)) }, groups)
            let kept = Set(split.kept.map(\.name))
            if let text = facingSentence(facing?.main, front.filter { kept.contains($0.name) }, floorY) ?? facing.flatMap({ facingSentence($0.main, [], 0) }) {
                return .sentence(text)
            }
        }
        if shelves.isEmpty { return .unknown }
        let near = toSpeak(shelves)
        return .around(clauses: clauses(near), shelves: near)
    }

    /// 「カップ麺の棚で、上の方にはA、真ん中にはBやC、下の方にはDがあります。」商品がなければ nil。
    static func facingSentence(_ main: String?, _ items: [MapItem], _ floorY: Float) -> String? {
        func height(_ m: MapItem) -> Float { m.position.y - floorY }
        let divided: [(String, [MapItem])] = [
            ("上の方", items.filter { height($0) >= upperM }),
            ("真ん中", items.filter { height($0) >= lowerM && height($0) < upperM }),
            ("下の方", items.filter { height($0) < lowerM }),
        ]
        let bands: [String] = divided.compactMap { band, list in
            let groups = list.orderedGroups { ShelfDescriber.spoken($0.name) }
            let kinds = groups.stableSorted { seen($0.values) > seen($1.values) }.prefix(kindsPerBand).map(\.key)
            return kinds.isEmpty ? nil : "\(band)には\(kinds.joined(separator: "や"))"
        }
        var top = main
        if top == nil {
            let groups = items.orderedGroups { $0.name }
            var best: (key: String, values: [MapItem])?
            for g in groups where best == nil || seen(g.values) > seen(best!.values) { best = g }
            top = best?.key
        }
        guard let top else { return nil }
        let shelf = "\(ShelfDescriber.spoken(top))の棚"
        return bands.isEmpty ? "\(shelf)です。" : "\(shelf)で、\(bands.joined(separator: "、"))があります。"
    }

    private static func seen(_ items: [MapItem]) -> Int { items.reduce(0) { $0 + $1.seen } }

    private static func whereIs(_ at: Vec3, _ l: Listener) -> String {
        let d = hypot(at.x - l.position.x, at.z - l.position.z)
        return "\(direction(degrees(Binaural.azimuth(l, at)))) \(String(format: "%.1f", d))m"
    }

    /// 質問に答える AI に渡す、今分かっていること。方向と距離は今の位置と向きから見たもの。
    static func context(_ view: StoreView?, _ l: Listener?, _ items: [MapItem], _ looked: [LookedUp], _ now: Int64, _ lastSpoken: String) -> String {
        var s = ""
        if let l {
            s += "- 方向と距離は、すべて利用者の今の位置と向きから見たものです。\n"
            let cs = view.map { cos($0.yaw) } ?? 1, sn = view.map { sin($0.yaw) } ?? 0
            if let r = view?.room { s += "- 店の広さ: 約\(String(format: "%.0f", r[2] - r[0]))m × \(String(format: "%.0f", r[3] - r[1]))m（歩いた範囲から）\n" }
            let shelves = (view?.fixtures ?? []).map { f -> (Float, Vec3, String) in
                let cu = f.axis == 0 ? f.c : (f.a + f.b) / 2
                let cv = f.axis == 0 ? (f.a + f.b) / 2 : f.c
                let at = Vec3(Float(cu * cs - cv * sn), l.position.y, Float(cu * sn + cv * cs))
                let goods = f.faces.values.flatMap { $0 }.flatMap { $0 }.orderedGroups { $0.0 }
                    .stableSorted { a, b in a.values.reduce(0) { $0 + $1.1 } > b.values.reduce(0) { $0 + $1.1 } }
                    .prefix(4).map(\.key).joined(separator: "、")
                var seenTheme = Set<String>()
                let themes = f.themes.values.flatMap { $0 }.compactMap { $0 }.filter { seenTheme.insert($0).inserted }.joined(separator: "、")
                let label = (kindNames[f.kind] ?? f.kind) + (themes.isEmpty ? "" : "（テーマ: \(themes)）")
                return (hypot(at.x - l.position.x, at.z - l.position.z), at, label + (goods.isEmpty ? "" : ": \(goods)"))
            }.stableSorted { $0.0 < $1.0 }
            if !shelves.isEmpty {
                s += "- 店の地図（近い順）:\n"
                for (_, at, text) in shelves.prefix(maxMapLines) { s += "  - \(whereIs(at, l)): \(text)\n" }
            }
            if !items.isEmpty {
                s += "- 見つけた商品の種類（いちばん近いものの場所、近い順）:\n"
                func distance(_ m: MapItem) -> Float { hypot(m.position.x - l.position.x, m.position.z - l.position.z) }
                let nearest = items.orderedGroups { $0.name }.map { g in (g.key, g.values.min { distance($0) < distance($1) }!) }
                    .stableSorted { distance($0.1) < distance($1.1) }
                for (name, m) in nearest.prefix(maxItemLines) { s += "  - \(name): \(whereIs(m.position, l))\n" }
            }
            if !looked.isEmpty {
                s += "- 利用者が商品名を調べた商品（新しい順、場所は調べたときにいた所）:\n"
                for x in looked.reversed().prefix(maxLookedLines) { s += "  - \(x.name): \(whereIs(x.at, l))、\((now - x.time) / 60000) 分前\n" }
            }
        } else {
            s += "- 位置はまだ分かりません。\n"
        }
        if !lastSpoken.isEmpty { s += "- 最後に利用者へ話したこと: \(lastSpoken)\n" }
        return s
    }
}

/// ガラス越しで奥行きを誤った点を、証拠のあるガラス面へ視線に沿って戻す。
enum GlassProjection {
    static func corrected(_ item: MapItem, _ view: StoreView?) -> Vec3? {
        guard let camera = item.cameraAt, let store = view else { return nil }
        let cs = cos(store.yaw), sn = sin(store.yaw)
        func local(_ p: Vec3) -> [Double] { [Double(p.x) * cs + Double(p.z) * sn, -Double(p.x) * sn + Double(p.z) * cs] }
        let origin = local(camera), raw = local(item.position)
        var best: (Double, Vec3)?
        for f in store.fixtures where f.kind == "glass" && (f.faces["+"] == nil) != (f.faces["-"] == nil) {
            let side = f.faces["+"] != nil ? 1 : -1
            let ax = f.axis, along = 1 - ax
            let face = f.c + Double(side) * f.d / 2
            guard let at = StoreModel.restoredAlong(origin[ax], origin[along], raw[ax], raw[along], face, side), at >= f.a, at <= f.b else { continue }
            let t = (face - origin[ax]) / (raw[ax] - origin[ax])
            let u = origin[0] + t * (raw[0] - origin[0]), v = origin[1] + t * (raw[1] - origin[1])
            let p = Vec3(Float(u * cs - v * sn), camera.y + Float(t) * (item.position.y - camera.y), Float(u * sn + v * cs))
            if best == nil || t < best!.0 { best = (t, p) }
        }
        return best?.1
    }
}

/// 棚の上に出すジャンル。u, v は店の座標、top は棚の高さ。
struct ShelfLabel {
    let u: Double, v: Double, top: Double
    let text: String
}

enum StoreLabels {
    /// 面ごとに、同じジャンルが続くモジュールを 1 つにまとめる。テーマが決まらないモジュールは一番多いカテゴリの名前にする。
    static func shelves(_ view: StoreView) -> [ShelfLabel] {
        var out: [ShelfLabel] = []
        for f in view.fixtures {
            for (side, modules) in f.orderedFaces where !modules.isEmpty {
                // 両面の棚は、それぞれの面の側に寄せて出す
                let across = f.c + (f.faces.count == 2 ? (side == "+" ? f.d / 4 : -f.d / 4) : 0)
                let names: [String?] = modules.indices.map { i in
                    if let t = f.themes[side], i < t.count, let name = t[i] { return name }
                    return modules[i].first?.0
                }
                var first = 0
                for i in 1...names.count {
                    if i < names.count && names[i] == names[first] { continue }
                    if let name = names[first] {
                        let along = (f.moduleStart(first) + f.moduleEnd(i - 1)) / 2
                        out.append(f.axis == 0 ? ShelfLabel(u: across, v: along, top: f.h, text: name) : ShelfLabel(u: along, v: across, top: f.h, text: name))
                    }
                    first = i
                }
            }
        }
        return out
    }
}
