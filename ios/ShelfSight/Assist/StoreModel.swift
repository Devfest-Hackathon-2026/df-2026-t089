import Foundation

/// 什器 1 つ（記述言語の 1 行）。axis は正面が向く軸（0 = u, 1 = v）。faces は "+" / "-" の面ごと・モジュールごとの主な商品。
final class StoreFixture {
    let kind: String
    let axis: Int
    let c: Double, a: Double, b: Double, d: Double, h: Double
    let n: Int
    let w: Double
    let faces: [String: [[(String, Double)]]]
    /// 面ごと・モジュールごとのテーマ（決められなければ nil）。
    let themes: [String: [String?]]
    let moduleWidths: [Double]

    init(_ kind: String, _ axis: Int, _ c: Double, _ a: Double, _ b: Double, _ d: Double, _ h: Double, _ n: Int, _ w: Double,
         _ faces: [String: [[(String, Double)]]], _ themes: [String: [String?]] = [:], _ moduleWidths: [Double]? = nil) {
        self.kind = kind; self.axis = axis; self.c = c; self.a = a; self.b = b; self.d = d; self.h = h; self.n = n; self.w = w
        self.faces = faces; self.themes = themes
        self.moduleWidths = moduleWidths ?? [Double](repeating: w, count: n)
    }

    func moduleStart(_ i: Int) -> Double { a + moduleWidths.prefix(i).reduce(0, +) }
    func moduleEnd(_ i: Int) -> Double { moduleStart(i) + moduleWidths[i] }
    func moduleCenter(_ i: Int) -> Double { (moduleStart(i) + moduleEnd(i)) / 2 }

    /// "+" の面、"-" の面の順。
    var orderedFaces: [(side: String, modules: [[(String, Double)]])] {
        ["+", "-"].compactMap { s in faces[s].map { (s, $0) } }
    }
}

/// 推定した店。room は (u0, v0, u1, v1)。yaw は世界 → 店の座標の回転（ラジアン）。
final class StoreView {
    let yaw: Double
    let room: [Double]?
    let fixtures: [StoreFixture]
    /// 点に multi-model RANSAC で当てた線分 (x0, z0, x1, z1)。世界座標。
    let ransac: [[Double]]
    /// ガラスと判断して動かした点の (元 x, 元 z, 新 x, 新 z)。世界座標。
    let mirror: [[Double]]

    init(_ yaw: Double, _ room: [Double]?, _ fixtures: [StoreFixture], ransac: [[Double]] = [], mirror: [[Double]] = []) {
        self.yaw = yaw; self.room = room; self.fixtures = fixtures; self.ransac = ransac; self.mirror = mirror
    }
}

/// ゴンドラのテーマの定型 1 つ（store_themes.json）。
struct Theme {
    let name: String
    let cats: Set<String>
    let genres: Set<String>
}

struct IntPair: Hashable {
    let a: Int, b: Int
    init(_ a: Int, _ b: Int) { self.a = a; self.b = b }
}

/**
 * コンビニ記述言語のモデルを、「商品＋位置」の連続した観測から推定する（Android 版の StoreModel.kt、products/store_model.py の移植）。
 * 同じ入力で text() が一致することを ShelfSightTests で確かめる。1 つのキューから使う。
 */
final class StoreModel {
    private final class Cell {
        var w = 0.0, sx = 0.0, sz = 0.0, nx = 0.0, nz = 0.0
        var cats: [String: Double] = [:]
        var cameras: [[Double]] = []
    }

    private final class Pt: Hashable {
        let p: [Double], w: Double, nu: Double, nv: Double
        let cats: [String: Double]
        let cameras: [[Double]]
        init(_ p: [Double], _ w: Double, _ nu: Double, _ nv: Double, _ cats: [String: Double], _ cameras: [[Double]]) {
            self.p = p; self.w = w; self.nu = nu; self.nv = nv; self.cats = cats; self.cameras = cameras
        }
        static func == (a: Pt, b: Pt) -> Bool { a === b }
        func hash(into h: inout Hasher) { h.combine(ObjectIdentifier(self)) }
    }

    private final class Run: Hashable {
        let ax: Int, s: Int
        var c: Double, a: Double, b: Double, w: Double
        var pts: [Pt]
        var corner: Bool
        init(_ ax: Int, _ c: Double, _ a: Double, _ b: Double, _ s: Int, _ w: Double, _ pts: [Pt], _ corner: Bool = false) {
            self.ax = ax; self.c = c; self.a = a; self.b = b; self.s = s; self.w = w; self.pts = pts; self.corner = corner
        }
        static func == (a: Run, b: Run) -> Bool { a === b }
        func hash(into h: inout Hasher) { h.combine(ObjectIdentifier(self)) }
    }

    private final class Draft {
        let kind: String, axis: Int, c: Double, a: Double, b: Double, d: Double
        let faces: [Run]
        init(_ kind: String, _ axis: Int, _ c: Double, _ a: Double, _ b: Double, _ d: Double, _ faces: [Run]) {
            self.kind = kind; self.axis = axis; self.c = c; self.a = a; self.b = b; self.d = d; self.faces = faces
        }
    }

    private struct Pairing { let front: Run, back: Run, score: Double }
    private struct Rect { let u0: Double, v0: Double, u1: Double, v1: Double }

    private final class Candidate {
        var draft: Draft
        var fixture: StoreFixture
        var explains: Set<Pt>
        let bonus: Double
        let through: Set<Pt>
        init(_ draft: Draft, _ fixture: StoreFixture, _ explains: Set<Pt>, _ bonus: Double, _ through: Set<Pt> = []) {
            self.draft = draft; self.fixture = fixture; self.explains = explains; self.bonus = bonus; self.through = through
        }
    }

    private let themes: [Theme]
    private var cellIndex: [IntPair: Cell] = [:]
    private var cells: [Cell] = []   // 入れた順（Kotlin の LinkedHashMap）
    private var counterSeen = Set<IntPair>()
    private var counters: [IntPair] = []
    private var trail: [[Double]] = []
    private var genre: [String: String] = [:]
    private var obs = 0
    /// 観測ごとの (カメラ x, カメラ z, 点 x, 点 z)。ガラスの補正は同じ視点の視線どうしを比べるので、格子にまとめる前の形で持つ。
    private var rays: [[Double]] = []
    /// バックエンドへ送る観測（カメラ x, z, 点 x, z と、カテゴリ、ジャンル）。
    private var uploads: [([Double], String, String)] = []
    private var yaw = 0.0
    private var yawAt = 0

    init(themes: [Theme] = []) { self.themes = themes }

    /// 商品 1 つの観測（世界座標）。category が空なら位置だけを使う。
    func addProduct(_ camX: Double, _ camZ: Double, _ x: Double, _ z: Double, _ category: String, _ genreName: String) {
        let d = hypot(camX - x, camZ - z)
        guard d.isFinite, d >= 1e-6 else { return }
        let k = key(x, z)
        let c: Cell
        if let found = cellIndex[k] { c = found } else {
            c = Cell()
            cellIndex[k] = c
            cells.append(c)
        }
        c.w += 1; c.sx += x; c.sz += z
        c.nx += (camX - x) / d
        c.nz += (camZ - z) / d
        if !c.cameras.contains(where: { hypot($0[0] - camX, $0[1] - camZ) < 0.35 }) {
            if c.cameras.count == 4 { c.cameras.removeFirst() }
            c.cameras.append([camX, camZ])
        }
        if !category.isEmpty {
            c.cats[category, default: 0] += 1
            genre[category] = genreName
        }
        obs += 1
        uploads.append(([camX, camZ, x, z], category, genreName))
        if uploads.count > Self.maxRays { uploads.removeFirst() }
        if d <= Self.maxDepthM { rays.append([camX, camZ, x, z]) }   // これより遠い深度は信頼できない
        if rays.count > Self.maxRays { rays.removeFirst() }
    }

    func addCamera(_ x: Double, _ z: Double) {
        guard x.isFinite, z.isFinite else { return }
        if let last = trail.last, hypot(last[0] - x, last[1] - z) < 0.3 { return }
        trail.append([x, z])
    }

    /// 腰の高さの水平面の上の点（レジの手がかり）。
    func addCounter(_ x: Double, _ z: Double) {
        guard x.isFinite, z.isFinite else { return }
        let k = key(x, z)
        if counterSeen.insert(k).inserted { counters.append(k) }
    }

    private func key(_ x: Double, _ z: Double) -> IntPair { IntPair(Self.floorI(x / Self.cell), Self.floorI(z / Self.cell)) }

    private func fitYaw() {
        let pts = cells.map { ($0.sx / $0.w, $0.sz / $0.w, min($0.w, 5.0)) }
        func score(_ deg: Double) -> Double {
            let cs = cos(Self.radians(deg)), sn = sin(Self.radians(deg))
            var us: [Int: Double] = [:], vs: [Int: Double] = [:]
            for (x, z, w) in pts {
                us[Self.floorI((x * cs + z * sn) / 0.2), default: 0] += w
                vs[Self.floorI((-x * sn + z * cs) / 0.2), default: 0] += w
            }
            return us.values.reduce(0) { $0 + $1 * $1 } + vs.values.reduce(0) { $0 + $1 * $1 }
        }
        func best(_ cands: [Double]) -> Double {
            var top = cands[0], topScore = -1.0
            for d in cands {
                let s = score(d)
                if s > topScore { top = d; topScore = s }
            }
            return top
        }
        var prev = yaw * 180 / Double.pi
        if yawAt == 0 || obs >= yawAt * 2 {
            yawAt = obs
            prev = best((0..<30).map { Double($0) * 3.0 })
        }
        let d = best((-6...6).map { prev + 0.5 * Double($0) })
        yaw = Self.radians((d.truncatingRemainder(dividingBy: 90) + 90).truncatingRemainder(dividingBy: 90))
    }

    /// 歩いた線がこの線を横切る位置（面に沿った座標）。そこは通路で、棚はつながらない。
    private func walkedAcross(_ t: [[Double]], _ ax: Int, _ line: Double) -> [Double] {
        var out: [Double] = []
        if t.count >= 2 {
            for i in 0..<(t.count - 1) {
                let p = t[i], q = t[i + 1]
                if (p[ax] - line) * (q[ax] - line) < 0 {
                    out.append(p[1 - ax] + (line - p[ax]) / (q[ax] - p[ax]) * (q[1 - ax] - p[1 - ax]))
                }
            }
        }
        return out.sorted()
    }

    private func piece(_ cuts: [Double], _ x: Double) -> Int { cuts.reduce(0) { $0 + ($1 <= x ? 1 : 0) } }

    /// 視線がこの線を通り抜けて、その先の点に届いた位置。棚は透けないので、複数の視線が通った場所に棚はない。
    private func sightCuts(_ pts: [Pt], _ ax: Int, _ line: Double) -> [Double] {
        var hits: [Int: Int] = [:]
        for p in pts {
            if abs(p.p[ax] - line) < Self.sightBeyond { continue }   // 棚の帯の中の点は、その棚の奥行きの誤差
            for cam in p.cameras where (cam[ax] - line) * (p.p[ax] - line) < 0 {
                let at = cam[1 - ax] + (line - cam[ax]) / (p.p[ax] - cam[ax]) * (p.p[1 - ax] - cam[1 - ax])
                hits[Self.floorI(at / 0.2), default: 0] += 1
            }
        }
        return hits.filter { $0.value >= Self.sightMin }.keys.map { (Double($0) + 0.5) * 0.2 }
    }

    /// この線の上で、棚がつながらない位置（歩いた場所と視線が通った場所）。
    private func openAt(_ pts: [Pt], _ frameTrail: [[Double]], _ ax: Int, _ line: Double) -> [Double] {
        (walkedAcross(frameTrail, ax, line) + sightCuts(pts, ax, line)).sorted()
    }

    /// 軸に平行な棚の面を、点の多い線から順に取り出す。線は歩いた線で区切り、面にならなかった点は直交する棚のために残す。
    private func runsOf(_ input: [Pt], _ frameTrail: [[Double]]) -> [Run] {
        var runs: [Run] = []
        var left = input
        var tried = Set<IntPair>()
        var cache: [IntPair: [Double]] = [:]
        func cuts(_ ax: Int, _ b: Int) -> [Double] {
            let k = IntPair(ax, b)
            if let c = cache[k] { return c }
            let c = openAt(input, frameTrail, ax, (Double(b) + 0.5) * 0.05)
            cache[k] = c
            return c
        }
        while left.count >= 5 {
            var score = 0, axis = 0, center = 0.0, line = 0
            for ax in 0...1 {
                var bins: [Int: Set<Int>] = [:]
                for p in left { bins[Self.floorI(p.p[ax] / 0.05), default: []].insert(Self.floorI(p.p[1 - ax] / 0.2)) }
                for b in bins.keys.sorted() {
                    if tried.contains(IntPair(ax, b)) { continue }
                    var pieces: [Int: Int] = [:]
                    let walked = cuts(ax, b)
                    for i in -3...3 {
                        for y in bins[b + i] ?? [] { pieces[piece(walked, (Double(y) + 0.5) * 0.2), default: 0] += 1 }
                    }
                    let s = pieces.values.max() ?? 0
                    if s > score { score = s; axis = ax; center = (Double(b) + 0.5) * 0.05; line = b }
                }
            }
            if score < 4 { break }
            tried.insert(IntPair(axis, line))
            let t = 1 - axis
            let ax = axis, mid = center
            let walked = cuts(axis, line)
            let band = left.filter { abs($0.p[ax] - mid) <= Self.band }.stableSorted { $0.p[t] < $1.p[t] }
            var taken = Set<Pt>()
            var seg: [Pt] = []
            for p in band.map({ Optional($0) }) + [nil] {
                if let last = seg.last, p == nil || p!.p[t] - last.p[t] > Self.gap || piece(walked, p!.p[t]) != piece(walked, last.p[t]) {
                    let a = seg[0].p[t], b = last.p[t]
                    let w = seg.reduce(0) { $0 + $1.w }
                    if b - a >= Self.minLen && w >= Self.minObs {
                        let sign = seg.reduce(0.0) { $0 + (ax == 0 ? $1.nu : $1.nv) } >= 0 ? 1 : -1
                        runs.append(Run(ax, Self.front(seg, ax, sign), a, b, sign, w, seg))
                        taken.formUnion(seg)
                    }
                    seg = []
                }
                if let p { seg.append(p) }
            }
            left = left.filter { !taken.contains($0) }
        }
        return runs
    }

    /// 同じ線の上で少しだけ途切れた面をつなぐ。間を歩いたか視線が通っていたら、通路なのでつながない。
    private func merge(_ runs: [Run], _ frameTrail: [[Double]], _ pts: [Pt]) -> [Run] {
        var out: [Run] = []
        let sorted = runs.stableSorted { ($0.ax, $0.s, $0.a) < ($1.ax, $1.s, $1.a) }
        for r in sorted {
            let hit = out.first { m in
                m.ax == r.ax && m.s == r.s && abs(m.c - r.c) < Self.sameFace && r.a - m.b < 1.0 &&
                    !openAt(pts, frameTrail, m.ax, (m.c + r.c) / 2).contains { $0 >= m.b && $0 <= r.a }
            }
            if let hit {
                hit.b = max(hit.b, r.b); hit.w += r.w; hit.pts += r.pts
                hit.c = Self.front(hit.pts, hit.ax, hit.s)
            } else {
                out.append(r)
            }
        }
        return out
    }

    private struct RunEnd: Hashable { let run: Run, low: Bool }

    /// 直交する 2 つの面の端が近ければ、そこは棚の角。両方の面の端を交点にそろえて、棚を角に当てる。
    private func snapCorners(_ runs: [Run]) {
        struct End { let d: Double, r: Run, rLow: Bool, q: Run, qLow: Bool }
        var ends: [End] = []
        for r in runs where r.ax == 0 {
            for q in runs where q.ax == 1 {
                for rLow in [true, false] {
                    for qLow in [true, false] {
                        let d = hypot((rLow ? r.a : r.b) - q.c, (qLow ? q.a : q.b) - r.c)
                        if d < Self.cornerR { ends.append(End(d: d, r: r, rLow: rLow, q: q, qLow: qLow)) }
                    }
                }
            }
        }
        var used = Set<RunEnd>()
        for e in ends.stableSorted(by: { $0.d < $1.d }) {
            if used.contains(RunEnd(run: e.r, low: e.rLow)) || used.contains(RunEnd(run: e.q, low: e.qLow)) { continue }
            let ra = e.rLow ? e.q.c : e.r.a, rb = e.rLow ? e.r.b : e.q.c
            let qa = e.qLow ? e.r.c : e.q.a, qb = e.qLow ? e.q.b : e.r.c
            if rb - ra < Self.minLen || qb - qa < Self.minLen { continue }
            used.insert(RunEnd(run: e.r, low: e.rLow))
            used.insert(RunEnd(run: e.q, low: e.qLow))
            e.r.a = ra; e.r.b = rb; e.q.a = qa; e.q.b = qb
            e.r.corner = true; e.q.corner = true
        }
    }

    /// 面の後ろを歩いたか、後ろに別の棚の面があれば、壁際の什器ではない。
    private func wallBehind(_ r: Run, _ runs: [Run], _ frameTrail: [[Double]]) -> Bool {
        func back(_ cross: Double) -> Double { (r.c - cross) * Double(r.s) }
        let t = 1 - r.ax
        if frameTrail.contains(where: { $0[t] >= r.a && $0[t] <= r.b && back($0[r.ax]) > 0.9 }) { return false }
        return !runs.contains { $0 !== r && $0.ax == r.ax && back($0.c) > 1.2 && Self.overlap(r.a, r.b, $0.a, $0.b) > 0.3 }
    }

    /// 向かい合う面を組にする。1 つの面で 2 つの島を説明しないよう、全体でいちばん良い組み合わせを選ぶ。
    private func islandPairs(_ runs: [Run]) -> [Pairing] {
        let fronts = runs.filter { $0.s == 1 && $0.b - $0.a >= 0.8 }
        let backs = runs.filter { $0.s == -1 && $0.b - $0.a >= 0.8 }
        func candidate(_ a: Run, _ b: Run) -> Pairing? {
            if a.ax != b.ax { return nil }
            let depth = a.c - b.c
            if depth < 0.5 || depth > 1.3 { return nil }
            let o = Self.overlap(a.a, a.b, b.a, b.b)
            let shorter = min(a.b - a.a, b.b - b.a)
            if o <= 0.3 * shorter { return nil }
            let score = o / shorter + (1.0 - abs(depth - Self.islandDepth) / Self.islandDepth) + min(a.w, b.w) / (a.w + b.w)
            return Pairing(front: a, back: b, score: score)
        }
        let all = fronts.flatMap { a in backs.compactMap { candidate(a, $0) } }
        if all.isEmpty { return [] }
        if backs.count > 18 {
            var usedA = Set<Run>(), usedB = Set<Run>()
            return all.stableSorted { $0.score > $1.score }.filter {
                if usedA.contains($0.front) || usedB.contains($0.back) { return false }
                usedA.insert($0.front); usedB.insert($0.back)
                return true
            }
        }
        var index: [Run: Int] = [:]
        for (i, b) in backs.enumerated() { index[b] = i }
        let byFront = fronts.map { a in all.filter { $0.front === a } }
        var memo: [IntPair: (Double, [Pairing])] = [:]
        func solve(_ i: Int, _ used: Int) -> (Double, [Pairing]) {
            if i == fronts.count { return (0, []) }
            let k = IntPair(i, used)
            if let m = memo[k] { return m }
            var best = solve(i + 1, used)
            for pair in byFront[i] {
                let bit = 1 << index[pair.back]!
                if used & bit != 0 { continue }
                let rest = solve(i + 1, used | bit)
                if pair.score + rest.0 > best.0 { best = (pair.score + rest.0, [pair] + rest.1) }
            }
            memo[k] = best
            return best
        }
        return solve(0, 0).1
    }

    private func classify(_ runs: [Run], _ frameTrail: [[Double]]) -> [Draft] {
        var out: [Draft] = [], islands: [Draft] = []
        var used = Set<Run>()
        for pair in islandPairs(runs) {
            let a = pair.front, b = pair.back
            used.insert(a); used.insert(b)
            islands.append(Draft("island", a.ax, (a.c + b.c) / 2, min(a.a, b.a), max(a.b, b.b), a.c - b.c, [a, b]))
        }
        func takeEnds() {
            for r in runs {
                if used.contains(r) || r.b - r.a > 1.5 { continue }
                let mid = (r.a + r.b) / 2
                for i in islands {
                    let atB = r.s == 1 && r.c >= i.b - 0.2 && r.c <= i.b + 0.8
                    let atA = r.s == -1 && r.c >= i.a - 0.8 && r.c <= i.a + 0.2
                    // 本当のエンドは、島の短い辺の両方の角に届く
                    let coversEdge = r.a <= i.c - i.d / 2 + 0.25 && r.b >= i.c + i.d / 2 - 0.25
                    if r.ax == 1 - i.axis && abs(mid - i.c) < 0.3 && coversEdge && (atA || atB) {
                        used.insert(r)
                        out.append(Draft("end", r.ax, r.c - Double(r.s) * Self.endDepth / 2, i.c - 0.45, i.c + 0.45, Self.endDepth, [r]))
                        break
                    }
                }
            }
        }
        takeEnds()
        for r in runs where !used.contains(r) && r.b - r.a >= 0.8 && !wallBehind(r, runs, frameTrail) {
            used.insert(r)
            islands.append(Draft("island", r.ax, r.c - Double(r.s) * Self.islandDepth / 2, r.a, r.b, Self.islandDepth, [r]))
        }
        takeEnds()
        for r in runs where !used.contains(r) {
            if wallBehind(r, runs, frameTrail) {
                let named = r.pts.flatMap { Array($0.cats) }
                let chilled = named.filter { genre[$0.key].map { Self.chilled.contains($0) } ?? false }.reduce(0) { $0 + $1.value }
                let isCase = !named.isEmpty && chilled >= named.reduce(0) { $0 + $1.value } / 2
                let depth = isCase ? Self.caseDepth : Self.gondolaDepth
                out.append(Draft(isCase ? "case" : "wall", r.ax, r.c - Double(r.s) * depth / 2, r.a, r.b, depth, [r]))
            } else {
                islands.append(Draft("island", r.ax, r.c - Double(r.s) * Self.islandDepth / 2, r.a, r.b, Self.islandDepth, [r]))
            }
        }
        return islands + out
    }

    private func inTheme(_ t: Theme, _ cat: String) -> Bool {
        t.cats.contains(cat) || (genre[cat].map { t.genres.contains($0) } ?? false)
    }

    /// モジュールごとに、観測に一番合うテーマを選ぶ（決められなければ nil）。
    private func assignThemes(_ mods: [[String: Double]]) -> [Theme?] {
        if themes.isEmpty { return [Theme?](repeating: nil, count: mods.count) }
        let n = Double(Set(themes.flatMap { $0.cats }).count)
        let scores: [[Double]] = mods.map { m in
            themes.map { t in
                var s = 0.0
                for (c, v) in m {
                    s += v * (inTheme(t, c) ? log(Self.themeIn / Double(t.cats.count) + (1 - Self.themeIn) / n) : log((1 - Self.themeIn) / n))
                }
                return s
            }
        }
        return mods.indices.map { i -> Theme? in
            let m = mods[i]
            let total = m.values.reduce(0, +)
            if total < Self.minThemeObs { return nil }
            let near = [i - 1, i + 1].filter { mods.indices.contains($0) }.map { scores[$0] }
            var best = 0
            var top = -Double.infinity
            for k in themes.indices {
                var sum = 0.0
                for x in near { sum += x[k] }
                let s = scores[i][k] + Self.themeNeighbor * sum
                if s > top { top = s; best = k }
            }
            let theme = themes[best]
            let share = m.filter { inTheme(theme, $0.key) }.values.reduce(0, +) / total
            return share >= Self.minThemeShare ? theme : nil
        }
    }

    /// 3 尺と 4 尺のモジュールの組み合わせを当てはめる。よく観測された列の端は保つ。
    private func moduleLayout(_ d: Draft) -> (start: Double, widths: [Double]) {
        let mid = (d.a + d.b) / 2
        if d.kind == "end" { return (mid - 0.45, [0.9]) }
        let measured = d.b - d.a
        var best3 = 0, best4 = 0, bestCost = Double.greatestFiniteMagnitude
        for n3 in 0...12 {
            for n4 in 0...12 {
                let n = n3 + n4
                if n < 1 || n > 12 { continue }
                let cost = abs(0.9 * Double(n3) + 1.2 * Double(n4) - measured) + Double(n) * 0.005
                if cost < bestCost { bestCost = cost; best3 = n3; best4 = n4 }
            }
        }
        let count = best3 + best4
        let prior = Double(best3) * 0.9 + Double(best4) * 1.2
        let observations = d.faces.reduce(0) { $0 + $1.w }
        let trust = d.faces.contains { $0.corner } ? 1.0 : (observations / 30.0).clamped(0, 1)   // 角に当たった端は測った位置を信じる
        let target = prior + (measured - prior) * trust
        let correction = (target - prior) / Double(count)
        let small = 0.9 + correction, large = 1.2 + correction
        let start = mid - target / 2
        let pts = d.faces.flatMap { $0.pts }
        var best: [Double] = []
        var bestEvidence = -Double.infinity
        func evaluate(_ widths: [Double]) {
            var cats = [[String: Double]](repeating: [:], count: widths.count)
            var at = start
            for (i, width) in widths.enumerated() {
                let end = at + width
                for p in pts where p.p[1 - d.axis] >= at && (p.p[1 - d.axis] < end || i == widths.count - 1) {
                    for (cat, weight) in p.cats { cats[i][cat, default: 0] += weight }
                }
                at = end
            }
            let evidence = cats.reduce(0) { $0 + ($1.values.max() ?? 0) }
            if evidence > bestEvidence + 1e-6 { bestEvidence = evidence; best = widths }
        }
        var widths: [Double] = []
        func enumerate(_ n3: Int, _ n4: Int) {
            if n3 + n4 == 0 { evaluate(widths); return }
            if n3 > 0 { widths.append(small); enumerate(n3 - 1, n4); widths.removeLast() }
            if n4 > 0 { widths.append(large); enumerate(n3, n4 - 1); widths.removeLast() }
        }
        enumerate(best3, best4)
        return (start, best)
    }

    private func build(_ d: Draft) -> StoreFixture {
        let layout = moduleLayout(d)
        let a = layout.start
        let widths = layout.widths
        let n = widths.count
        let w = widths.reduce(0, +) / Double(n)
        var edges = [Double](repeating: a, count: n + 1)
        for i in 0..<n { edges[i + 1] = edges[i] + widths[i] }
        var faces: [String: [[(String, Double)]]] = [:]
        var themeOf: [String: [String?]] = [:]
        for sign in [1, -1] {
            let sides = d.faces.filter { $0.s == sign }
            if sides.isEmpty { continue }
            var mods = [[String: Double]](repeating: [:], count: n)
            for r in sides {
                for p in r.pts {
                    let coord = p.p[1 - r.ax]
                    let index = (0..<(n - 1)).first { coord < edges[$0 + 1] } ?? n - 1
                    for (cat, v) in p.cats { mods[index][cat, default: 0] += v }
                }
            }
            let side = sign == 1 ? "+" : "-"
            let assigned = assignThemes(mods)
            themeOf[side] = assigned.map { $0?.name }
            faces[side] = mods.indices.map { i -> [(String, Double)] in
                let theme = assigned[i]
                let kept = mods[i].filter { theme == nil || inTheme(theme!, $0.key) }
                let sum = kept.values.reduce(0, +)
                let total = sum == 0 ? 1 : sum
                return kept.sorted { ($0.value, $1.key) > ($1.value, $0.key) }.prefix(3).map { ($0.key, $0.value / total) }
            }
        }
        return StoreFixture(d.kind, d.axis, d.c, a, edges[n], d.d, Self.height[d.kind]!, n, w, faces, themeOf, widths)
    }

    private func rect(_ f: StoreFixture) -> Rect {
        f.axis == 0 ? Rect(u0: f.c - f.d / 2, v0: f.a, u1: f.c + f.d / 2, v1: f.b) : Rect(u0: f.a, v0: f.c - f.d / 2, u1: f.b, v1: f.c + f.d / 2)
    }

    /// Liang-Barsky。p-q が r に入る・出るところのパラメータ。当たらなければ nil。
    private func clip(_ p: [Double], _ q: [Double], _ r: Rect, _ inset: Double) -> (Double, Double)? {
        let lo = [r.u0 + inset, r.v0 + inset], hi = [r.u1 - inset, r.v1 - inset]
        if lo[0] >= hi[0] || lo[1] >= hi[1] { return nil }
        var enter = 0.0, leave = 1.0
        for axis in 0...1 {
            let delta = q[axis] - p[axis]
            if abs(delta) < 1e-9 {
                if p[axis] <= lo[axis] || p[axis] >= hi[axis] { return nil }
            } else {
                let a = (lo[axis] - p[axis]) / delta, b = (hi[axis] - p[axis]) / delta
                enter = max(enter, min(a, b)); leave = min(leave, max(a, b))
                if enter >= leave { return nil }
            }
        }
        return (enter, leave)
    }

    private func crosses(_ p: [Double], _ q: [Double], _ r: Rect, _ inset: Double) -> Bool { clip(p, q, r, inset) != nil }

    private func cornerSupport(_ d: Draft, _ all: [Draft]) -> Int {
        func endpoints(_ r: Run) -> [[Double]] { r.ax == 0 ? [[r.c, r.a], [r.c, r.b]] : [[r.a, r.c], [r.b, r.c]] }
        let others = all.filter { $0 !== d }.flatMap { $0.faces }
        return d.faces.reduce(0) { sum, face in
            let ends = endpoints(face)
            return sum + others.filter { other in
                other.ax != face.ax && ends.contains { a in endpoints(other).contains { b in hypot(a[0] - b[0], a[1] - b[1]) < 0.35 } }
            }.count
        }
    }

    /// どの棚でも説明できない点の塊のうち、視線方向に大きくばらけたものはガラス越しの深度の失敗とみなし、塊のカメラ側の端のガラス面へ戻す。
    private func glassCandidates(_ rest: [Pt]) -> [Candidate] {
        var groups: [[Pt]] = []
        var left = rest
        while !left.isEmpty {
            var group: [Pt] = []
            var frontier = [left.removeFirst()]
            while let p = frontier.popLast() {
                group.append(p)
                frontier += left.filter { hypot(p.p[0] - $0.p[0], p.p[1] - $0.p[1]) <= Self.glassGap }
                left = left.filter { hypot(p.p[0] - $0.p[0], p.p[1] - $0.p[1]) > Self.glassGap }
            }
            groups.append(group)
        }
        var out: [Candidate] = []
        for group in groups {
            let views = group.flatMap { p in p.cameras.map { (p, $0) } }
            if group.count < Self.glassMin || views.isEmpty { continue }
            let sum0 = views.reduce(0) { $0 + abs($1.0.p[0] - $1.1[0]) }
            let sum1 = views.reduce(0) { $0 + abs($1.0.p[1] - $1.1[1]) }
            let ax = sum0 >= sum1 ? 0 : 1
            let along = 1 - ax
            let sign = views.reduce(0.0) { $0 + $1.1[ax] - $1.0.p[ax] } >= 0 ? 1 : -1
            let values = group.map { ($0.p[ax], $0.w) }
            if Self.quantile(values, 0.9) - Self.quantile(values, 0.1) < Self.glassSpread { continue }
            let face = Self.quantile(values, sign > 0 ? 0.9 : 0.1)
            let moved: [Pt] = group.compactMap { p in
                let hits = p.cameras.compactMap { Self.restoredAlong($0[ax], $0[along], p.p[ax], p.p[along], face, sign) }
                if hits.isEmpty { return nil }
                let at = hits.reduce(0, +) / Double(hits.count)
                return Pt(ax == 0 ? [face, at] : [at, face], p.w, p.nu, p.nv, p.cats, p.cameras)
            }
            if moved.count < Self.glassMin { continue }
            let a = moved.map { $0.p[along] }.min()!, b = moved.map { $0.p[along] }.max()!
            if b - a < Self.minLen { continue }
            let run = Run(ax, face, a, b, sign, moved.reduce(0) { $0 + $1.w }, moved)
            let draft = Draft("glass", ax, face - Double(sign) * Self.caseDepth / 2, a, b, Self.caseDepth, [run])
            out.append(Candidate(draft, build(draft), Set(group), Self.glassBonus))
        }
        return out
    }

    /// 歩いた場所をふさぐ棚、透ける棚、裏側からの観測、棚の芯にある点を弾く。
    private func valid(_ f: StoreFixture, _ d: Draft, _ pts: [Pt], _ frameTrail: [[Double]]) -> Bool {
        let r = rect(f)
        if frameTrail.contains(where: { crosses($0, $0, r, Self.trailMargin) }) ||
            zip(frameTrail, frameTrail.dropFirst()).contains(where: { crosses($0, $1, r, Self.trailMargin) }) { return false }
        var wrongSide = 0, views = 0
        for face in d.faces {
            for p in face.pts {
                for cam in p.cameras {
                    views += 1
                    if (cam[face.ax] - face.c) * Double(face.s) < -0.1 { wrongSide += 1 }
                }
            }
        }
        if views >= 4 && wrongSide * 4 > views { return false }
        let own = Set(d.faces.flatMap { $0.pts })
        if f.kind == "glass" { return true }   // ガラスの奥に点があるのは当然なので、下は調べない
        // 正面側のカメラから見えた点は、その棚の奥行きの誤差なので矛盾しない
        let core = pts.filter { p in
            !own.contains(p) && p.p[0] > r.u0 + 0.15 && p.p[0] < r.u1 - 0.15 && p.p[1] > r.v0 + 0.15 && p.p[1] < r.v1 - 0.15 &&
                !d.faces.contains { face in p.cameras.contains { ($0[face.ax] - face.c) * Double(face.s) > 0 } }
        }.count
        return core <= max(6, pts.count / 50)
    }

    /// 棚を長手方向に 10 cm ずつ見て、視線が突き抜けた区間を除いた、棚として残せる区間を返す。棚は透けないため。
    private func carve(_ d: Draft, _ f: StoreFixture, _ pts: [Pt]) -> [Draft] {
        let r = rect(f)
        let along = 1 - d.axis
        let lo = f.a, hi = f.b
        let n = max(1, Int((hi - lo) / 0.1 + 0.999))
        var cover = [Int](repeating: 0, count: n)
        for p in pts {
            if max(r.u0 - p.p[0], p.p[0] - r.u1, r.v0 - p.p[1], p.p[1] - r.v1) <= Self.explain { continue }   // この棚の点（奥行きの誤差も含む）
            for cam in p.cameras {
                guard let hit = clip(cam, p.p, r, 0.02) else { continue }
                let a0 = cam[along] + hit.0 * (p.p[along] - cam[along])
                let a1 = cam[along] + hit.1 * (p.p[along] - cam[along])
                for i in stride(from: max(0, Int((min(a0, a1) - lo) / 0.1)), through: min(n - 1, Int((max(a0, a1) - lo) / 0.1)), by: 1) { cover[i] += 1 }
            }
        }
        var pieces: [Draft] = []
        var i = 0
        while i < n {
            if cover[i] >= Self.sightMin { i += 1; continue }
            var j = i
            while j < n && cover[j] < Self.sightMin { j += 1 }
            let a = lo + Double(i) * 0.1, b = min(hi, lo + Double(j) * 0.1)
            // 削った端は測った位置のまま
            if b - a >= Self.minLen {
                pieces.append(Draft(d.kind, d.axis, d.c, a, b, d.d, d.faces.map { Run($0.ax, $0.c, $0.a, $0.b, $0.s, $0.w, $0.pts, true) }))
            }
            i = j
        }
        return pieces
    }

    /// カメラから棚の向こうの点へ、棚を突き抜けて届いた点。不透明な棚では起きない。
    private func seeThrough(_ f: StoreFixture, _ pts: [Pt]) -> Set<Pt> {
        let r = rect(f)
        let low = f.c - f.d / 2 - 0.2, high = f.c + f.d / 2 + 0.2
        return Set(pts.filter { p in
            p.cameras.contains { cam in
                ((cam[f.axis] < low && p.p[f.axis] > high) || (cam[f.axis] > high && p.p[f.axis] < low)) && crosses(cam, p.p, r, 0.1)
            }
        })
    }

    private func glassFront(_ f: StoreFixture, _ d: Draft) -> StoreFixture {
        let sign = Double(d.faces[0].s)
        let face = f.c + sign * f.d / 2
        return StoreFixture("glass", f.axis, face - sign * Self.caseDepth / 2, f.a, f.b, Self.caseDepth, Self.height["glass"]!, f.n, f.w, f.faces, f.themes, f.moduleWidths)
    }

    private func overlaps(_ f: StoreFixture, _ g: StoreFixture) -> Bool {
        let r = rect(f), q = rect(g)
        return min(r.u1, q.u1) - max(r.u0, q.u0) > Self.touch && min(r.v1, q.v1) - max(r.v0, q.v0) > Self.touch
    }

    /// 棚を長手方向に削って other に当たらないようにする。エンドは幅が決まっているので削らない。
    private func trim(_ c: Candidate, _ other: StoreFixture) -> Bool {
        let d = c.draft
        if d.kind == "end" { return false }
        let q = rect(other)
        let lo = d.axis == 1 ? q.u0 : q.v0, hi = d.axis == 1 ? q.u1 : q.v1
        let below = (d.a + d.b) / 2 < (lo + hi) / 2
        let a = below ? d.a : max(d.a, hi)
        let b = below ? min(d.b, lo) : d.b
        if b - a < Self.minLen || (a == d.a && b == d.b) { return false }
        c.draft = Draft(d.kind, d.axis, d.c, a, b, d.d, d.faces)
        c.fixture = build(c.draft)
        return true
    }

    /// エンドは島 1 つの端に付く。2 つの列をつなぐと通路がふさがる。
    private func endTouchesOneIsland(_ f: StoreFixture, _ accepted: [Candidate]) -> Bool {
        if f.kind != "end" { return true }
        let r = rect(f)
        return accepted.filter { other in
            if other.fixture.kind != "island" { return false }
            let q = rect(other.fixture)
            return r.u0 <= q.u1 + 0.15 && r.u1 >= q.u0 - 0.15 && r.v0 <= q.v1 + 0.15 && r.v1 >= q.v0 - 0.15
        }.count == 1
    }

    /// 棚は重なれないので、当たる棚を削る。削っても解けないなら置かない。
    private func place(_ c: Candidate, _ accepted: [Candidate]) -> Bool {
        for _ in 0..<4 {
            guard let other = accepted.first(where: { overlaps(c.fixture, $0.fixture) }) else { return endTouchesOneIsland(c.fixture, accepted) }
            // 既に置いた棚を削るのは、幅の決まったエンドを島の先端に付けるときだけ
            if !trim(c, other.fixture) && !(c.fixture.kind == "end" && trim(other, c.fixture)) { return false }
        }
        return false
    }

    /// 新しく説明できる点が棚のコストを上回る間だけ、いちばん得な棚を足す。点を説明できる最も少ない棚の組を選ぶため。
    private func choose(_ candidates: [Candidate], _ accepted: inout [Candidate]) {
        var explained = Set(accepted.flatMap { $0.explains })
        var left = candidates
        func gain(_ c: Candidate) -> Double {
            Double(c.explains.filter { !explained.contains($0) }.count) + c.bonus - Self.shelfCost - Self.moduleCost * Double(c.fixture.n)
        }
        while !left.isEmpty {
            var best = left[0]
            var bestKey = (gain(best), best.fixture.kind != "end" ? 1 : 0, -best.fixture.c)
            for c in left.dropFirst() {   // 同点は先のものを残す（Kotlin の maxWith）
                let key = (gain(c), c.fixture.kind != "end" ? 1 : 0, -c.fixture.c)
                if bestKey < key { best = c; bestKey = key }
            }
            left.removeAll { $0 === best }
            if gain(best) <= 0 { break }
            if place(best, accepted) {
                accepted.append(best)
                explained.formUnion(best.explains)
            }
        }
    }

    private func register(_ cs: Double, _ sn: Double) -> StoreFixture? {
        if counters.count < 6 { return nil }
        let centers = counters.map { ((Double($0.a) + 0.5) * Self.cell, (Double($0.b) + 0.5) * Self.cell) }
        let us = centers.map { $0.0 * cs + $0.1 * sn }.sorted()
        let vs = centers.map { -$0.0 * sn + $0.1 * cs }.sorted()
        let cut = us.count / 20
        let u0 = us[cut], u1 = us[us.count - 1 - cut], v0 = vs[cut], v1 = vs[vs.count - 1 - cut]
        let axis = u1 - u0 < v1 - v0 ? 0 : 1
        let c = axis == 0 ? (u0 + u1) / 2 : (v0 + v1) / 2
        let a = axis == 0 ? v0 : u0, b = axis == 0 ? v1 : u1
        return StoreFixture("register", axis, c, a, b, max(0.5, axis == 0 ? u1 - u0 : v1 - v0), Self.height["register"]!, 1, b - a, [:])
    }

    private func roomOf(_ fixtures: [StoreFixture], _ frameTrail: [[Double]]) -> [Double] {
        var low: [[Double]] = [[], []], high: [[Double]] = [[], []]
        for f in fixtures {
            low[f.axis].append(f.c - f.d / 2); high[f.axis].append(f.c + f.d / 2)
            low[1 - f.axis].append(f.a); high[1 - f.axis].append(f.b)
        }
        for p in frameTrail {
            for ax in 0...1 { low[ax].append(p[ax] - 0.5); high[ax].append(p[ax] + 0.5) }
        }
        return [low[0].min()!, low[1].min()!, high[0].max()!, high[1].max()!]
    }

    /// バックエンドの /v1/maps に送る本文（緯度経度は BackendClient が足す）。
    func requestBody() -> [String: Any] {
        [
            "observations": uploads.map { ["cam": [$0.0[0], $0.0[1]], "point": [$0.0[2], $0.0[3]], "category": $0.1, "genre": $0.2] as [String: Any] },
            "path": trail,
            "counters": counters.map { [(Double($0.a) + 0.5) * Self.cell, (Double($0.b) + 0.5) * Self.cell] },
        ]
    }

    /// 今の観測から店のモデルを作る。
    func estimate() -> StoreView {
        if cells.count < 30 { return StoreView(0, nil, []) }
        fitYaw()
        let cs = cos(yaw), sn = sin(yaw)
        let pts = cells.map { c -> Pt in
            let x = c.sx / c.w, z = c.sz / c.w
            return Pt([x * cs + z * sn, -x * sn + z * cs], c.w, c.nx * cs + c.nz * sn, -c.nx * sn + c.nz * cs, c.cats,
                      c.cameras.map { [$0[0] * cs + $0[1] * sn, -$0[0] * sn + $0[1] * cs] })
        }
        let frameTrail = trail.map { [$0[0] * cs + $0[1] * sn, -$0[0] * sn + $0[1] * cs] }
        let runs = merge(runsOf(pts, frameTrail), frameTrail, pts)
        snapCorners(runs)
        let drafts = classify(runs, frameTrail)
        var opaque: [Candidate] = [], doubtful: [Candidate] = []
        var pending = drafts
        func candidate(_ d: Draft, _ f: StoreFixture, _ through: Set<Pt> = []) -> Candidate {
            let lo = d.a - 0.1, hi = d.b + 0.1, along = 1 - d.axis
            let own: [Pt] = d.faces.flatMap { $0.pts }
            let kept = Set(own.filter { (p: Pt) -> Bool in p.p[along] >= lo && p.p[along] <= hi })
            let r = rect(f)
            let u0 = r.u0 - Self.explain, u1 = r.u1 + Self.explain, v0 = r.v0 - Self.explain, v1 = r.v1 + Self.explain
            let near = Set(pts.filter { (p: Pt) -> Bool in
                let u = p.p[0], v = p.p[1]
                return u >= u0 && u <= u1 && v >= v0 && v <= v1
            })
            return Candidate(d, f, near.union(kept), Self.cornerBonus * Double(cornerSupport(d, drafts)), through)
        }
        while !pending.isEmpty {
            let d = pending.removeFirst()
            let single = d.faces.count == 1 && d.kind != "end"
            // 面があれば、その裏に棚がある。歩いた線に当たるなら棚を薄くする。視線が通る区間は棚を削る（棚は透けない）
            var best: (Draft, StoreFixture, [Draft])?
            var bestLength = -1.0
            for scale in single ? Self.depthScales : [1.0] {
                let depth = d.d * scale
                let trial = Draft(d.kind, d.axis, single ? d.faces[0].c - Double(d.faces[0].s) * depth / 2 : d.c, d.a, d.b, depth, d.faces)
                let f = build(trial)
                if !valid(f, trial, pts, frameTrail) { continue }
                let pieces = carve(trial, f, pts)
                let length = pieces.reduce(0) { $0 + $1.b - $1.a }
                if best == nil || length > bestLength + 1e-6 { best = (trial, f, pieces); bestLength = length }
            }
            guard let chosen = best, !chosen.2.isEmpty else {
                if d.faces.count == 2 {   // 島として成り立たなくても、面ごとの棚は残す
                    for r in d.faces { pending.append(Draft("wall", r.ax, r.c - Double(r.s) * Self.gondolaDepth / 2, r.a, r.b, Self.gondolaDepth, [r])) }
                } else if let b = best, d.kind == "wall" || d.kind == "case" {   // 全長を視線に貫かれる面の説明は、壁際のガラスだけ
                    doubtful.append(candidate(b.0, b.1, seeThrough(b.1, pts)))
                }
                continue
            }
            for piece in chosen.2 { opaque.append(candidate(piece, build(piece))) }
        }
        var accepted: [Candidate] = []
        choose(opaque, &accepted)
        var explained = Set(accepted.flatMap { $0.explains })
        for c in doubtful {
            c.fixture = glassFront(c.fixture, c.draft)
            c.explains = c.explains.union(c.through.subtracting(explained))
        }
        choose(doubtful, &accepted)
        explained = Set(accepted.flatMap { $0.explains })
        choose(glassCandidates(pts.filter { !explained.contains($0) }).filter { valid($0.fixture, $0.draft, pts, frameTrail) }, &accepted)
        var fixtures: [StoreFixture] = []
        for c in accepted where !fixtures.contains(where: { overlaps(c.fixture, $0) }) { fixtures.append(c.fixture) }   // 削った棚はモジュール幅を合わせ直すので、最後にもう一度確かめる
        if let reg = register(cs, sn), !fixtures.contains(where: { overlaps(reg, $0) }) { fixtures.append(reg) }
        let sorted = fixtures.stableSorted { ($0.kind, $0.axis, $0.c, $0.a) < ($1.kind, $1.axis, $1.c, $1.a) }
        let mirror = MirrorCorrection.correct(rays)
        return StoreView(yaw, roomOf(sorted, frameTrail), sorted, ransac: MultiLineRansac.fit(MirrorCorrection.cellMeans(mirror.points), yaw: yaw), mirror: mirror.moved)
    }

    // ------------------------------------------------------------ 定数（Android 版と同じ）

    /// 商品の位置が信じられる距離（ShelfMap の範囲と同じ）。
    static let maxDepthM = 3.5
    private static let themeIn = 0.9
    private static let themeNeighbor = 0.3
    private static let minThemeObs = 3.0
    private static let minThemeShare = 0.5
    private static let cell = 0.1
    private static let band = 0.18
    private static let gap = 0.5
    private static let minLen = 0.4
    private static let minObs = 8.0
    private static let sameFace = 0.45
    private static let islandDepth = 0.88
    private static let endDepth = 0.44
    private static let gondolaDepth = 0.45
    private static let caseDepth = 0.8
    private static let maxRays = 20000
    private static let frontQuantile = 0.85
    private static let explain = 0.25
    private static let touch = 0.01
    private static let shelfCost = 3.0
    private static let moduleCost = 0.0
    private static let depthScales = [1.0, 0.66, 0.4]
    private static let sightBeyond = 0.5
    private static let sightMin = 2
    private static let cornerBonus = 12.0
    private static let cornerR = 0.6
    private static let trailMargin = 0.2
    private static let glassGap = 1.2
    private static let glassSpread = 0.6
    private static let glassMin = 8
    private static let glassBonus = 12.0
    private static let height = ["island": 1.8, "wall": 1.8, "case": 2.0, "glass": 2.0, "end": 1.5, "register": 1.0]
    private static let chilled: Set<String> = ["ソフトドリンク", "栄養・エナジードリンク", "紙パック飲料・乳製品", "酒", "冷凍食品", "アイス", "弁当・おにぎり", "サンドイッチ・サラダ・惣菜", "チルドデザート", "飲料", "お酒", "パン・米飯・軽食", "冷蔵・冷凍・デザート・惣菜"]

    private static func floorI(_ v: Double) -> Int { Int(floor(v)) }
    private static func radians(_ deg: Double) -> Double { deg * 0.017453292519943295 }
    private static func overlap(_ a0: Double, _ a1: Double, _ b0: Double, _ b1: Double) -> Double { min(a1, b1) - max(a0, b0) }

    private static func quantile(_ values: [(Double, Double)], _ q: Double) -> Double {
        let v = values.stableSorted { ($0.0, $0.1) < ($1.0, $1.1) }
        let limit = v.reduce(0) { $0 + $1.1 } * q
        var s = 0.0
        for (x, w) in v {
            s += w
            if s >= limit { return x }
        }
        return v.last!.0
    }

    /// 面の前面の位置。点は棚の段と深度の誤差で奥行き方向に帯になるので、中央ではなくカメラ側の端を使う。
    private static func front(_ pts: [Pt], _ ax: Int, _ sign: Int) -> Double {
        quantile(pts.map { ($0.p[ax], $0.w) }, sign > 0 ? frontQuantile : 1 - frontQuantile)
    }

    /// カメラから点への視線がガラス面 face と交わる位置（面に沿った座標）。視線がガラスに届かない点は nil。
    static func restoredAlong(_ camAx: Double, _ camAlong: Double, _ xAx: Double, _ xAlong: Double, _ face: Double, _ sign: Int) -> Double? {
        if (camAx - face) * Double(sign) < 0.3 || abs(xAx - camAx) < 1e-6 { return nil }
        let t = (face - camAx) / (xAx - camAx)
        return t >= 0.05 && t <= 1.3 ? camAlong + t * (xAlong - camAlong) : nil
    }

    private static func fmt(_ v: Double, _ digits: Int = 2) -> String {
        let k = digits == 1 ? 10.0 : 100.0
        return String(format: "%.\(digits)f", floor(v * k + 0.5) / k)
    }

    static func parseThemes(_ data: Data) -> [Theme] {
        guard let a = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] else { return [] }
        return a.map { Theme(name: ($0["name"] as? String) ?? "", cats: Set(($0["cats"] as? [String]) ?? []), genres: Set(($0["genres"] as? [String]) ?? [])) }
    }

    /// 記述言語の文字列にする。
    static func text(_ m: StoreView) -> String {
        guard !m.fixtures.isEmpty, let r = m.room else { return "store (empty)" }
        var s = "store yaw=\(fmt(m.yaw * 180 / Double.pi, 1)) room=\(fmt(r[0])),\(fmt(r[1]))..\(fmt(r[2])),\(fmt(r[3]))"
        for f in m.fixtures {
            let modules = f.moduleWidths.allSatisfy { abs($0 - f.w) < 0.005 } ? "\(f.n)x\(fmt(f.w))" : f.moduleWidths.map { fmt($0) }.joined(separator: "+")
            s += "\n\(f.kind) axis=\(f.axis == 0 ? "u" : "v") c=\(fmt(f.c)) span=\(fmt(f.a))..\(fmt(f.b)) d=\(fmt(f.d)) h=\(fmt(f.h, 1)) mods=\(modules)"
            for i in 0..<f.n {
                let parts = f.faces.keys.sorted().compactMap { sign -> String? in
                    let faces = f.faces[sign]!
                    guard i < faces.count, !faces[i].isEmpty else { return nil }
                    let names = f.themes[sign]
                    let theme = names.flatMap { i < $0.count ? $0[i] : nil }.map { "[\($0)]" } ?? ""
                    return sign + theme + faces[i].map { "\($0.0):\(fmt($0.1))" }.joined(separator: ",")
                }
                if !parts.isEmpty { s += "\n  m\(i) \(parts.joined(separator: " "))" }
            }
        }
        return s
    }
}
