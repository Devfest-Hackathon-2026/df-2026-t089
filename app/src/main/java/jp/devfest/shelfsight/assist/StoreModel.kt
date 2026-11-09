package jp.devfest.shelfsight.assist

import java.util.Locale
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/** 什器 1 つ（記述言語の 1 行）。[axis] は正面が向く軸（0 = u, 1 = v）。[faces] は "+" / "-" の面ごと・モジュールごとの主な商品。 */
class StoreFixture(
    val kind: String, val axis: Int, val c: Double, val a: Double, val b: Double, val d: Double, val h: Double,
    val n: Int, val w: Double, val faces: Map<String, List<List<Pair<String, Double>>>>,
    /** 面ごと・モジュールごとのテーマ（決められなければ null）。 */
    val themes: Map<String, List<String?>> = emptyMap(),
    /** Measured module spans; 0.9 m and 1.2 m are priors, so real observations can adjust them. */
    val moduleWidths: List<Double> = List(n) { w },
) {
    fun moduleStart(i: Int): Double = a + moduleWidths.take(i).sum()
    fun moduleEnd(i: Int): Double = moduleStart(i) + moduleWidths[i]
    fun moduleCenter(i: Int): Double = (moduleStart(i) + moduleEnd(i)) / 2
}

/** 推定した店。[room] は (u0, v0, u1, v1)。[yaw] は世界 → 店の座標の回転（ラジアン）。 */
class StoreView(val yaw: Double, val room: DoubleArray?, val fixtures: List<StoreFixture>,
    /** 点に multi-model RANSAC で当てた線分 (x0, z0, x1, z1)。世界座標。 */
    val ransac: List<DoubleArray> = emptyList(),
    /** ガラスと判断して動かした点の (元 x, 元 z, 新 x, 新 z)。世界座標。 */
    val mirror: List<DoubleArray> = emptyList())

/**
 * コンビニ記述言語のモデルを、「商品＋位置」の連続した観測から推定する。products/store_model.py の移植で、
 * 記述言語の定義と推定の手順はそちらに書いてある。同じ入力で text() が一致することを StoreModelTest で確かめる。
 * 1 つのスレッドから使う。
 */
/** ゴンドラのテーマの定型 1 つ（assets/store_themes.json）。[cats] は 120 カテゴリの名前、[genres] は今のアプリの 26 分類。 */
class Theme(val name: String, val cats: Set<String>, val genres: Set<String>)

class StoreModel(private val themes: List<Theme> = emptyList()) {
    private class Cell(var w: Double = 0.0, var sx: Double = 0.0, var sz: Double = 0.0, var nx: Double = 0.0, var nz: Double = 0.0,
        val cats: LinkedHashMap<String, Double> = LinkedHashMap(), val cameras: ArrayList<DoubleArray> = ArrayList())
    private class Pt(val p: DoubleArray, val w: Double, val nu: Double, val nv: Double,
        val cats: Map<String, Double>, val cameras: List<DoubleArray>)
    private class Run(val ax: Int, var c: Double, var a: Double, var b: Double, val s: Int, var w: Double, var pts: List<Pt>, var corner: Boolean = false)
    private class Draft(val kind: String, val axis: Int, val c: Double, val a: Double, val b: Double, val d: Double, val faces: List<Run>)
    private data class ModuleLayout(val start: Double, val widths: List<Double>)
    private data class Pairing(val front: Run, val back: Run, val score: Double)
    private data class Rect(val u0: Double, val v0: Double, val u1: Double, val v1: Double)
    private class Candidate(var draft: Draft, var fixture: StoreFixture, var explains: Set<Pt>, val bonus: Double, val through: Set<Pt> = emptySet())

    private val cells = LinkedHashMap<Pair<Int, Int>, Cell>()
    private val counters = LinkedHashMap<Pair<Int, Int>, Int>()
    private val trail = ArrayList<DoubleArray>()
    private val genre = HashMap<String, String>()
    private var obs = 0
    /** 観測ごとの (カメラ x, カメラ z, 点 x, 点 z)。ガラスの補正は同じ視点の視線どうしを比べるので、格子にまとめる前の形で持つ。 */
    private val rays = ArrayDeque<DoubleArray>()
    private var yaw = 0.0
    private var yawAt = 0

    /** 商品 1 つの観測（世界座標）。[category] が空なら位置だけを使う。 */
    fun addProduct(camX: Double, camZ: Double, x: Double, z: Double, category: String, genreName: String) {
        val d = hypot(camX - x, camZ - z)
        if (d < 1e-6) return
        val c = cells.getOrPut(key(x, z)) { Cell() }
        c.w += 1; c.sx += x; c.sz += z
        c.nx += (camX - x) / d
        c.nz += (camZ - z) / d
        if (c.cameras.none { hypot(it[0] - camX, it[1] - camZ) < 0.35 }) {
            if (c.cameras.size == 4) c.cameras.removeAt(0)
            c.cameras += doubleArrayOf(camX, camZ)
        }
        if (category.isNotEmpty()) {
            c.cats[category] = (c.cats[category] ?: 0.0) + 1
            genre[category] = genreName
        }
        obs++
        if (d <= ShelfMap.MAX_DEPTH_M) rays.addLast(doubleArrayOf(camX, camZ, x, z))   // これより遠い深度は信頼できない
        if (rays.size > MAX_RAYS) rays.removeFirst()
    }

    fun addCamera(x: Double, z: Double) {
        val last = trail.lastOrNull()
        if (last == null || hypot(last[0] - x, last[1] - z) >= 0.3) trail += doubleArrayOf(x, z)
    }

    /** 腰の高さの水平面の上の点（レジの手がかり）。 */
    fun addCounter(x: Double, z: Double) {
        counters.merge(key(x, z), 1, Int::plus)
    }

    private fun key(x: Double, z: Double) = floorI(x / CELL) to floorI(z / CELL)

    private fun fitYaw() {
        val pts = cells.values.map { doubleArrayOf(it.sx / it.w, it.sz / it.w, min(it.w, 5.0)) }
        fun score(deg: Double): Double {
            val cs = cos(Math.toRadians(deg)); val sn = sin(Math.toRadians(deg))
            val us = HashMap<Int, Double>(); val vs = HashMap<Int, Double>()
            for ((x, z, w) in pts.map { Triple(it[0], it[1], it[2]) }) {
                us.merge(floorI((x * cs + z * sn) / 0.2), w, Double::plus)
                vs.merge(floorI((-x * sn + z * cs) / 0.2), w, Double::plus)
            }
            return us.values.sumOf { it * it } + vs.values.sumOf { it * it }
        }
        fun best(cands: List<Double>): Double {
            var top = cands[0]; var topScore = -1.0
            for (d in cands) { val s = score(d); if (s > topScore) { top = d; topScore = s } }
            return top
        }
        var prev = yaw * 180 / Math.PI
        if (yawAt == 0 || obs >= yawAt * 2) {
            yawAt = obs
            prev = best((0 until 30).map { it * 3.0 })
        }
        val d = best((-6..6).map { prev + 0.5 * it })
        yaw = Math.toRadians(((d % 90.0) + 90.0) % 90.0)
    }

    /** 歩いた線がこの線を横切る位置（面に沿った座標）。そこは通路で、棚はつながらない。 */
    private fun walkedAcross(t: List<DoubleArray>, ax: Int, line: Double): DoubleArray {
        val out = ArrayList<Double>()
        for (i in 0 until t.size - 1) {
            val p = t[i]; val q = t[i + 1]
            if ((p[ax] - line) * (q[ax] - line) < 0) out += p[1 - ax] + (line - p[ax]) / (q[ax] - p[ax]) * (q[1 - ax] - p[1 - ax])
        }
        return out.sorted().toDoubleArray()
    }

    private fun piece(cuts: DoubleArray, x: Double) = cuts.count { it <= x }

    /** 視線がこの線を通り抜けて、その先の点に届いた位置。棚は透けないので、複数の視線が通った場所に棚はない。 */
    private fun sightCuts(pts: List<Pt>, ax: Int, line: Double): List<Double> {
        val hits = HashMap<Int, Int>()
        for (p in pts) {
            if (abs(p.p[ax] - line) < SIGHT_BEYOND) continue   // 棚の帯の中の点は、その棚の奥行きの誤差
            for (cam in p.cameras) {
                if ((cam[ax] - line) * (p.p[ax] - line) < 0) {
                    val at = cam[1 - ax] + (line - cam[ax]) / (p.p[ax] - cam[ax]) * (p.p[1 - ax] - cam[1 - ax])
                    hits.merge(floorI(at / 0.2), 1, Int::plus)
                }
            }
        }
        return hits.filterValues { it >= SIGHT_MIN }.keys.map { (it + 0.5) * 0.2 }
    }

    /** この線の上で、棚がつながらない位置（歩いた場所と視線が通った場所）。 */
    private fun openAt(pts: List<Pt>, frameTrail: List<DoubleArray>, ax: Int, line: Double): DoubleArray =
        (walkedAcross(frameTrail, ax, line).toList() + sightCuts(pts, ax, line)).sorted().toDoubleArray()

    /** 軸に平行な棚の面を、点の多い線から順に取り出す。線は歩いた線で区切り、面にならなかった点は直交する棚のために残す。 */
    private fun runsOf(input: List<Pt>, frameTrail: List<DoubleArray>): List<Run> {
        val runs = ArrayList<Run>()
        var left = input
        val tried = HashSet<Pair<Int, Int>>()
        val cache = HashMap<Pair<Int, Int>, DoubleArray>()
        fun cuts(ax: Int, b: Int) = cache.getOrPut(ax to b) { openAt(input, frameTrail, ax, (b + 0.5) * 0.05) }
        while (left.size >= 5) {
            var score = 0; var axis = 0; var center = 0.0; var line = 0
            for (ax in 0..1) {
                val bins = HashMap<Int, HashSet<Int>>()
                for (p in left) bins.getOrPut(floorI(p.p[ax] / 0.05)) { HashSet() }.add(floorI(p.p[1 - ax] / 0.2))
                for (b in bins.keys.sorted()) {
                    if ((ax to b) in tried) continue
                    val pieces = HashMap<Int, Int>()
                    for (i in -3..3) for (y in bins[b + i].orEmpty()) pieces.merge(piece(cuts(ax, b), (y + 0.5) * 0.2), 1, Int::plus)
                    val s = pieces.values.maxOrNull() ?: 0
                    if (s > score) { score = s; axis = ax; center = (b + 0.5) * 0.05; line = b }
                }
            }
            if (score < 4) break
            tried += axis to line
            val t = 1 - axis
            val ax = axis; val mid = center
            val walked = cuts(axis, line)
            val band = left.filter { abs(it.p[ax] - mid) <= BAND }.sortedBy { it.p[t] }
            val taken = HashSet<Pt>()
            var seg = ArrayList<Pt>()
            for (p in band + listOf<Pt?>(null)) {
                if (seg.isNotEmpty() && (p == null || p.p[t] - seg.last().p[t] > GAP || piece(walked, p.p[t]) != piece(walked, seg.last().p[t]))) {
                    val a = seg.first().p[t]; val b = seg.last().p[t]; val w = seg.sumOf { it.w }
                    if (b - a >= MIN_LEN && w >= MIN_OBS) {
                        val sign = if (seg.sumOf { if (ax == 0) it.nu else it.nv } >= 0) 1 else -1
                        runs += Run(ax, front(seg, ax, sign), a, b, sign, w, seg)
                        taken += seg
                    }
                    seg = ArrayList()
                }
                if (p != null) seg.add(p)
            }
            left = left.filter { it !in taken }
        }
        return runs
    }

    /** 同じ線の上で少しだけ途切れた面をつなぐ。間を歩いたか視線が通っていたら、通路なのでつながない。 */
    private fun merge(runs: List<Run>, frameTrail: List<DoubleArray>, pts: List<Pt>): List<Run> {
        val out = ArrayList<Run>()
        for (r in runs.sortedWith(compareBy({ it.ax }, { it.s }, { it.a }))) {
            val hit = out.firstOrNull { m -> m.ax == r.ax && m.s == r.s && abs(m.c - r.c) < SAME_FACE && r.a - m.b < 1.0 &&
                openAt(pts, frameTrail, m.ax, (m.c + r.c) / 2).none { it >= m.b && it <= r.a } }
            if (hit == null) out += r else {
                hit.b = max(hit.b, r.b); hit.w += r.w; hit.pts = hit.pts + r.pts
                hit.c = front(hit.pts, hit.ax, hit.s)
            }
        }
        return out
    }

    /** 直交する 2 つの面の端が近ければ、そこは棚の角。両方の面の端を交点にそろえて、棚を角に当てる。 */
    private fun snapCorners(runs: List<Run>) {
        class End(val d: Double, val r: Run, val rLow: Boolean, val q: Run, val qLow: Boolean)
        val ends = ArrayList<End>()
        for (r in runs) {
            if (r.ax != 0) continue
            for (q in runs) {
                if (q.ax != 1) continue
                for (rLow in listOf(true, false)) for (qLow in listOf(true, false)) {
                    val d = hypot((if (rLow) r.a else r.b) - q.c, (if (qLow) q.a else q.b) - r.c)
                    if (d < CORNER_R) ends += End(d, r, rLow, q, qLow)
                }
            }
        }
        val used = HashSet<Pair<Run, Boolean>>()
        for (e in ends.sortedBy { it.d }) {
            if ((e.r to e.rLow) in used || (e.q to e.qLow) in used) continue
            val ra = if (e.rLow) e.q.c else e.r.a; val rb = if (e.rLow) e.r.b else e.q.c
            val qa = if (e.qLow) e.r.c else e.q.a; val qb = if (e.qLow) e.q.b else e.r.c
            if (rb - ra < MIN_LEN || qb - qa < MIN_LEN) continue
            used += e.r to e.rLow
            used += e.q to e.qLow
            e.r.a = ra; e.r.b = rb; e.q.a = qa; e.q.b = qb
            e.r.corner = true; e.q.corner = true
        }
    }

    /** 面の後ろを歩いたか、後ろに別の棚の面があれば、壁際の什器ではない。 */
    private fun wallBehind(r: Run, runs: List<Run>, frameTrail: List<DoubleArray>): Boolean {
        val back = { cross: Double -> (r.c - cross) * r.s }
        val t = 1 - r.ax
        if (frameTrail.any { it[t] >= r.a && it[t] <= r.b && back(it[r.ax]) > 0.9 }) return false
        return runs.none { it !== r && it.ax == r.ax && back(it.c) > 1.2 && overlap(r.a, r.b, it.a, it.b) > 0.3 }
    }

    /** Select opposite faces jointly so one measured face cannot explain two different islands. */
    private fun islandPairs(runs: List<Run>): List<Pairing> {
        val fronts = runs.filter { it.s == 1 && it.b - it.a >= 0.8 }
        val backs = runs.filter { it.s == -1 && it.b - it.a >= 0.8 }
        fun candidate(a: Run, b: Run): Pairing? {
            if (a.ax != b.ax) return null
            val depth = a.c - b.c
            if (depth !in 0.5..1.3) return null
            val overlap = overlap(a.a, a.b, b.a, b.b)
            val shorter = min(a.b - a.a, b.b - b.a)
            if (overlap <= 0.3 * shorter) return null
            // Overlap and the ordinary 880 mm depth are soft evidence, not hard dimensions.
            val score = overlap / shorter + (1.0 - abs(depth - ISLAND_DEPTH) / ISLAND_DEPTH) +
                min(a.w, b.w) / (a.w + b.w)
            return Pairing(a, b, score)
        }
        val all = fronts.flatMap { a -> backs.mapNotNull { b -> candidate(a, b) } }
        if (all.isEmpty()) return emptyList()
        if (backs.size > 18) {
            val usedA = HashSet<Run>(); val usedB = HashSet<Run>()
            return all.sortedByDescending { it.score }.filter {
                if (it.front in usedA || it.back in usedB) false
                else { usedA += it.front; usedB += it.back; true }
            }
        }
        val index = backs.withIndex().associate { it.value to it.index }
        val byFront = fronts.map { a -> all.filter { it.front === a } }
        val memo = HashMap<Pair<Int, Int>, Pair<Double, List<Pairing>>>()
        fun solve(i: Int, used: Int): Pair<Double, List<Pairing>> {
            if (i == fronts.size) return 0.0 to emptyList()
            val key = i to used
            memo[key]?.let { return it }
            var best = solve(i + 1, used)
            for (pair in byFront[i]) {
                val bit = 1 shl index.getValue(pair.back)
                if (used and bit != 0) continue
                val rest = solve(i + 1, used or bit)
                if (pair.score + rest.first > best.first) best = pair.score + rest.first to (listOf(pair) + rest.second)
            }
            memo[key] = best
            return best
        }
        return solve(0, 0).second
    }

    private fun classify(runs: List<Run>, frameTrail: List<DoubleArray>): List<Draft> {
        val out = ArrayList<Draft>(); val islands = ArrayList<Draft>(); val used = HashSet<Run>()
        for ((a, b) in islandPairs(runs)) {
            used += a; used += b
            islands += Draft("island", a.ax, (a.c + b.c) / 2,
                min(a.a, b.a), max(a.b, b.b), a.c - b.c, listOf(a, b))
        }
        fun takeEnds() {
            for (r in runs) {
                if (r in used || r.b - r.a > 1.5) continue
                val mid = (r.a + r.b) / 2
                for (i in islands) {
                    val atB = r.s == 1 && r.c >= i.b - 0.2 && r.c <= i.b + 0.8
                    val atA = r.s == -1 && r.c >= i.a - 0.8 && r.c <= i.a + 0.2
                    // A real end cap supplies both corners of the island's short edge.
                    val coversEdge = r.a <= i.c - i.d / 2 + 0.25 && r.b >= i.c + i.d / 2 - 0.25
                    if (r.ax == 1 - i.axis && abs(mid - i.c) < 0.3 && coversEdge && (atA || atB)) {
                        used += r
                        out += Draft("end", r.ax, r.c - r.s * END_DEPTH / 2, i.c - 0.45, i.c + 0.45, END_DEPTH, listOf(r))
                        break
                    }
                }
            }
        }
        takeEnds()
        for (r in runs) if (r !in used && r.b - r.a >= 0.8 && !wallBehind(r, runs, frameTrail)) {
            used += r
            islands += Draft("island", r.ax, r.c - r.s * ISLAND_DEPTH / 2, r.a, r.b, ISLAND_DEPTH, listOf(r))
        }
        takeEnds()
        for (r in runs) {
            if (r in used) continue
            if (wallBehind(r, runs, frameTrail)) {
                val named = r.pts.flatMap { it.cats.entries }
                val chilled = named.filter { genre[it.key] in CHILLED }.sumOf { it.value }
                val isCase = named.isNotEmpty() && chilled >= named.sumOf { it.value } / 2
                val depth = if (isCase) CASE_DEPTH else GONDOLA_DEPTH
                out += Draft(if (isCase) "case" else "wall", r.ax, r.c - r.s * depth / 2, r.a, r.b, depth, listOf(r))
            } else {
                islands += Draft("island", r.ax, r.c - r.s * ISLAND_DEPTH / 2, r.a, r.b, ISLAND_DEPTH, listOf(r))
            }
        }
        return islands + out
    }

    private fun inTheme(t: Theme, cat: String) = cat in t.cats || genre[cat] in t.genres

    /** モジュールごとに、観測に一番合うテーマを選ぶ（決められなければ null）。 */
    private fun assignThemes(mods: List<Map<String, Double>>): List<Theme?> {
        if (themes.isEmpty()) return List(mods.size) { null }
        val n = themes.flatMap { it.cats }.toSet().size
        val scores = mods.map { m ->
            themes.map { t ->
                var s = 0.0
                for ((c, v) in m) s += v * (if (inTheme(t, c)) ln(THEME_IN / t.cats.size + (1 - THEME_IN) / n) else ln((1 - THEME_IN) / n))
                s
            }
        }
        return mods.indices.map { i ->
            val m = mods[i]
            val total = m.values.sum()
            if (total < MIN_THEME_OBS) return@map null
            val near = listOf(i - 1, i + 1).filter { it in mods.indices }.map { scores[it] }
            var best = 0
            var top = Double.NEGATIVE_INFINITY
            for (k in themes.indices) {
                var s = scores[i][k]
                var sum = 0.0
                for (x in near) sum += x[k]
                s += THEME_NEIGHBOR * sum
                if (s > top) { top = s; best = k }
            }
            val theme = themes[best]
            val share = m.entries.filter { inTheme(theme, it.key) }.sumOf { it.value } / total
            theme.takeIf { share >= MIN_THEME_SHARE }
        }
    }

    /** Fit a mix of 3-shaku and 4-shaku modules while preserving well-observed row endpoints. */
    private fun moduleLayout(d: Draft): ModuleLayout {
        val mid = (d.a + d.b) / 2
        if (d.kind == "end") return ModuleLayout(mid - 0.45, listOf(0.9))
        val measured = d.b - d.a
        var best3 = 0; var best4 = 0; var bestCost = Double.MAX_VALUE
        for (n3 in 0..12) for (n4 in 0..12) {
            val n = n3 + n4
            if (n !in 1..12) continue
            val cost = abs(0.9 * n3 + 1.2 * n4 - measured) + n * 0.005
            if (cost < bestCost) { bestCost = cost; best3 = n3; best4 = n4 }
        }
        val count = best3 + best4
        val prior = best3 * 0.9 + best4 * 1.2
        val observations = d.faces.sumOf { it.w }
        val trust = if (d.faces.any { it.corner }) 1.0 else (observations / 30.0).coerceIn(0.0, 1.0)   // 角に当たった端は測った位置を信じる
        val target = prior + (measured - prior) * trust
        val correction = (target - prior) / count
        val small = 0.9 + correction; val large = 1.2 + correction
        val start = mid - target / 2
        val pts = d.faces.flatMap { it.pts }
        var best = emptyList<Double>()
        var bestEvidence = Double.NEGATIVE_INFINITY
        fun evaluate(widths: List<Double>) {
            val cats = List(widths.size) { HashMap<String, Double>() }
            var at = start
            for ((i, width) in widths.withIndex()) {
                val end = at + width
                for (p in pts) if (p.p[1 - d.axis] >= at && (p.p[1 - d.axis] < end || i == widths.lastIndex)) {
                    for ((cat, weight) in p.cats) cats[i].merge(cat, weight, Double::plus)
                }
                at = end
            }
            val evidence = cats.sumOf { it.values.maxOrNull() ?: 0.0 }
            if (evidence > bestEvidence + 1e-6) { bestEvidence = evidence; best = widths }
        }
        fun enumerate(n3: Int, n4: Int, widths: MutableList<Double>) {
            if (n3 + n4 == 0) { evaluate(widths.toList()); return }
            if (n3 > 0) { widths += small; enumerate(n3 - 1, n4, widths); widths.removeAt(widths.lastIndex) }
            if (n4 > 0) { widths += large; enumerate(n3, n4 - 1, widths); widths.removeAt(widths.lastIndex) }
        }
        enumerate(best3, best4, ArrayList(count))
        return ModuleLayout(start, best)
    }

    private fun build(d: Draft): StoreFixture {
        val layout = moduleLayout(d)
        val a = layout.start
        val widths = layout.widths
        val n = widths.size
        val w = widths.sum() / n
        val edges = DoubleArray(n + 1)
        edges[0] = a
        for (i in 0 until n) edges[i + 1] = edges[i] + widths[i]
        val faces = LinkedHashMap<String, List<List<Pair<String, Double>>>>()
        val themeOf = LinkedHashMap<String, List<String?>>()
        for (sign in listOf(1, -1)) {
            val sides = d.faces.filter { it.s == sign }
            if (sides.isEmpty()) continue
            val mods = List(n) { LinkedHashMap<String, Double>() }
            for (r in sides) for (p in r.pts) {
                val coord = p.p[1 - r.ax]
                val index = (0 until n - 1).firstOrNull { coord < edges[it + 1] } ?: n - 1
                val m = mods[index]
                for ((cat, v) in p.cats) m[cat] = (m[cat] ?: 0.0) + v
            }
            val side = if (sign == 1) "+" else "-"
            val assigned = assignThemes(mods)
            themeOf[side] = assigned.map { it?.name }
            faces[side] = mods.indices.map { i ->
                val theme = assigned[i]
                val kept = mods[i].entries.filter { theme == null || inTheme(theme, it.key) }
                val total = kept.sumOf { it.value }.let { if (it == 0.0) 1.0 else it }
                kept.sortedWith(compareBy({ -it.value }, { it.key })).take(3).map { it.key to it.value / total }
            }
        }
        return StoreFixture(d.kind, d.axis, d.c, a, edges[n], d.d, HEIGHT.getValue(d.kind), n, w, faces, themeOf, widths)
    }

    private fun rect(f: StoreFixture): Rect = if (f.axis == 0)
        Rect(f.c - f.d / 2, f.a, f.c + f.d / 2, f.b)
    else Rect(f.a, f.c - f.d / 2, f.b, f.c + f.d / 2)

    /** Liang-Barsky line clipping. @return the parameters where p-q enters and leaves [r], or null. */
    private fun clip(p: DoubleArray, q: DoubleArray, r: Rect, inset: Double): DoubleArray? {
        val lo = doubleArrayOf(r.u0 + inset, r.v0 + inset)
        val hi = doubleArrayOf(r.u1 - inset, r.v1 - inset)
        if (lo[0] >= hi[0] || lo[1] >= hi[1]) return null
        var enter = 0.0; var leave = 1.0
        for (axis in 0..1) {
            val delta = q[axis] - p[axis]
            if (abs(delta) < 1e-9) {
                if (p[axis] <= lo[axis] || p[axis] >= hi[axis]) return null
            } else {
                val a = (lo[axis] - p[axis]) / delta
                val b = (hi[axis] - p[axis]) / delta
                enter = max(enter, min(a, b)); leave = min(leave, max(a, b))
                if (enter >= leave) return null
            }
        }
        return doubleArrayOf(enter, leave)
    }

    private fun crosses(p: DoubleArray, q: DoubleArray, r: Rect, inset: Double) = clip(p, q, r, inset) != null

    private fun cornerSupport(d: Draft, all: List<Draft>): Int {
        fun endpoints(r: Run): List<DoubleArray> = if (r.ax == 0)
            listOf(doubleArrayOf(r.c, r.a), doubleArrayOf(r.c, r.b))
        else listOf(doubleArrayOf(r.a, r.c), doubleArrayOf(r.b, r.c))
        return d.faces.sumOf { face ->
            val ends = endpoints(face)
            all.filter { it !== d }.flatMap { it.faces }.count { other ->
                other.ax != face.ax && ends.any { a -> endpoints(other).any { b -> hypot(a[0] - b[0], a[1] - b[1]) < 0.35 } }
            }
        }
    }

    /**
     * どの棚でも説明できない点の塊のうち、視線方向に大きくばらけたものはガラス越しの深度の失敗とみなす。
     * ガラスは塊のカメラ側の端にあるとして、各点をカメラからの視線に沿ってガラス面へ戻す。
     */
    private fun glassCandidates(rest: List<Pt>): List<Candidate> {
        val groups = ArrayList<List<Pt>>()
        var left = rest.toMutableList()
        while (left.isNotEmpty()) {
            val group = ArrayList<Pt>()
            val frontier = arrayListOf(left.removeAt(0))
            while (frontier.isNotEmpty()) {
                val p = frontier.removeAt(frontier.lastIndex)
                group += p
                frontier += left.filter { hypot(p.p[0] - it.p[0], p.p[1] - it.p[1]) <= GLASS_GAP }
                left = left.filter { hypot(p.p[0] - it.p[0], p.p[1] - it.p[1]) > GLASS_GAP }.toMutableList()
            }
            groups += group
        }
        val out = ArrayList<Candidate>()
        for (group in groups) {
            val views = group.flatMap { p -> p.cameras.map { p to it } }
            if (group.size < GLASS_MIN || views.isEmpty()) continue
            val ax = if (views.sumOf { (p, c) -> abs(p.p[0] - c[0]) } >= views.sumOf { (p, c) -> abs(p.p[1] - c[1]) }) 0 else 1
            val along = 1 - ax
            val sign = if (views.sumOf { (p, c) -> c[ax] - p.p[ax] } >= 0) 1 else -1
            val values = group.map { it.p[ax] to it.w }
            if (quantile(values, 0.9) - quantile(values, 0.1) < GLASS_SPREAD) continue
            val face = quantile(values, if (sign > 0) 0.9 else 0.1)
            val moved = group.mapNotNull { p ->
                val hits = p.cameras.mapNotNull { restoredAlong(it[ax], it[along], p.p[ax], p.p[along], face, sign) }
                if (hits.isEmpty()) null else {
                    val at = hits.sum() / hits.size
                    Pt(if (ax == 0) doubleArrayOf(face, at) else doubleArrayOf(at, face), p.w, p.nu, p.nv, p.cats, p.cameras)
                }
            }
            if (moved.size < GLASS_MIN) continue
            val a = moved.minOf { it.p[along] }; val b = moved.maxOf { it.p[along] }
            if (b - a < MIN_LEN) continue
            val draft = Draft("glass", ax, face - sign * CASE_DEPTH / 2, a, b, CASE_DEPTH, listOf(Run(ax, face, a, b, sign, moved.sumOf { it.w }, moved)))
            out += Candidate(draft, build(draft), group.toHashSet(), GLASS_BONUS)
        }
        return out
    }

    /** Reject occupied walkways, see-through shelves, back-side observations, and products in the rack core. */
    private fun valid(f: StoreFixture, d: Draft, pts: List<Pt>, frameTrail: List<DoubleArray>): Boolean {
        val r = rect(f)
        if (frameTrail.any { crosses(it, it, r, TRAIL_MARGIN) } ||
            frameTrail.zipWithNext().any { (a, b) -> crosses(a, b, r, TRAIL_MARGIN) }) return false

        var wrongSide = 0; var views = 0
        for (face in d.faces) for (p in face.pts) for (cam in p.cameras) {
            views++
            if ((cam[face.ax] - face.c) * face.s < -0.1) wrongSide++
        }
        if (views >= 4 && wrongSide * 4 > views) return false

        val own = d.faces.flatMap { it.pts }.toHashSet()
        if (f.kind == "glass") return true   // ガラスの奥に点があるのは当然なので、下の 2 つは調べない
        // 正面側のカメラから見えた点は、その棚の奥行きの誤差なので矛盾しない
        val core = pts.count { p -> p !in own &&
            p.p[0] > r.u0 + 0.15 && p.p[0] < r.u1 - 0.15 &&
            p.p[1] > r.v0 + 0.15 && p.p[1] < r.v1 - 0.15 &&
            d.faces.none { face -> p.cameras.any { (it[face.ax] - face.c) * face.s > 0 } } }
        return core <= max(6, pts.size / 50)
    }

    /** 棚を長手方向に 10 cm ずつ見て、視線が突き抜けた区間を除いた、棚として残せる区間の draft を返す。棚は透けないため。 */
    private fun carve(d: Draft, f: StoreFixture, pts: List<Pt>): List<Draft> {
        val r = rect(f)
        val along = 1 - d.axis
        val lo = f.a; val hi = f.b
        val n = max(1, ((hi - lo) / 0.1 + 0.999).toInt())
        val cover = IntArray(n)
        for (p in pts) {
            if (maxOf(r.u0 - p.p[0], p.p[0] - r.u1, r.v0 - p.p[1], p.p[1] - r.v1) <= EXPLAIN) continue   // この棚の点（奥行きの誤差も含む）
            for (cam in p.cameras) {
                val hit = clip(cam, p.p, r, 0.02) ?: continue
                val a0 = cam[along] + hit[0] * (p.p[along] - cam[along])
                val a1 = cam[along] + hit[1] * (p.p[along] - cam[along])
                for (i in max(0, ((min(a0, a1) - lo) / 0.1).toInt())..min(n - 1, ((max(a0, a1) - lo) / 0.1).toInt())) cover[i]++
            }
        }
        val pieces = ArrayList<Draft>()
        var i = 0
        while (i < n) {
            if (cover[i] >= SIGHT_MIN) { i++; continue }
            var j = i
            while (j < n && cover[j] < SIGHT_MIN) j++
            val a = lo + i * 0.1; val b = min(hi, lo + j * 0.1)
            // 削った端は測った位置のまま
            if (b - a >= MIN_LEN) pieces += Draft(d.kind, d.axis, d.c, a, b, d.d, d.faces.map { Run(it.ax, it.c, it.a, it.b, it.s, it.w, it.pts, true) })
            i = j
        }
        return pieces
    }

    /** カメラから棚の向こうの点へ、棚を突き抜けて届いた点。不透明な棚では起きない。 */
    private fun seeThrough(f: StoreFixture, pts: List<Pt>): Set<Pt> {
        val r = rect(f)
        val low = f.c - f.d / 2 - 0.2; val high = f.c + f.d / 2 + 0.2
        return pts.filterTo(HashSet()) { p -> p.cameras.any { cam ->
            ((cam[f.axis] < low && p.p[f.axis] > high) || (cam[f.axis] > high && p.p[f.axis] < low)) && crosses(cam, p.p, r, 0.1)
        } }
    }

    private fun glassFront(f: StoreFixture, d: Draft): StoreFixture {
        val sign = d.faces[0].s
        val face = f.c + sign * f.d / 2
        return StoreFixture("glass", f.axis, face - sign * CASE_DEPTH / 2, f.a, f.b, CASE_DEPTH,
            HEIGHT.getValue("glass"), f.n, f.w, f.faces, f.themes, f.moduleWidths)
    }

    private fun overlaps(f: StoreFixture, g: StoreFixture): Boolean {
        val r = rect(f); val q = rect(g)
        return min(r.u1, q.u1) - max(r.u0, q.u0) > TOUCH && min(r.v1, q.v1) - max(r.v0, q.v0) > TOUCH
    }

    /** 棚を長手方向に削って [other] に当たらないようにする。エンドは幅が決まっているので削らない。 */
    private fun trim(c: Candidate, other: StoreFixture): Boolean {
        val d = c.draft
        if (d.kind == "end") return false
        val q = rect(other)
        val lo = if (d.axis == 1) q.u0 else q.v0
        val hi = if (d.axis == 1) q.u1 else q.v1
        val below = (d.a + d.b) / 2 < (lo + hi) / 2
        val a = if (below) d.a else max(d.a, hi)
        val b = if (below) min(d.b, lo) else d.b
        if (b - a < MIN_LEN || (a == d.a && b == d.b)) return false
        c.draft = Draft(d.kind, d.axis, d.c, a, b, d.d, d.faces)
        c.fixture = build(c.draft)
        return true
    }

    /** エンドは島 1 つの端に付く。2 つの列をつなぐと通路がふさがる。 */
    private fun endTouchesOneIsland(f: StoreFixture, accepted: List<Candidate>): Boolean {
        if (f.kind != "end") return true
        val r = rect(f)
        return accepted.count { other ->
            if (other.fixture.kind != "island") false else {
                val q = rect(other.fixture)
                r.u0 <= q.u1 + 0.15 && r.u1 >= q.u0 - 0.15 && r.v0 <= q.v1 + 0.15 && r.v1 >= q.v0 - 0.15
            }
        } == 1
    }

    /** 棚は重なれないので、当たる棚を削る。削っても解けないなら置かない。 */
    private fun place(c: Candidate, accepted: List<Candidate>): Boolean {
        repeat(4) {
            val other = accepted.firstOrNull { overlaps(c.fixture, it.fixture) } ?: return endTouchesOneIsland(c.fixture, accepted)
            // 既に置いた棚を削るのは、幅の決まったエンドを島の先端に付けるときだけ
            if (!trim(c, other.fixture) && !(c.fixture.kind == "end" && trim(other, c.fixture))) return false
        }
        return false
    }

    /** 新しく説明できる点が棚のコストを上回る間だけ、いちばん得な棚を足す。点を説明できる最も少ない棚の組を選ぶため。 */
    private fun choose(candidates: List<Candidate>, accepted: MutableList<Candidate>) {
        val explained = accepted.flatMapTo(HashSet()) { it.explains }
        val left = candidates.toMutableList()
        fun gain(c: Candidate) = c.explains.count { it !in explained } + c.bonus - SHELF_COST - MODULE_COST * c.fixture.n
        while (left.isNotEmpty()) {
            val best = left.maxWithOrNull(compareBy<Candidate>({ gain(it) }, { it.fixture.kind != "end" }, { -it.fixture.c }))!!
            left.remove(best)
            if (gain(best) <= 0) break
            if (place(best, accepted)) {
                accepted += best
                explained += best.explains
            }
        }
    }

    private fun register(cs: Double, sn: Double): StoreFixture? {
        if (counters.size < 6) return null
        val us = counters.keys.map { (i, k) -> (i + 0.5) * CELL * cs + (k + 0.5) * CELL * sn }.sorted()
        val vs = counters.keys.map { (i, k) -> -(i + 0.5) * CELL * sn + (k + 0.5) * CELL * cs }.sorted()
        val cut = us.size / 20
        val u0 = us[cut]; val u1 = us[us.size - 1 - cut]; val v0 = vs[cut]; val v1 = vs[vs.size - 1 - cut]
        val axis = if (u1 - u0 < v1 - v0) 0 else 1
        val c = if (axis == 0) (u0 + u1) / 2 else (v0 + v1) / 2
        val a = if (axis == 0) v0 else u0
        val b = if (axis == 0) v1 else u1
        return StoreFixture("register", axis, c, a, b, max(0.5, if (axis == 0) u1 - u0 else v1 - v0), HEIGHT.getValue("register"), 1, b - a, emptyMap())
    }

    private fun roomOf(fixtures: List<StoreFixture>, frameTrail: List<DoubleArray>): DoubleArray {
        val low = arrayOf(ArrayList<Double>(), ArrayList()); val high = arrayOf(ArrayList<Double>(), ArrayList())
        for (f in fixtures) {
            low[f.axis] += f.c - f.d / 2; high[f.axis] += f.c + f.d / 2
            low[1 - f.axis] += f.a; high[1 - f.axis] += f.b
        }
        for (p in frameTrail) for (ax in 0..1) { low[ax] += p[ax] - 0.5; high[ax] += p[ax] + 0.5 }
        return doubleArrayOf(low[0].min(), low[1].min(), high[0].max(), high[1].max())
    }

    /** 今の観測から店のモデルを作る。 */
    fun estimate(): StoreView {
        if (cells.size < 30) return StoreView(0.0, null, emptyList())
        fitYaw()
        val cs = cos(yaw); val sn = sin(yaw)
        val pts = cells.values.map { c ->
            val x = c.sx / c.w; val z = c.sz / c.w
            Pt(doubleArrayOf(x * cs + z * sn, -x * sn + z * cs), c.w, c.nx * cs + c.nz * sn, -c.nx * sn + c.nz * cs, c.cats,
                c.cameras.map { doubleArrayOf(it[0] * cs + it[1] * sn, -it[0] * sn + it[1] * cs) })
        }
        val frameTrail = trail.map { doubleArrayOf(it[0] * cs + it[1] * sn, -it[0] * sn + it[1] * cs) }
        val runs = merge(runsOf(pts, frameTrail), frameTrail, pts)
        snapCorners(runs)
        val drafts = classify(runs, frameTrail)
        val opaque = ArrayList<Candidate>(); val doubtful = ArrayList<Candidate>()
        val pending = ArrayList(drafts)
        fun candidate(d: Draft, f: StoreFixture, through: Set<Pt> = emptySet()): Candidate {
            val kept = d.faces.flatMap { it.pts }.filterTo(HashSet()) { it.p[1 - d.axis] >= d.a - 0.1 && it.p[1 - d.axis] <= d.b + 0.1 }
            val r = rect(f)
            val near = pts.filterTo(HashSet()) { it.p[0] >= r.u0 - EXPLAIN && it.p[0] <= r.u1 + EXPLAIN && it.p[1] >= r.v0 - EXPLAIN && it.p[1] <= r.v1 + EXPLAIN }
            return Candidate(d, f, near + kept, CORNER_BONUS * cornerSupport(d, drafts), through)
        }
        while (pending.isNotEmpty()) {
            val d = pending.removeAt(0)
            val single = d.faces.size == 1 && d.kind != "end"
            // 面があれば、その裏に棚がある。歩いた線に当たるなら棚を薄くする。視線が通る区間は棚を削る（棚は透けない）
            var best: Triple<Draft, StoreFixture, List<Draft>>? = null
            var bestLength = -1.0
            for (scale in if (single) DEPTH_SCALES else listOf(1.0)) {
                val depth = d.d * scale
                val trial = Draft(d.kind, d.axis, if (single) d.faces[0].c - d.faces[0].s * depth / 2 else d.c, d.a, d.b, depth, d.faces)
                val f = build(trial)
                if (!valid(f, trial, pts, frameTrail)) continue
                val pieces = carve(trial, f, pts)
                val length = pieces.sumOf { it.b - it.a }
                if (best == null || length > bestLength + 1e-6) { best = Triple(trial, f, pieces); bestLength = length }
            }
            if (best == null || best.third.isEmpty()) {
                if (d.faces.size == 2) {   // 島として成り立たなくても、面ごとの棚は残す
                    for (r in d.faces) pending += Draft("wall", r.ax, r.c - r.s * GONDOLA_DEPTH / 2, r.a, r.b, GONDOLA_DEPTH, listOf(r))
                } else if (best != null && (d.kind == "wall" || d.kind == "case")) {   // 全長を視線に貫かれる面の説明は、壁際のガラスだけ
                    doubtful += candidate(best.first, best.second, seeThrough(best.second, pts))
                }
                continue
            }
            for (piece in best.third) opaque += candidate(piece, build(piece))
        }
        val accepted = ArrayList<Candidate>()
        choose(opaque, accepted)
        var explained = accepted.flatMapTo(HashSet()) { it.explains }
        for (c in doubtful) {
            c.fixture = glassFront(c.fixture, c.draft)
            c.explains = c.explains + (c.through - explained)
        }
        choose(doubtful, accepted)
        explained = accepted.flatMapTo(HashSet()) { it.explains }
        choose(glassCandidates(pts.filter { it !in explained }).filter { valid(it.fixture, it.draft, pts, frameTrail) }, accepted)
        val fixtures = ArrayList<StoreFixture>()
        for (c in accepted) if (fixtures.none { overlaps(c.fixture, it) }) fixtures += c.fixture   // 削った棚はモジュール幅を合わせ直すので、最後にもう一度確かめる
        register(cs, sn)?.let { reg -> if (fixtures.none { overlaps(reg, it) }) fixtures += reg }
        val sorted = fixtures.sortedWith(compareBy({ it.kind }, { it.axis }, { it.c }, { it.a }))
        val mirror = MirrorCorrection.correct(rays.toList())
        return StoreView(yaw, roomOf(sorted, frameTrail), sorted, MultiLineRansac.fit(MirrorCorrection.cellMeans(mirror.points), yaw), mirror.moved)
    }

    companion object {
        private const val THEME_IN = 0.9
        private const val THEME_NEIGHBOR = 0.3
        private const val MIN_THEME_OBS = 3.0
        private const val MIN_THEME_SHARE = 0.5
        private const val CELL = 0.1
        private const val BAND = 0.18
        private const val GAP = 0.5
        private const val MIN_LEN = 0.4
        private const val MIN_OBS = 8.0
        private const val SAME_FACE = 0.45
        private const val ISLAND_DEPTH = 0.88
        private const val END_DEPTH = 0.44
        private const val GONDOLA_DEPTH = 0.45
        private const val CASE_DEPTH = 0.8
        private const val MAX_RAYS = 20000
        private const val FRONT = 0.85
        private const val EXPLAIN = 0.25
        private const val TOUCH = 0.01
        private const val SHELF_COST = 3.0
        private const val MODULE_COST = 0.0
        private val DEPTH_SCALES = listOf(1.0, 0.66, 0.4)
        private const val SIGHT_BEYOND = 0.5
        private const val SIGHT_MIN = 2
        private const val CORNER_BONUS = 12.0
        private const val CORNER_R = 0.6
        private const val TRAIL_MARGIN = 0.2
        private const val GLASS_GAP = 1.2
        private const val GLASS_SPREAD = 0.6
        private const val GLASS_MIN = 8
        private const val GLASS_BONUS = 12.0
        private val HEIGHT = mapOf("island" to 1.8, "wall" to 1.8, "case" to 2.0, "glass" to 2.0, "end" to 1.5, "register" to 1.0)
        private val CHILLED = setOf("ソフトドリンク", "栄養・エナジードリンク", "紙パック飲料・乳製品", "酒", "冷凍食品", "アイス", "弁当・おにぎり", "サンドイッチ・サラダ・惣菜", "チルドデザート", "飲料", "お酒", "パン・米飯・軽食", "冷蔵・冷凍・デザート・惣菜")

        private fun floorI(v: Double) = floor(v).toInt()
        private fun overlap(a0: Double, a1: Double, b0: Double, b1: Double) = min(a1, b1) - max(a0, b0)

        private fun quantile(values: List<Pair<Double, Double>>, q: Double): Double {
            val v = values.sortedWith(compareBy({ it.first }, { it.second }))
            val limit = v.sumOf { it.second } * q
            var s = 0.0
            for ((x, w) in v) { s += w; if (s >= limit) return x }
            return v.last().first
        }

        /** 面の前面の位置。点は棚の段と深度の誤差で奥行き方向に帯になるので、中央ではなくカメラ側の端を使う。 */
        private fun front(pts: List<Pt>, ax: Int, sign: Int) = quantile(pts.map { it.p[ax] to it.w }, if (sign > 0) FRONT else 1 - FRONT)

        /** カメラから点への視線がガラス面 [face] と交わる位置（面に沿った座標）。視線がガラスに届かない点は null。 */
        fun restoredAlong(camAx: Double, camAlong: Double, xAx: Double, xAlong: Double, face: Double, sign: Int): Double? {
            if ((camAx - face) * sign < 0.3 || abs(xAx - camAx) < 1e-6) return null
            val t = (face - camAx) / (xAx - camAx)
            return if (t in 0.05..1.3) camAlong + t * (xAlong - camAlong) else null
        }

        private fun fmt(v: Double, digits: Int = 2): String {
            val k = if (digits == 1) 10.0 else 100.0
            return String.format(Locale.ROOT, "%.${digits}f", floor(v * k + 0.5) / k)
        }

        fun parseThemes(json: String): List<Theme> {
            val a = org.json.JSONArray(json)
            fun strings(o: org.json.JSONArray) = (0 until o.length()).map { o.getString(it) }.toSet()
            return (0 until a.length()).map { i ->
                val o = a.getJSONObject(i)
                Theme(o.getString("name"), strings(o.getJSONArray("cats")), strings(o.getJSONArray("genres")))
            }
        }

        /** 記述言語の文字列にする。 */
        fun text(m: StoreView): String {
            if (m.fixtures.isEmpty()) return "store (empty)"
            val r = m.room!!
            return buildString {
                append("store yaw=${fmt(m.yaw * 180 / Math.PI, 1)} room=${fmt(r[0])},${fmt(r[1])}..${fmt(r[2])},${fmt(r[3])}")
                for (f in m.fixtures) {
                    val modules = if (f.moduleWidths.all { abs(it - f.w) < 0.005 }) "${f.n}x${fmt(f.w)}"
                        else f.moduleWidths.joinToString("+") { fmt(it) }
                    append("\n${f.kind} axis=${"uv"[f.axis]} c=${fmt(f.c)} span=${fmt(f.a)}..${fmt(f.b)} d=${fmt(f.d)} h=${fmt(f.h, 1)} mods=$modules")
                    for (i in 0 until f.n) {
                        val parts = f.faces.entries.sortedBy { it.key }.filter { i < it.value.size && it.value[i].isNotEmpty() }
                            .map { (sign, faces) -> sign + (f.themes[sign]?.getOrNull(i)?.let { "[$it]" } ?: "") + faces[i].joinToString(",") { (c, s) -> "$c:${fmt(s)}" } }
                        if (parts.isNotEmpty()) append("\n  m$i ${parts.joinToString(" ")}")
                    }
                }
            }
        }
    }
}
