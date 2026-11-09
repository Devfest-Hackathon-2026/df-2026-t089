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
class StoreView(val yaw: Double, val room: DoubleArray?, val fixtures: List<StoreFixture>)

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
    private class Run(val ax: Int, var c: Double, val a: Double, var b: Double, val s: Int, var w: Double, var pts: List<Pt>)
    private class Draft(val kind: String, val axis: Int, val c: Double, val a: Double, val b: Double, val d: Double, val faces: List<Run>)
    private data class ModuleLayout(val start: Double, val widths: List<Double>)
    private data class Pairing(val front: Run, val back: Run, val score: Double)
    private data class Rect(val u0: Double, val v0: Double, val u1: Double, val v1: Double)
    private data class Candidate(val draft: Draft, val fixture: StoreFixture, val score: Double)

    private val cells = LinkedHashMap<Pair<Int, Int>, Cell>()
    private val counters = LinkedHashMap<Pair<Int, Int>, Int>()
    private val trail = ArrayList<DoubleArray>()
    private val genre = HashMap<String, String>()
    private var obs = 0
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

    private fun runsOf(input: List<Pt>): List<Run> {
        val runs = ArrayList<Run>()
        var left = input
        while (left.size >= 5) {
            var score = 0; var axis = 0; var center = 0.0
            for (ax in 0..1) {
                val bins = HashMap<Int, HashSet<Int>>()
                for (p in left) bins.getOrPut(floorI(p.p[ax] / 0.05)) { HashSet() }.add(floorI(p.p[1 - ax] / 0.2))
                for (b in bins.keys.sorted()) {
                    val s = (-3..3).sumOf { bins[b + it]?.size ?: 0 }
                    if (s > score) { score = s; axis = ax; center = (b + 0.5) * 0.05 }
                }
            }
            if (score < 4) break
            val t = 1 - axis
            val ax = axis; val mid = center
            val band = left.filter { abs(it.p[ax] - mid) <= BAND }.sortedBy { it.p[t] }
            left = left.filter { abs(it.p[ax] - mid) > BAND }
            var seg = ArrayList<Pt>()
            for (p in band + listOf<Pt?>(null)) {
                if (seg.isNotEmpty() && (p == null || p.p[t] - seg.last().p[t] > GAP)) {
                    val a = seg.first().p[t]; val b = seg.last().p[t]; val w = seg.sumOf { it.w }
                    if (b - a >= MIN_LEN && w >= MIN_OBS) {
                        val sum = seg.sumOf { if (ax == 0) it.nu else it.nv }
                        runs += Run(ax, median(seg.map { it.p[ax] to it.w }), a, b, if (sum >= 0) 1 else -1, w, seg)
                    }
                    seg = ArrayList()
                }
                if (p != null) seg.add(p)
            }
        }
        return runs
    }

    private fun merge(runs: List<Run>): List<Run> {
        val out = ArrayList<Run>()
        for (r in runs.sortedWith(compareBy({ it.ax }, { it.s }, { it.a }))) {
            val hit = out.firstOrNull { it.ax == r.ax && it.s == r.s && abs(it.c - r.c) < SAME_FACE && r.a - it.b < 1.0 }
            if (hit == null) out += r else {
                hit.c = (hit.c * hit.w + r.c * r.w) / (hit.w + r.w)
                hit.b = max(hit.b, r.b); hit.w += r.w; hit.pts = hit.pts + r.pts
            }
        }
        return out
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
        val trust = (observations / 30.0).coerceIn(0.0, 1.0)
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

    /** Liang-Barsky line clipping. A path segment is free space, including between sampled poses. */
    private fun crosses(p: DoubleArray, q: DoubleArray, r: Rect, inset: Double): Boolean {
        val lo = doubleArrayOf(r.u0 + inset, r.v0 + inset)
        val hi = doubleArrayOf(r.u1 - inset, r.v1 - inset)
        if (lo[0] >= hi[0] || lo[1] >= hi[1]) return false
        var enter = 0.0; var leave = 1.0
        for (axis in 0..1) {
            val delta = q[axis] - p[axis]
            if (abs(delta) < 1e-9) {
                if (p[axis] <= lo[axis] || p[axis] >= hi[axis]) return false
            } else {
                val a = (lo[axis] - p[axis]) / delta
                val b = (hi[axis] - p[axis]) / delta
                enter = max(enter, min(a, b)); leave = min(leave, max(a, b))
                if (enter >= leave) return false
            }
        }
        return true
    }

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

    /** A raw depth outlier behind a glass front is explained by its camera ray hitting that front. */
    private fun glassHit(p: Pt, cam: DoubleArray, f: StoreFixture): Boolean {
        val sign = when {
            f.faces.containsKey("+") && !f.faces.containsKey("-") -> 1
            f.faces.containsKey("-") && !f.faces.containsKey("+") -> -1
            else -> return false
        }
        val ax = f.axis; val along = 1 - ax
        val face = f.c + sign * f.d / 2
        val delta = p.p[ax] - cam[ax]
        if (abs(delta) < 1e-6 || (cam[ax] - face) * sign < 0.3 || (face - p.p[ax]) * sign < 0.35) return false
        val t = (face - cam[ax]) / delta
        if (t !in 0.1..0.85) return false
        val at = cam[along] + t * (p.p[along] - cam[along])
        return at > f.a + 0.1 && at < f.b - 0.1
    }

    private fun glassEvidence(f: StoreFixture, pts: List<Pt>): Int {
        if (f.kind != "case" && f.kind != "wall") return 0
        val scattered = pts.filter { p -> p.cameras.any { glassHit(p, it, f) } }
        if (scattered.size < max(8, pts.size / 100)) return 0
        val spread = scattered.maxOf { it.p[f.axis] } - scattered.minOf { it.p[f.axis] }
        return scattered.size.takeIf { spread > 0.6 } ?: 0
    }

    private fun glassFixture(f: StoreFixture): StoreFixture {
        val sign = if (f.faces.containsKey("+")) 1 else -1
        val face = f.c + sign * f.d / 2
        return StoreFixture("glass", f.axis, face - sign * CASE_DEPTH / 2, f.a, f.b, CASE_DEPTH,
            HEIGHT.getValue("glass"), f.n, f.w, f.faces, f.themes, f.moduleWidths)
    }

    /** Reject occupied walkways, see-through shelves, back-side observations, and products in the rack core. */
    private fun valid(f: StoreFixture, d: Draft, pts: List<Pt>, frameTrail: List<DoubleArray>): Boolean {
        val r = rect(f)
        if (frameTrail.any { crosses(it, it, r, 0.06) } ||
            frameTrail.zipWithNext().any { (a, b) -> crosses(a, b, r, 0.06) }) return false

        var wrongSide = 0; var views = 0
        for (face in d.faces) for (p in face.pts) for (cam in p.cameras) {
            views++
            if ((cam[face.ax] - face.c) * face.s < -0.1) wrongSide++
        }
        if (views >= 4 && wrongSide * 4 > views) return false

        val own = d.faces.flatMap { it.pts }.toHashSet()
        val core = pts.count { p -> p !in own &&
            p.p[0] > r.u0 + 0.15 && p.p[0] < r.u1 - 0.15 &&
            p.p[1] > r.v0 + 0.15 && p.p[1] < r.v1 - 0.15 }
        if (f.kind != "glass" && core > max(6, pts.size / 50)) return false

        val across = f.axis
        var blocked = 0
        for (p in pts) {
            val target = p.p
            for (cam in p.cameras) {
                val through = (cam[across] < f.c - f.d / 2 - 0.2 && target[across] > f.c + f.d / 2 + 0.2) ||
                    (cam[across] > f.c + f.d / 2 + 0.2 && target[across] < f.c - f.d / 2 - 0.2)
                val explainedByGlass = f.kind == "glass" &&
                    (p.cats.isEmpty() || p.cats.keys.any { genre[it] in CHILLED }) && glassHit(p, cam, f)
                if (through && !explainedByGlass && crosses(cam, target, r, 0.1)) {
                    blocked++
                    break
                }
            }
            if (blocked > max(4, pts.size / 100)) return false
        }
        return true
    }

    private fun compatible(f: StoreFixture, accepted: List<StoreFixture>): Boolean {
        val r = rect(f)
        if (accepted.any { other ->
                val q = rect(other)
                min(r.u1, q.u1) - max(r.u0, q.u0) > 0.15 &&
                    min(r.v1, q.v1) - max(r.v0, q.v0) > 0.15
            }) return false
        // End caps may join one island; joining two rows would close an aisle.
        if (f.kind == "end" && accepted.count { other ->
                if (other.kind != "island") false else {
                    val q = rect(other)
                    r.u0 <= q.u1 + 0.15 && r.u1 >= q.u0 - 0.15 &&
                        r.v0 <= q.v1 + 0.15 && r.v1 >= q.v0 - 0.15
                }
            } != 1) return false
        return true
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
        val runs = merge(runsOf(pts))
        val drafts = classify(runs, frameTrail)
        val fixtures = ArrayList<StoreFixture>()
        val candidates = drafts.map { d ->
            val base = build(d)
            val glass = glassEvidence(base, pts)
            val f = if (glass > 0) glassFixture(base) else base
            val support = d.faces.flatMap { it.pts }.toSet().size
            Candidate(d, f, support + 12.0 * cornerSupport(d, drafts) + 2.0 * glass - 16.0 - 3.0 * f.n)
        }.sortedWith(compareByDescending<Candidate> { it.score }.thenBy { it.fixture.kind == "end" }.thenBy { it.fixture.c })
        for (candidate in candidates) {
            val f = candidate.fixture
            if (candidate.score > 0 && valid(f, candidate.draft, pts, frameTrail) && compatible(f, fixtures)) fixtures += f
        }
        register(cs, sn)?.let { fixtures += it }
        val sorted = fixtures.sortedWith(compareBy({ it.kind }, { it.axis }, { it.c }, { it.a }))
        return StoreView(yaw, roomOf(sorted, frameTrail), sorted)
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
        private val HEIGHT = mapOf("island" to 1.8, "wall" to 1.8, "case" to 2.0, "glass" to 2.0, "end" to 1.5, "register" to 1.0)
        private val CHILLED = setOf("ソフトドリンク", "栄養・エナジードリンク", "紙パック飲料・乳製品", "酒", "冷凍食品", "アイス", "弁当・おにぎり", "サンドイッチ・サラダ・惣菜", "チルドデザート", "飲料", "お酒", "パン・米飯・軽食", "冷蔵・冷凍・デザート・惣菜")

        private fun floorI(v: Double) = floor(v).toInt()
        private fun overlap(a0: Double, a1: Double, b0: Double, b1: Double) = min(a1, b1) - max(a0, b0)

        private fun median(values: List<Pair<Double, Double>>): Double {
            val v = values.sortedWith(compareBy({ it.first }, { it.second }))
            val half = v.sumOf { it.second } / 2
            var s = 0.0
            for ((x, w) in v) { s += w; if (s >= half) return x }
            return v.last().first
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
