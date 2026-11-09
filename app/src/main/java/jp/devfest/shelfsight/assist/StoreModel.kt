package jp.devfest.shelfsight.assist

import java.util.Locale
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/** 什器 1 つ（記述言語の 1 行）。[axis] は正面が向く軸（0 = u, 1 = v）。[faces] は "+" / "-" の面ごと・モジュールごとの主な商品。 */
class StoreFixture(
    val kind: String, val axis: Int, val c: Double, val a: Double, val b: Double, val d: Double, val h: Double,
    val n: Int, val w: Double, val faces: Map<String, List<List<Pair<String, Double>>>>,
)

/** 推定した店。[room] は (u0, v0, u1, v1)。[yaw] は世界 → 店の座標の回転（ラジアン）。 */
class StoreView(val yaw: Double, val room: DoubleArray?, val fixtures: List<StoreFixture>)

/**
 * コンビニ記述言語のモデルを、「商品＋位置」の連続した観測から推定する。products/store_model.py の移植で、
 * 記述言語の定義と推定の手順はそちらに書いてある。同じ入力で text() が一致することを StoreModelTest で確かめる。
 * 1 つのスレッドから使う。
 */
class StoreModel {
    private class Cell(var w: Double = 0.0, var sx: Double = 0.0, var sz: Double = 0.0, var nx: Double = 0.0, var nz: Double = 0.0, val cats: LinkedHashMap<String, Double> = LinkedHashMap())
    private class Pt(val p: DoubleArray, val w: Double, val nu: Double, val nv: Double, val cats: Map<String, Double>)
    private class Run(val ax: Int, var c: Double, val a: Double, var b: Double, val s: Int, var w: Double, var pts: List<Pt>)
    private class Draft(val kind: String, val axis: Int, val c: Double, val a: Double, val b: Double, val d: Double, val faces: List<Run>)

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

    private fun classify(runs: List<Run>, frameTrail: List<DoubleArray>): List<Draft> {
        val out = ArrayList<Draft>(); val islands = ArrayList<Draft>(); val used = HashSet<Run>()
        for (A in runs.filter { it.s == 1 && it.b - it.a >= 0.8 }.sortedBy { it.a - it.b }) {
            if (A in used) continue
            val bs = runs.filter { it !in used && it.ax == A.ax && it.s == -1 && A.c - it.c >= 0.5 && A.c - it.c <= 1.3 &&
                overlap(A.a, A.b, it.a, it.b) > 0.3 * min(A.b - A.a, it.b - it.a) }
            if (bs.isNotEmpty()) {
                used += A; used += bs
                val c = (A.c + bs.sumOf { it.c } / bs.size) / 2
                islands += Draft("island", A.ax, c, min(A.a, bs.minOf { it.a }), max(A.b, bs.maxOf { it.b }), ISLAND_DEPTH, listOf(A) + bs)
            }
        }
        fun takeEnds() {
            for (r in runs) {
                if (r in used || r.b - r.a > 1.5) continue
                val mid = (r.a + r.b) / 2
                for (i in islands) {
                    val atB = r.s == 1 && r.c >= i.b - 0.2 && r.c <= i.b + 0.8
                    val atA = r.s == -1 && r.c >= i.a - 0.8 && r.c <= i.a + 0.2
                    if (r.ax == 1 - i.axis && abs(mid - i.c) < 0.6 && (atA || atB)) {
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

    private fun build(d: Draft): StoreFixture {
        val mid = (d.a + d.b) / 2; val length = d.b - d.a
        var n = 1; var w = 0.9
        if (d.kind != "end") {
            var bestErr = Double.MAX_VALUE
            for (cand in WIDTHS) {
                val k = max(1, Math.rint(length / cand).toInt())
                val err = abs(k * cand - length)
                if (err < bestErr) { bestErr = err; n = k; w = cand }
            }
        }
        val a = if (d.kind == "end") mid - 0.45 else mid - n * w / 2
        val faces = LinkedHashMap<String, List<List<Pair<String, Double>>>>()
        for (sign in listOf(1, -1)) {
            val sides = d.faces.filter { it.s == sign }
            if (sides.isEmpty()) continue
            val mods = List(n) { LinkedHashMap<String, Double>() }
            for (r in sides) for (p in r.pts) {
                val m = mods[min(n - 1, max(0, floorI((p.p[1 - r.ax] - a) / w)))]
                for ((cat, v) in p.cats) m[cat] = (m[cat] ?: 0.0) + v
            }
            faces[if (sign == 1) "+" else "-"] = mods.map { m ->
                val total = m.values.sum().let { if (it == 0.0) 1.0 else it }
                m.entries.sortedWith(compareBy({ -it.value }, { it.key })).take(3).map { it.key to it.value / total }
            }
        }
        return StoreFixture(d.kind, d.axis, d.c, a, a + n * w, d.d, HEIGHT.getValue(d.kind), n, w, faces)
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
            Pt(doubleArrayOf(x * cs + z * sn, -x * sn + z * cs), c.w, c.nx * cs + c.nz * sn, -c.nx * sn + c.nz * cs, c.cats)
        }
        val frameTrail = trail.map { doubleArrayOf(it[0] * cs + it[1] * sn, -it[0] * sn + it[1] * cs) }
        val runs = merge(runsOf(pts))
        val fixtures = classify(runs, frameTrail).map(::build).toMutableList()
        register(cs, sn)?.let { fixtures += it }
        val sorted = fixtures.sortedWith(compareBy({ it.kind }, { it.axis }, { it.c }, { it.a }))
        return StoreView(yaw, roomOf(sorted, frameTrail), sorted)
    }

    companion object {
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
        private val WIDTHS = listOf(0.9, 1.2)
        private val HEIGHT = mapOf("island" to 1.8, "wall" to 1.8, "case" to 2.0, "end" to 1.5, "register" to 1.0)
        private val CHILLED = setOf("ソフトドリンク", "栄養・エナジードリンク", "紙パック飲料・乳製品", "酒", "冷凍食品", "アイス", "弁当・おにぎり", "サンドイッチ・サラダ・惣菜", "チルドデザート")

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

        /** 記述言語の文字列にする。 */
        fun text(m: StoreView): String {
            if (m.fixtures.isEmpty()) return "store (empty)"
            val r = m.room!!
            return buildString {
                append("store yaw=${fmt(m.yaw * 180 / Math.PI, 1)} room=${fmt(r[0])},${fmt(r[1])}..${fmt(r[2])},${fmt(r[3])}")
                for (f in m.fixtures) {
                    append("\n${f.kind} axis=${"uv"[f.axis]} c=${fmt(f.c)} span=${fmt(f.a)}..${fmt(f.b)} d=${fmt(f.d)} h=${fmt(f.h, 1)} mods=${f.n}x${fmt(f.w)}")
                    for (i in 0 until f.n) {
                        val parts = f.faces.entries.sortedBy { it.key }.filter { i < it.value.size && it.value[i].isNotEmpty() }
                            .map { (sign, faces) -> sign + faces[i].joinToString(",") { (c, s) -> "$c:${fmt(s)}" } }
                        if (parts.isNotEmpty()) append("\n  m$i ${parts.joinToString(" ")}")
                    }
                }
            }
        }
    }
}
