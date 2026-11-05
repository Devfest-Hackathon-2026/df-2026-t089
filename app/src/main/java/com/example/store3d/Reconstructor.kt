package com.example.store3d

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/** ARCore plane snapshot. poly = world xyz triples, normal = plane +Y axis (points out of the surface). */
class PlaneSnap(val vertical: Boolean, val poly: FloatArray, val normal: FloatArray)

/** World-space ray (d normalized). */
class Ray(val o: FloatArray, val d: FloatArray)
class LabelRay(val ray: Ray, val probs: FloatArray)

enum class Kind { SHELF, FREEZER, FLOOR }

/** size = (along face, height, depth); local +z = shelf front, rotated by yaw about world Y. */
class Box(val center: FloatArray, val size: FloatArray, val yaw: Float, val label: String, val kind: Kind)

const val UNKNOWN = "不明"

/**
 * Turns noisy planes + per-frame recognition rays into a clean, axis-aligned store model:
 * 1. dominant store orientation from vertical plane normals (Manhattan world, 90° multiples)
 * 2. vertical planes snapped to that grid and merged into long shelf faces
 * 3. image-detected shelf boundaries projected onto faces; only boundaries seen in >= 2 frames survive
 * 4. each unit gets the label with the highest summed classifier probability
 * 5. horizontal planes at waist height become freezer/flat cases
 */
object Reconstructor {
    const val DEPTH = 0.45f // shortcut: fixed gondola depth, measure from back-side planes if needed
    const val MERGE_DIST = 0.15f
    const val MERGE_GAP = 0.3f
    const val MIN_FACE = 0.4f
    const val MIN_UNIT = 0.4f
    const val MAX_UNIT = 2.2f
    const val MIN_SUPPORT = 2
    const val CUT_CLUSTER = 0.15f
    const val MAX_RANGE = 6f
    private const val SNAP = (20 * PI / 180).toFloat()

    private class Seg(val k: Int, var d: Float, var s0: Float, var s1: Float, var yMax: Float, var w: Float)
    private class Rect(var u0: Float, var u1: Float, var w0: Float, var w1: Float, var h: Float)

    fun build(planes: List<PlaneSnap>, labelRays: List<LabelRay>, boundaryRays: List<Ray>, labels: List<String>): List<Box> {
        if (planes.isEmpty()) return emptyList()
        val verts = planes.filter { it.vertical }
        val horiz = planes.filter { !it.vertical }
        val floor = horiz.filter { polyArea(it.poly) >= 0.5f }.minOfOrNull { it.poly[1] }
            ?: planes.minOf { p -> (1 until p.poly.size step 3).minOf { p.poly[it] } }

        var sx = 0.0; var cx = 0.0
        for (p in verts) {
            val a = atan2(p.normal[0], p.normal[2]); val w = polyArea(p.poly)
            sx += w * sin(4.0 * a); cx += w * cos(4.0 * a)
        }
        val phi = (atan2(sx, cx) / 4).toFloat()

        val segs = verts.mapNotNull { p ->
            val a = atan2(p.normal[0], p.normal[2]) - phi
            val k = Math.floorMod((a / (PI / 2)).roundToInt(), 4)
            if (abs(Math.IEEEremainder(a - k * PI / 2, 2 * PI)) > SNAP) return@mapNotNull null
            val n = axisN(phi, k); val t = axisT(phi, k)
            var d = 0f; var s0 = Float.MAX_VALUE; var s1 = -Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
            val m = p.poly.size / 3
            for (i in 0 until m) {
                d += dot(p.poly, i * 3, n) / m
                val s = dot(p.poly, i * 3, t); s0 = min(s0, s); s1 = max(s1, s)
                y1 = max(y1, p.poly[i * 3 + 1])
            }
            Seg(k, d, s0, s1, y1, polyArea(p.poly))
        }.toMutableList()
        mergeSegs(segs)
        segs.removeAll { it.s1 - it.s0 < MIN_FACE }

        val t0 = axisT(phi, 0); val n0 = axisN(phi, 0)
        val rects = horiz.mapNotNull { p ->
            val h = p.poly[1] - floor
            if (h !in 0.5f..1.3f || polyArea(p.poly) < 0.15f) return@mapNotNull null
            val m = p.poly.size / 3
            val us = (0 until m).map { dot(p.poly, it * 3, t0) }; val ws = (0 until m).map { dot(p.poly, it * 3, n0) }
            Rect(us.min(), us.max(), ws.min(), ws.max(), h)
        }.toMutableList()
        mergeRects(rects)
        rects.removeAll { (it.u1 - it.u0) * (it.w1 - it.w0) < 0.2f }
        // freezer fronts show up as low vertical planes; the freezer box already covers them
        segs.removeAll { s ->
            val mid = (s.s0 + s.s1) / 2; val n = axisN(phi, s.k); val t = axisT(phi, s.k)
            val p = floatArrayOf(n[0] * s.d + t[0] * mid, 0f, n[2] * s.d + t[2] * mid)
            val u = dot(p, 0, t0); val w = dot(p, 0, n0)
            s.yMax - floor < 1.3f && rects.any { u in it.u0 - 0.3f..it.u1 + 0.3f && w in it.w0 - 0.3f..it.w1 + 0.3f }
        }

        fun height(s: Seg) = (s.yMax - floor).coerceIn(1.2f, 2.2f)

        // returns (target, coord): target < segs.size -> face index with along-face coord, else freezer index
        fun cast(r: Ray): Pair<Int, Float>? {
            var best = MAX_RANGE; var hit: Pair<Int, Float>? = null
            segs.forEachIndexed { i, s ->
                val n = axisN(phi, s.k); val dn = dot(r.d, 0, n)
                if (dn > -0.05f) return@forEachIndexed
                val tt = (s.d - dot(r.o, 0, n)) / dn
                if (tt < 0.1f || tt >= best) return@forEachIndexed
                val p = FloatArray(3) { r.o[it] + r.d[it] * tt }
                val sc = dot(p, 0, axisT(phi, s.k)); val y = p[1] - floor
                if (sc in s.s0 - 0.1f..s.s1 + 0.1f && y in -0.1f..height(s) + 0.3f) { best = tt; hit = i to sc }
            }
            if (r.d[1] < -0.05f) rects.forEachIndexed { i, q ->
                val tt = (floor + q.h - r.o[1]) / r.d[1]
                if (tt < 0.1f || tt >= best) return@forEachIndexed
                val p = FloatArray(3) { r.o[it] + r.d[it] * tt }
                if (dot(p, 0, t0) in q.u0..q.u1 && dot(p, 0, n0) in q.w0..q.w1) { best = tt; hit = segs.size + i to 0f }
            }
            return hit
        }

        val cuts = Array(segs.size) { mutableListOf<Float>() }
        for (r in boundaryRays) cast(r)?.let { (i, s) -> if (i < segs.size) cuts[i].add(s) }
        val obs = Array(segs.size + rects.size) { mutableListOf<Pair<Float, FloatArray>>() }
        for (lr in labelRays) cast(lr.ray)?.let { (i, s) -> obs[i].add(s to lr.probs) }

        fun label(o: List<Pair<Float, FloatArray>>, a: Float, b: Float): String {
            val sum = FloatArray(labels.size); var cnt = 0
            for ((s, p) in o) if (s in a..b) { cnt++; for (j in sum.indices) sum[j] += p[j] }
            if (cnt == 0) return UNKNOWN
            val best = sum.indices.maxBy { sum[it] }
            return if (sum[best] / cnt < 0.25f) UNKNOWN else labels[best]
        }

        val boxes = mutableListOf<Box>()
        segs.forEachIndexed { i, s ->
            val edges = listOf(s.s0) + clusterCuts(cuts[i], s.s0, s.s1) + listOf(s.s1)
            val n = axisN(phi, s.k); val t = axisT(phi, s.k); val h = height(s)
            for (e in 0 until edges.size - 1) {
                val pieces = max(1, ceil((edges[e + 1] - edges[e]) / MAX_UNIT).toInt())
                val len = (edges[e + 1] - edges[e]) / pieces
                for (q in 0 until pieces) {
                    val a = edges[e] + q * len; val b = a + len; val m = (a + b) / 2
                    val c = floatArrayOf(n[0] * (s.d - DEPTH / 2) + t[0] * m, floor + h / 2, n[2] * (s.d - DEPTH / 2) + t[2] * m)
                    boxes += Box(c, floatArrayOf(len - 0.04f, h, DEPTH), phi + s.k * (PI / 2).toFloat(), label(obs[i], a, b), Kind.SHELF)
                }
            }
        }
        // shortcut: freezer cases are not split by image boundaries (top-down photos have no dividers), add if needed
        rects.forEachIndexed { i, q ->
            val u = (q.u0 + q.u1) / 2; val w = (q.w0 + q.w1) / 2
            val c = floatArrayOf(t0[0] * u + n0[0] * w, floor + q.h / 2, t0[2] * u + n0[2] * w)
            boxes += Box(c, floatArrayOf(q.u1 - q.u0, q.h, q.w1 - q.w0), phi, label(obs[segs.size + i], -1f, 1f), Kind.FREEZER)
        }
        if (boxes.isNotEmpty()) {
            var u0 = Float.MAX_VALUE; var u1 = -Float.MAX_VALUE; var w0 = Float.MAX_VALUE; var w1 = -Float.MAX_VALUE
            for (b in boxes) {
                val r = max(b.size[0], b.size[2]) / 2
                val u = dot(b.center, 0, t0); val w = dot(b.center, 0, n0)
                u0 = min(u0, u - r); u1 = max(u1, u + r); w0 = min(w0, w - r); w1 = max(w1, w + r)
            }
            u0 -= 0.6f; u1 += 0.6f; w0 -= 0.6f; w1 += 0.6f
            val u = (u0 + u1) / 2; val w = (w0 + w1) / 2
            boxes += Box(floatArrayOf(t0[0] * u + n0[0] * w, floor - 0.01f, t0[2] * u + n0[2] * w),
                floatArrayOf(u1 - u0, 0.02f, w1 - w0), phi, "床", Kind.FLOOR)
        }
        return boxes
    }

    /** 1-D clustering of boundary hits; drops clusters seen fewer than MIN_SUPPORT times (noise). */
    fun clusterCuts(values: List<Float>, s0: Float, s1: Float): List<Float> {
        val v = values.sorted()
        val groups = mutableListOf<MutableList<Float>>()
        for (x in v) if (groups.isNotEmpty() && x - groups.last().last() < CUT_CLUSTER) groups.last() += x else groups += mutableListOf(x)
        val kept = mutableListOf<Float>()
        groups.filter { it.size >= MIN_SUPPORT }.sortedByDescending { it.size }.forEach { g ->
            val c = g[g.size / 2]
            if (c > s0 + MIN_UNIT && c < s1 - MIN_UNIT && kept.none { abs(it - c) < MIN_UNIT }) kept += c
        }
        return kept.sorted()
    }

    private fun mergeSegs(segs: MutableList<Seg>) {
        outer@ while (true) {
            for (i in segs.indices) for (j in i + 1 until segs.size) {
                val a = segs[i]; val b = segs[j]
                if (a.k != b.k || abs(a.d - b.d) > MERGE_DIST || max(a.s0, b.s0) - min(a.s1, b.s1) > MERGE_GAP) continue
                a.d = (a.d * a.w + b.d * b.w) / (a.w + b.w)
                a.s0 = min(a.s0, b.s0); a.s1 = max(a.s1, b.s1); a.yMax = max(a.yMax, b.yMax); a.w += b.w
                segs.removeAt(j)
                continue@outer
            }
            return
        }
    }

    private fun mergeRects(rects: MutableList<Rect>) {
        outer@ while (true) {
            for (i in rects.indices) for (j in i + 1 until rects.size) {
                val a = rects[i]; val b = rects[j]
                if (abs(a.h - b.h) > 0.2f || a.u0 > b.u1 + 0.1f || b.u0 > a.u1 + 0.1f || a.w0 > b.w1 + 0.1f || b.w0 > a.w1 + 0.1f) continue
                a.u0 = min(a.u0, b.u0); a.u1 = max(a.u1, b.u1); a.w0 = min(a.w0, b.w0); a.w1 = max(a.w1, b.w1); a.h = max(a.h, b.h)
                rects.removeAt(j)
                continue@outer
            }
            return
        }
    }

    fun axisN(phi: Float, k: Int): FloatArray { val a = phi + k * PI / 2; return floatArrayOf(sin(a).toFloat(), 0f, cos(a).toFloat()) }
    fun axisT(phi: Float, k: Int): FloatArray { val a = phi + k * PI / 2; return floatArrayOf(cos(a).toFloat(), 0f, -sin(a).toFloat()) }
    private fun dot(p: FloatArray, o: Int, v: FloatArray) = p[o] * v[0] + p[o + 1] * v[1] + p[o + 2] * v[2]

    fun polyArea(p: FloatArray): Float {
        var x = 0f; var y = 0f; var z = 0f; val m = p.size / 3
        for (i in 0 until m) {
            val a = i * 3; val b = ((i + 1) % m) * 3
            x += p[a + 1] * p[b + 2] - p[a + 2] * p[b + 1]
            y += p[a + 2] * p[b] - p[a] * p[b + 2]
            z += p[a] * p[b + 1] - p[a + 1] * p[b]
        }
        return 0.5f * sqrt(x * x + y * y + z * z)
    }
}
