package jp.devfest.shelfsight.assist

import java.util.Random
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/** 点の集まりに、直線を 1 本ずつ RANSAC で当てては内点を取り除く（逐次型の multi-model RANSAC）。向きは軸に揃えない。 */
internal object MultiLineRansac {
    const val MAX_LINES = 24
    private const val ITERATIONS = 400
    private const val INLIER_M = 0.1
    private const val MIN_INLIERS = 12
    private const val GAP_M = 0.6
    private const val MIN_LENGTH_M = 0.5

    /** @return 線分 (x0, z0, x1, z1) の一覧。内点の多い順。[points] は (x, z)。 */
    fun fit(points: List<DoubleArray>, seed: Long = 1L): List<DoubleArray> {
        val rng = Random(seed)
        val out = ArrayList<DoubleArray>()
        var left = points
        while (out.size < MAX_LINES && left.size >= MIN_INLIERS) {
            var best = emptyList<DoubleArray>()
            repeat(ITERATIONS) {
                val a = left[rng.nextInt(left.size)]
                val b = left[rng.nextInt(left.size)]
                val length = hypot(b[0] - a[0], b[1] - a[1])
                if (length < 0.2) return@repeat
                val dx = (b[0] - a[0]) / length
                val dz = (b[1] - a[1]) / length
                val inliers = left.filter { abs((it[0] - a[0]) * -dz + (it[1] - a[1]) * dx) <= INLIER_M }
                val run = longestRun(inliers, a, dx, dz)
                if (run.size > best.size) best = run
            }
            if (best.size < MIN_INLIERS) break
            segment(best)?.let { out += it }
            val used = best.toHashSet()
            left = left.filter { it !in used }
        }
        return out
    }

    /** 線に沿って途切れない、いちばん点の多い区間（離れた棚を 1 本の線にしない）。 */
    private fun longestRun(inliers: List<DoubleArray>, origin: DoubleArray, dx: Double, dz: Double): List<DoubleArray> {
        val sorted = inliers.sortedBy { (it[0] - origin[0]) * dx + (it[1] - origin[1]) * dz }
        var best = emptyList<DoubleArray>()
        var start = 0
        for (i in 1..sorted.size) {
            val gap = i == sorted.size || ((sorted[i][0] - sorted[i - 1][0]) * dx + (sorted[i][1] - sorted[i - 1][1]) * dz) > GAP_M
            if (gap) {
                if (i - start > best.size) best = sorted.subList(start, i)
                start = i
            }
        }
        return best
    }

    /** 内点に主成分で直線を当て直し、両端の線分にする。短すぎれば null。 */
    private fun segment(points: List<DoubleArray>): DoubleArray? {
        val mx = points.sumOf { it[0] } / points.size
        val mz = points.sumOf { it[1] } / points.size
        val sxx = points.sumOf { (it[0] - mx) * (it[0] - mx) }
        val szz = points.sumOf { (it[1] - mz) * (it[1] - mz) }
        val sxz = points.sumOf { (it[0] - mx) * (it[1] - mz) }
        val angle = 0.5 * atan2(2 * sxz, sxx - szz)
        val dx = cos(angle)
        val dz = sin(angle)
        val t = points.map { (it[0] - mx) * dx + (it[1] - mz) * dz }
        val lo = t.min()
        val hi = t.max()
        if (hi - lo < MIN_LENGTH_M) return null
        return doubleArrayOf(mx + lo * dx, mz + lo * dz, mx + hi * dx, mz + hi * dz)
    }
}
