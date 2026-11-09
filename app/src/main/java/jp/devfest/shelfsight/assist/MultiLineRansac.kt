package jp.devfest.shelfsight.assist

import java.util.Random
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * 局所 RANSAC。各点の周囲だけを見て 1 m 未満の短い線分を当て、内点を取り除いて次の点へ進む。
 * 店の棚は互いに平行か垂直なので、線の向きは店の 2 つの軸だけを試す。
 */
internal object MultiLineRansac {
    const val MAX_LINES = 256
    private const val ITERATIONS = 30
    private const val RADIUS_M = 0.5   // 直径が線分の上限の 1 m になる
    private const val INLIER_M = 0.1
    private const val MIN_INLIERS = 6
    private const val MIN_INLIER_SHARE = 0.35   // 周囲の点の大半が線に乗らないなら、ただの散らばり
    private const val GAP_M = 0.3
    private const val MIN_LENGTH_M = 0.25
    private const val MAX_LENGTH_M = 1.0

    /**
     * @param points (x, z)
     * @param yaw 世界 → 店の座標の回転（ラジアン）。線は店の u 軸か v 軸に平行になる。
     * @return 線分 (x0, z0, x1, z1) の一覧
     */
    fun fit(points: List<DoubleArray>, yaw: Double, seed: Long = 1L): List<DoubleArray> {
        val rng = Random(seed)
        val axes = listOf(cos(yaw) to sin(yaw), -sin(yaw) to cos(yaw))
        val out = ArrayList<DoubleArray>()
        val used = HashSet<DoubleArray>()
        for (center in points.shuffled(rng)) {
            if (out.size == MAX_LINES) break
            if (center in used) continue
            val local = points.filter { it !in used && hypot(it[0] - center[0], it[1] - center[1]) <= RADIUS_M }
            if (local.size < MIN_INLIERS) continue
            var best = emptyList<DoubleArray>()
            var bestAxis = axes[0]
            repeat(ITERATIONS) {
                val a = local[rng.nextInt(local.size)]
                for (axis in axes) {
                    val inliers = local.filter { abs((it[0] - a[0]) * -axis.second + (it[1] - a[1]) * axis.first) <= INLIER_M }
                    val run = longestRun(inliers, a, axis.first, axis.second)
                    if (run.size > best.size) { best = run; bestAxis = axis }
                }
            }
            if (best.size < MIN_INLIERS || best.size < local.size * MIN_INLIER_SHARE) continue
            segment(best, bestAxis.first, bestAxis.second)?.let { out += it }
            used += best
        }
        return out
    }

    /** 線に沿って途切れない、いちばん点の多い区間。 */
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

    /** 内点の重心を通り軸に平行な線分にする。長さが範囲の外なら null。 */
    private fun segment(points: List<DoubleArray>, dx: Double, dz: Double): DoubleArray? {
        val mx = points.sumOf { it[0] } / points.size
        val mz = points.sumOf { it[1] } / points.size
        val t = points.map { (it[0] - mx) * dx + (it[1] - mz) * dz }
        val lo = t.min()
        val hi = t.max()
        if (hi - lo < MIN_LENGTH_M || hi - lo >= MAX_LENGTH_M) return null
        return doubleArrayOf(mx + lo * dx, mz + lo * dz, mx + hi * dx, mz + hi * dz)
    }
}
