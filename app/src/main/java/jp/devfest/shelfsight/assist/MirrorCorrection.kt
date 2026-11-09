package jp.devfest.shelfsight.assist

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * RANSAC の前に、ガラス越しに奥行きを誤った点を直す。同じ視点から同じ方位を見ているのに奥行きが大きくばらつくなら、
 * そこはガラス面なので、各点を視線に沿ってばらつきの手前の端へ戻す。斜めに見た棚は方位につれて奥行きが素直に変わるだけなので、
 * 方位に対する傾きを除いた残差でばらつきを測る。
 */
internal object MirrorCorrection {
    private const val VIEWPOINT_M = 0.35
    private val SECTOR_RAD = Math.toRadians(10.0)
    private const val MIN_RAYS = 6
    private const val STD_LIMIT_M = 2.0   // 奥行きの標準偏差がこれ未満なら、棚の帯や通路越しの見え方として補正しない
    private const val NEAR_QUANTILE = 0.15
    private const val MIN_DEPTH_M = 0.2

    /** @param points 補正後の全ての点 (x, z)。@param moved 動かした点の (元 x, 元 z, 新 x, 新 z)。 */
    class Result(val points: List<DoubleArray>, val moved: List<DoubleArray>)

    private class Ray(val index: Int, val bearing: Double, val depth: Double)

    /** @param rays 観測ごとの (カメラ x, カメラ z, 点 x, 点 z)。 */
    fun correct(rays: List<DoubleArray>): Result {
        val points = rays.mapTo(ArrayList(rays.size)) { doubleArrayOf(it[2], it[3]) }
        val moved = ArrayList<DoubleArray>()
        val viewpoints = rays.indices.groupBy { floor(rays[it][0] / VIEWPOINT_M) to floor(rays[it][1] / VIEWPOINT_M) }
        for (viewpoint in viewpoints.values) {
            val sectors = viewpoint.map { i ->
                val r = rays[i]
                Ray(i, atan2(r[3] - r[1], r[2] - r[0]), hypot(r[2] - r[0], r[3] - r[1]))
            }.groupBy { floor(it.bearing / SECTOR_RAD) }
            for (sector in sectors.values) {
                if (sector.size < MIN_RAYS) continue
                val meanBearing = sector.sumOf { it.bearing } / sector.size
                val meanDepth = sector.sumOf { it.depth } / sector.size
                val spreadB = sector.sumOf { (it.bearing - meanBearing) * (it.bearing - meanBearing) }
                val slope = if (spreadB > 1e-9) sector.sumOf { (it.bearing - meanBearing) * (it.depth - meanDepth) } / spreadB else 0.0
                fun trend(r: Ray) = meanDepth + slope * (r.bearing - meanBearing)
                val residuals = sector.map { it.depth - trend(it) }.sorted()
                fun quantile(q: Double) = residuals[min(residuals.size - 1, (q * residuals.size).toInt())]
                if (sqrt(residuals.sumOf { it * it } / residuals.size) < STD_LIMIT_M) continue
                val near = quantile(NEAR_QUANTILE)
                for (r in sector) {
                    val depth = trend(r) + near
                    if (r.depth - trend(r) <= near || depth <= MIN_DEPTH_M) continue
                    val cam = rays[r.index]
                    val x = cam[0] + depth * cos(r.bearing)
                    val z = cam[1] + depth * sin(r.bearing)
                    moved += doubleArrayOf(cam[2], cam[3], x, z)
                    points[r.index] = doubleArrayOf(x, z)
                }
            }
        }
        return Result(points, moved)
    }

    /** RANSAC の計算量を抑えるため、10 cm の格子ごとの平均にする。 */
    fun cellMeans(points: List<DoubleArray>): List<DoubleArray> =
        points.groupBy { floor(it[0] / 0.1) to floor(it[1] / 0.1) }.values
            .map { g -> doubleArrayOf(g.sumOf { it[0] } / g.size, g.sumOf { it[1] } / g.size) }
}
