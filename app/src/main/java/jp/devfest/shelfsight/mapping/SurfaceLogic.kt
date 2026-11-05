package jp.devfest.shelfsight.mapping

import kotlin.math.abs

/** 地図に描く面の種類。天井は地図を隠すので持たない。 */
internal enum class SurfaceKind { FLOOR, RAISED, WALL }

/** ARCore に依存しない面の判定。 */
internal object SurfaceLogic {
    /** これより小さい面は誤検出が多く、地図を見にくくするだけなので描かない。 */
    const val MIN_AREA_M2 = 0.05f

    /** 床の高さの基準にできる広さ。棚板のような狭い面を床と間違えないようにする。 */
    private const val MIN_FLOOR_AREA_M2 = 0.5f

    /** 床の検出誤差の許容範囲。これより高い水平面は棚やケースの上面として扱う。 */
    private const val FLOOR_TOLERANCE_M = 0.2f

    /** @param polygon 面のローカル座標 x, z の順に並んだ多角形の頂点 */
    fun polygonArea(polygon: FloatArray): Float {
        val count = polygon.size / 2
        var doubled = 0f
        for (i in 0 until count) {
            val j = (i + 1) % count
            doubled += polygon[i * 2] * polygon[j * 2 + 1] - polygon[j * 2] * polygon[i * 2 + 1]
        }
        return abs(doubled) / 2f
    }

    /** 十分に広い水平面のうち一番低いものを床とする。見つからなければ null。 */
    fun floorHeight(horizontals: List<Pair<Float, Float>>): Float? =
        horizontals.filter { (_, area) -> area >= MIN_FLOOR_AREA_M2 }.minOfOrNull { (y, _) -> y }

    /** @param floorY 床がまだ分からないときは null。そのとき水平面は床として扱う */
    fun horizontalKind(y: Float, floorY: Float?): SurfaceKind =
        if (floorY == null || y - floorY < FLOOR_TOLERANCE_M) SurfaceKind.FLOOR else SurfaceKind.RAISED
}
