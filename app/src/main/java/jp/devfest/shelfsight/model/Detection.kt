package jp.devfest.shelfsight.model

import kotlin.math.sqrt

/** 正立画像上の矩形。各値は左上原点で 0.0〜1.0 に正規化されている。 */
data class NormalizedBox(val left: Float, val top: Float, val right: Float, val bottom: Float)

/** 分類器が 1 枚の画像から見つけた棚の領域。 */
data class ShelfDetection(val category: ProductCategory, val box: NormalizedBox)

/** ワールド座標に復元した検出結果。 */
data class WorldDetection(
    val category: ProductCategory,
    val position: Vec3,
    /** 深度が取れず、仮の距離で置いた場合は false。 */
    val hasMeasuredDepth: Boolean,
)

data class Vec3(val x: Float, val y: Float, val z: Float) {
    fun distanceTo(other: Vec3): Float {
        val dx = x - other.x
        val dy = y - other.y
        val dz = z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
