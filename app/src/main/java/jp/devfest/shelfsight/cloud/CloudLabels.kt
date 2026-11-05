package jp.devfest.shelfsight.cloud

import jp.devfest.shelfsight.model.Vec3

enum class CloudConfidence { HIGH, MEDIUM, LOW }

/** 棚の 1 列。範囲は正立画像の横方向で 0〜1。 */
data class ShelfColumn(
    val xStart: Float,
    val xEnd: Float,
    /** labels.txt の 26 分類のどれか。 */
    val category: String,
    val confidence: CloudConfidence,
    /** 判断の根拠として読めた商品名や POP。 */
    val evidence: String,
)

data class ShelfColumnsResult(val isShelfFront: Boolean, val columns: List<ShelfColumn>)

/** 地図に置くための、3D 位置つきの列。 */
data class PlacedShelfColumn(
    val column: ShelfColumn,
    val position: Vec3,
    /** 深度が取れず、仮の距離で置いた場合は false。 */
    val hasMeasuredDepth: Boolean,
)
