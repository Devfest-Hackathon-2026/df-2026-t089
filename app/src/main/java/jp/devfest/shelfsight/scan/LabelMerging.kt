package jp.devfest.shelfsight.scan

import jp.devfest.shelfsight.model.ProductCategory
import jp.devfest.shelfsight.model.Vec3

internal object LabelMerging {
    /** 同じ売り場を繰り返し映しても、ラベルが増え続けないようにするための距離。 */
    private const val MERGE_RADIUS_M = 0.5f

    fun <T> isDuplicate(
        category: ProductCategory,
        position: Vec3,
        existing: Iterable<T>,
        categoryOf: (T) -> ProductCategory,
        positionOf: (T) -> Vec3,
    ): Boolean = existing.any {
        categoryOf(it) == category && positionOf(it).distanceTo(position) < MERGE_RADIUS_M
    }
}
