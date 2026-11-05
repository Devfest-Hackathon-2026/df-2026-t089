package jp.devfest.shelfsight

import jp.devfest.shelfsight.model.ProductCategory
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.scan.LabelMerging
import jp.devfest.shelfsight.scan.ScanGate
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ScanLogicTest {
    private val origin = Vec3(0f, 0f, 0f)
    private val forward = Vec3(0f, 0f, -1f)

    @Test
    fun `still camera does not trigger a scan`() {
        assertFalse(ScanGate.hasMovedEnough(origin, forward, Vec3(0.01f, 0f, 0f), forward))
    }

    @Test
    fun `walking triggers a scan`() {
        assertTrue(ScanGate.hasMovedEnough(origin, forward, Vec3(0.3f, 0f, 0f), forward))
    }

    @Test
    fun `turning triggers a scan including a full reversal`() {
        assertTrue(ScanGate.hasMovedEnough(origin, forward, origin, Vec3(0.5f, 0f, -0.866f)))
        assertTrue(ScanGate.hasMovedEnough(origin, forward, origin, Vec3(0f, 0f, 1f)))
    }

    private data class Item(val category: ProductCategory, val position: Vec3)

    private fun duplicate(category: ProductCategory, position: Vec3, items: List<Item>) =
        LabelMerging.isDuplicate(category, position, items, { it.category }, { it.position })

    @Test
    fun `same category nearby is a duplicate but other category or far is not`() {
        val items = listOf(Item(ProductCategory.COFFEE, origin))
        assertTrue(duplicate(ProductCategory.COFFEE, Vec3(0.1f, 0f, 0f), items))
        assertFalse(duplicate(ProductCategory.TEA, Vec3(0.1f, 0f, 0f), items))
        assertFalse(duplicate(ProductCategory.COFFEE, Vec3(2f, 0f, 0f), items))
    }
}
