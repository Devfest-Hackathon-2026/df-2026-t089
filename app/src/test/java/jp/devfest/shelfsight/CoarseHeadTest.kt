package jp.devfest.shelfsight

import jp.devfest.shelfsight.walking.CoarseHead
import jp.devfest.shelfsight.walking.CropRect
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/** 期待値は tools/coarse_experiment.py と同じ計算を Python で行って求めたもの。端末と PC で判別が変わらないことを確かめる。 */
class CoarseHeadTest {
    private val classes = listOf("a", "b", "c", "d", "e", "f", "g")

    private fun head(weights: Array<FloatArray> = Array(7) { FloatArray(2) }, bias: FloatArray = FloatArray(7)) =
        CoarseHead("test", classes, weights, bias, columns = 8, windowColumns = 3, switchCost = 1.5f)

    @Test
    fun cropsAWindowThreeColumnsWideAndTakesTheCentreSquareOfAPortraitImage() {
        val expected = listOf(
            CropRect(0, 550, 180, 730), CropRect(0, 505, 270, 775), CropRect(90, 505, 360, 775), CropRect(180, 505, 450, 775),
            CropRect(270, 505, 540, 775), CropRect(360, 505, 630, 775), CropRect(450, 505, 720, 775), CropRect(540, 550, 720, 730),
        )
        assertEquals(expected, (0 until 8).map { head().cropFor(it, 720, 1280) })
    }

    @Test
    fun cropsTheCentreSquareWhenTheWindowIsWiderThanTall() {
        val expected = listOf(
            CropRect(10, 0, 310, 300), CropRect(90, 0, 390, 300), CropRect(250, 0, 550, 300), CropRect(410, 0, 710, 300),
            CropRect(570, 0, 870, 300), CropRect(730, 0, 1030, 300), CropRect(890, 0, 1190, 300), CropRect(970, 0, 1270, 300),
        )
        assertEquals(expected, (0 until 8).map { head().cropFor(it, 1280, 300) })
    }

    @Test
    fun smoothingRemovesSingleColumnSwitchesLikeThePcTool() {
        val probs = listOf(
            floatArrayOf(2.7969e-03f, 3.5236e-01f, 3.3781e-03f, 2.5427e-01f, 5.8357e-02f, 7.0703e-02f, 2.5813e-01f),
            floatArrayOf(3.1624e-02f, 2.1974e-01f, 8.6934e-07f, 3.4863e-02f, 6.2644e-01f, 8.6972e-02f, 3.6055e-04f),
            floatArrayOf(1.5846e-01f, 1.4141e-02f, 2.9286e-02f, 2.0757e-01f, 3.5567e-01f, 1.7437e-01f, 6.0506e-02f),
            floatArrayOf(2.7133e-01f, 2.2576e-01f, 1.0550e-01f, 2.5831e-01f, 5.9634e-02f, 7.3414e-02f, 6.0510e-03f),
            floatArrayOf(9.8665e-02f, 2.7588e-01f, 1.6634e-01f, 4.5150e-01f, 3.9862e-04f, 9.8180e-04f, 6.2309e-03f),
            floatArrayOf(3.1849e-03f, 4.8149e-02f, 6.6599e-01f, 9.0516e-03f, 1.0361e-01f, 7.1091e-02f, 9.8927e-02f),
            floatArrayOf(7.4096e-02f, 6.9838e-02f, 2.4328e-01f, 1.1609e-04f, 1.2728e-02f, 4.5654e-02f, 5.5429e-01f),
            floatArrayOf(2.2823e-02f, 5.6540e-02f, 6.0344e-02f, 4.9429e-01f, 1.3067e-01f, 1.9177e-01f, 4.3563e-02f),
        )
        // 平滑化しないと [1, 4, 4, 0, 3, 2, 6, 3]
        assertArrayEquals(intArrayOf(1, 4, 4, 3, 3, 2, 2, 3), head().smooth(probs))
    }

    @Test
    fun probabilitiesNormaliseTheEmbeddingFirst() {
        val weights = Array(7) { k -> if (k == 2) floatArrayOf(10f, 0f) else FloatArray(2) }
        val small = head(weights).probabilities(floatArrayOf(0.001f, 0f))
        val large = head(weights).probabilities(floatArrayOf(1000f, 0f))
        assertArrayEquals(small, large, 1e-6f)
        assertEquals(2, small.indices.maxBy { small[it] })
        assertEquals(1f, small.sum(), 1e-5f)
    }

    @Test
    fun parsesTheHeadWrittenByThePcTool() {
        val json = """{"model": "M/p", "columns": 8, "window_columns": 3, "classes": ["x", "y"],
            "weights": [[1, 2], [3, 4]], "bias": [0.5, -0.5], "switch_cost": 1.5, "note": ""}"""
        val head = CoarseHead.parse(json)
        assertEquals(listOf("x", "y"), head.classes)
        assertEquals(8, head.columns)
        assertEquals(2, head.embeddingSize)
        assertEquals("M/p", head.modelName)
    }
}
