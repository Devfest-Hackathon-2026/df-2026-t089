package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.ShelfSegment
import jp.devfest.shelfsight.assist.ShelfSpeech
import org.junit.Assert.assertEquals
import org.junit.Test

class ShelfSpeechTest {
    @Test
    fun `the shelf in the middle is in front, the others are left and right`() {
        val columns = listOf("飲み物・お酒", "飲み物・お酒", "お菓子・おつまみ", "お菓子・おつまみ", "お菓子・おつまみ", "お菓子・おつまみ", "食品", "食品")
        val segments = ShelfSegment.from(columns, FloatArray(8) { 0.9f })
        assertEquals("正面は、お菓子やおつまみの棚です。左は、飲み物やお酒。右は、食品。", ShelfSpeech.describe(segments))
    }

    @Test
    fun `one shelf, no shelf, and nothing in front`() {
        assertEquals("正面は、食品の棚です。", ShelfSpeech.describe(ShelfSegment.from(List(8) { "食品" }, FloatArray(8) { 0.9f })))
        assertEquals("棚が見えません。棚の方にスマホを向けてください。", ShelfSpeech.describe(emptyList()))
        val columns = listOf("食品", "食品", "商品なし", "商品なし", "商品なし", "商品なし", "商品なし", "商品なし")
        assertEquals("正面には棚が見えません。左は、食品。", ShelfSpeech.describe(ShelfSegment.from(columns, FloatArray(8) { 0.9f })))
    }
}
