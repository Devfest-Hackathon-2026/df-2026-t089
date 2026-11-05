package jp.devfest.shelfsight

import jp.devfest.shelfsight.ocr.CloudVisionParser
import jp.devfest.shelfsight.ocr.ProductReading
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CloudVisionParserTest {
    /** 文字 1 つ分の JSON。y0 から y1 の高さで、break があれば改行や空白を表す。 */
    private fun symbol(text: String, x: Int, y0: Int, y1: Int, brk: String? = null): String {
        val property = brk?.let { """, "property": {"detectedBreak": {"type": "$it"}}""" }.orEmpty()
        return """{"text": "$text", "boundingBox": {"vertices": [{"x": $x, "y": $y0}, {"x": ${x + 10}, "y": $y0},
            {"x": ${x + 10}, "y": $y1}, {"x": $x, "y": $y1}]}$property}"""
    }

    private fun response(vararg words: List<String>): String {
        val ws = words.joinToString(",") { """{"symbols": [${it.joinToString(",")}]}""" }
        return """{"responses": [{"fullTextAnnotation": {"pages": [{"blocks": [{"paragraphs": [{"words": [$ws]}]}]}]}}]}"""
    }

    @Test
    fun splitsLinesAtLineBreaksAndUsesTheMedianSymbolHeight() {
        val json = response(
            listOf(symbol("ぶ", 0, 100, 200), symbol("ど", 10, 100, 200), symbol("う", 20, 90, 260, "SPACE")),
            listOf(symbol("バ", 30, 100, 200), symbol("ー", 40, 100, 200, "LINE_BREAK")),
            listOf(symbol("1", 0, 400, 420), symbol("0", 10, 400, 420), symbol("本", 20, 400, 420, "EOL_SURE_SPACE")),
        )
        val lines = CloudVisionParser.lines(json, 1000)
        assertEquals(listOf("ぶどう バー", "10本"), lines.map { it.text })
        assertEquals(0.1f, lines[0].height, 1e-6f) // 1 文字だけ背の高い「う」に引きずられない
        assertEquals(0.09f, lines[0].top, 1e-6f)
        assertEquals(0.02f, lines[1].height, 1e-6f)
    }

    @Test
    fun theLargestLineBecomesTheProductName() {
        val json = response(
            listOf(symbol("L", 0, 0, 30, "LINE_BREAK")),
            listOf(symbol("フ", 0, 100, 200), symbol("ル", 10, 100, 200), symbol("ー", 20, 100, 200), symbol("ツ", 30, 100, 200, "LINE_BREAK")),
            listOf(symbol("内", 0, 900, 915), symbol("容", 10, 900, 915), symbol("量", 20, 900, 915, "LINE_BREAK")),
        )
        assertEquals("フルーツ", ProductReading.from(CloudVisionParser.lines(json, 1000)).name)
    }

    @Test
    fun noTextGivesNoLines() {
        assertTrue(CloudVisionParser.lines("""{"responses": [{}]}""", 1000).isEmpty())
    }

    @Test(expected = IllegalStateException::class)
    fun anErrorInTheResponseIsReported() {
        CloudVisionParser.lines("""{"responses": [{"error": {"code": 7, "message": "billing"}}]}""", 1000)
    }
}
