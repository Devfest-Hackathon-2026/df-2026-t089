package jp.devfest.shelfsight

import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.ProductCategory
import jp.devfest.shelfsight.vision.GemmaResponseParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GemmaResponseParserTest {
    @Test
    fun `box order is ymin xmin ymax xmax`() {
        val result = GemmaResponseParser.parse("""[{"category":"coffee","box_2d":[100,200,500,800]}]""")
        assertEquals(ProductCategory.COFFEE, result.single().category)
        assertEquals(NormalizedBox(left = 0.2f, top = 0.1f, right = 0.8f, bottom = 0.5f), result.single().box)
    }

    @Test
    fun `extracts array from markdown fence and prose`() {
        val text = "Here you go:\n```json\n[{\"category\":\"alcohol\",\"box_2d\":[0,0,1000,1000]}]\n```\nDone."
        assertEquals(ProductCategory.ALCOHOL, GemmaResponseParser.parse(text).single().category)
    }

    @Test
    fun `unknown category becomes other`() {
        val result = GemmaResponseParser.parse("""[{"category":"spaceship","box_2d":[0,0,10,10]}]""")
        assertEquals(ProductCategory.OTHER, result.single().category)
    }

    @Test
    fun `invalid items are skipped and valid items kept`() {
        val text = """[
            {"category":"tea","box_2d":[500,500,100,100]},
            {"category":"tea","box_2d":[1,2,3]},
            {"category":"tea","box_2d":[0,"x",10,10]},
            {"box_2d":[0,0,10,10]},
            42,
            {"category":"tea","box_2d":[0,0,10,10]}
        ]"""
        assertEquals(1, GemmaResponseParser.parse(text).size)
    }

    @Test
    fun `out of range coordinates are clamped`() {
        val box = GemmaResponseParser.parse("""[{"category":"tea","box_2d":[-50,0,1200,500]}]""").single().box
        assertEquals(0f, box.top, 0f)
        assertEquals(1f, box.bottom, 0f)
    }

    @Test
    fun `broken or empty responses yield nothing`() {
        assertTrue(GemmaResponseParser.parse("").isEmpty())
        assertTrue(GemmaResponseParser.parse("no shelf here").isEmpty())
        assertTrue(GemmaResponseParser.parse("[{\"category\":").isEmpty())
        assertTrue(GemmaResponseParser.parse("[]").isEmpty())
    }
}
