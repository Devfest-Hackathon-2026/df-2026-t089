package jp.devfest.shelfsight.assist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class PrecomputedRecognitionTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun product(left: Float, name: String) = RecognizedProduct(left, 0f, left + 0.2f, 0.2f, name, null, null, 0.9f)

    @Test fun `takeDue gives the latest due entry once and rewinds`() {
        val a = PrecomputedRecognition.Entry(10, emptyList())
        val b = PrecomputedRecognition.Entry(20, emptyList())
        val p = PrecomputedRecognition(listOf(a, b))
        assertNull(p.takeDue(5))
        assertEquals(b, p.takeDue(25))
        assertNull(p.takeDue(30))
        p.rewind()
        assertEquals(a, p.takeDue(10))
    }

    @Test fun `match keeps the live box and drops boxes with no overlap`() {
        val e = PrecomputedRecognition.Entry(0, listOf(product(0f, "tea"), product(0.5f, "bread")))
        val live = floatArrayOf(0.51f, 0f, 0.71f, 0.2f)
        val out = e.match(listOf(live, floatArrayOf(0.8f, 0.8f, 1f, 1f)))
        assertEquals(1, out.size)
        assertEquals("bread", out[0].name)
        assertEquals(0.51f, out[0].left)
    }

    private fun recording(): File = tmp.newFolder().also {
        File(it, "keyframes.jsonl").writeText(
            listOf(
                """{"t":0,"image":"a.jpg","state":"TRACKING"}""",
                """{"t":500000000,"image":"b.jpg","state":"TRACKING"}""",
                """{"t":1000000000,"image":"c.jpg","state":"PAUSED"}""",
                """{"t":1100000000,"image":"d.jpg","state":"TRACKING"}""",
            ).joinToString("\n") + "\n",
        )
    }

    @Test fun `load needs every tracked one-second keyframe and the same version`() {
        val dir = recording()
        val results = File(dir, "recognition.jsonl")
        val t0 = """{"t":0,"p":[[0,0,0.2,0.2,"tea",null,"green",0.9]]}"""
        val t1 = """{"t":1100000000,"p":[]}"""
        results.writeText("v1\n$t0\n")
        assertNull(PrecomputedRecognition.load(dir, "v1"))
        results.writeText("v1\n$t0\n$t1\n")
        assertNull(PrecomputedRecognition.load(dir, "v2"))
        val loaded = PrecomputedRecognition.load(dir, "v1")
        assertNotNull(loaded)
        val first = loaded!!.takeDue(0)!!
        assertEquals("tea", first.products.single().name)
        assertEquals("green", first.products.single().fine)
        assertNull(first.products.single().genre)
    }
}
