package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.Listener
import jp.devfest.shelfsight.assist.LookedUp
import jp.devfest.shelfsight.assist.MapItem
import jp.devfest.shelfsight.assist.StoreFixture
import jp.devfest.shelfsight.assist.StoreGuide
import jp.devfest.shelfsight.assist.StoreView
import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class StoreGuideTest {
    /** x=-2 の面が +x を向く棚（カップ麺）と、x=2 の面が -x を向く棚（お茶）。あいだの通路に立つ。 */
    private fun fixture(c: Double, sign: String, name: String) =
        StoreFixture("wall", 0, c, -0.6, 0.6, 0.0, 1.8, 1, 1.2, mapOf(sign to listOf(listOf(name to 0.8))))

    private val view = StoreView(0.0, doubleArrayOf(-3.0, -3.0, 3.0, 3.0), listOf(fixture(-2.0, "+", "カップ麺"), fixture(2.0, "-", "お茶")))
    private val alongAisle = Listener(Vec3(0f, 1.3f, 0f), 0f, -1f)   // -z へ歩いている。右が +x

    private fun item(name: String, x: Float, y: Float, z: Float, seen: Int = 3) = MapItem(name, null, null, Vec3(x, y, z), 0.1f, 0.2f, seen, 0L)

    @Test
    fun `directions`() {
        assertEquals("前", StoreGuide.direction(10f))
        assertEquals("右前", StoreGuide.direction(45f))
        assertEquals("左", StoreGuide.direction(-90f))
        assertEquals("左後ろ", StoreGuide.direction(-135f))
        assertEquals("後ろ", StoreGuide.direction(170f))
    }

    @Test
    fun `walking down an aisle hears both sides and is not facing either`() {
        val shelves = StoreGuide.nearShelves(view, alongAisle)
        assertFalse(shelves.any { it.facing })
        val clauses = StoreGuide.clauses(StoreGuide.toSpeak(shelves))
        assertEquals(listOf("左にはカップ麺の棚、", "右にはお茶の棚があります。"), clauses)
    }

    @Test
    fun `turning to a shelf is facing it, and shelves behind their back face are skipped`() {
        val turned = Listener(Vec3(1f, 1.3f, 0f), 1f, 0f)   // +x の棚を向く
        val near = StoreGuide.nearShelves(view, turned)
        assertEquals("お茶", near.first { it.facing }.main)
        val behindShelf = Listener(Vec3(2.5f, 1.3f, 0f), -1f, 0f)
        assertTrue(StoreGuide.nearShelves(view, behindShelf).none { it.main == "お茶" })
    }

    @Test
    fun `facing sentence is split by height from the floor`() {
        val items = listOf(item("緑茶", 2f, 1.5f, 0f), item("麦茶", 2f, 1.4f, 0.2f), item("コーラ", 2f, 1.0f, 0f), item("水", 2f, 0.4f, 0f))
        assertEquals("お茶の棚で、上の方には緑茶や麦茶、真ん中にはコーラ、下の方には水があります。", StoreGuide.facingSentence("お茶", items, 0f))
        assertEquals("緑茶の棚で、上の方には緑茶があります。", StoreGuide.facingSentence(null, listOf(item("緑茶", 2f, 1.5f, 0f)), 0f))
        assertNull(StoreGuide.facingSentence(null, emptyList(), 0f))
    }

    @Test
    fun `AI context has the map, found products and looked-up products relative to the user`() {
        val text = StoreGuide.context(view, alongAisle, listOf(item("カップ焼きそば", -2f, 1f, 0f)), listOf(LookedUp("日清 カップヌードル", Vec3(0f, 1.3f, -1f), 0L)), 120_000L, "左には棚があります。")
        assertTrue(text, "左 2.0m: 壁ぎわの棚: カップ麺" in text)
        assertTrue(text, "カップ焼きそば: 左 2.0m" in text)
        assertTrue(text, "日清 カップヌードル: 前 1.0m、2 分前" in text)
        assertTrue(text, "最後に利用者へ話したこと" in text)
        assertTrue("位置はまだ" in StoreGuide.context(null, null, emptyList(), emptyList(), 0L, ""))
        val themed = StoreView(0.0, null, listOf(StoreFixture("island", 0, -2.0, -0.6, 0.6, 0.88, 1.8, 1, 1.2, mapOf("+" to listOf(listOf("カップ麺" to 1.0))), mapOf("+" to listOf("カップ麺・スープ・レトルト")))))
        assertTrue("テーマ: カップ麺・スープ・レトルト" in StoreGuide.context(themed, alongAisle, emptyList(), emptyList(), 0L, ""))
    }

    @Test
    fun `tap answers and AI context match python`() {
        val root = generateSequence(java.io.File("").absoluteFile) { it.parentFile }.first { java.io.File(it, "settings.gradle.kts").exists() }
        val groups = jp.devfest.shelfsight.assist.ShelfDescriber.Groups.parse(java.io.File(root, "app/src/main/assets/coarse/categories.json").readText())
        val model = jp.devfest.shelfsight.assist.StoreModel(jp.devfest.shelfsight.assist.StoreModel.parseThemes(java.io.File(root, "app/src/main/assets/store_themes.json").readText()))
        val items = ArrayList<MapItem>()
        val text = javaClass.classLoader!!.getResource("store_guide_trace.tsv")!!.readText(Charsets.UTF_8).replace("\r\n", "\n")
        val (input, expected) = text.split("\n===\n")
        for (line in input.lines()) {
            val p = line.split("\t")
            when (p[0]) {
                "P" -> model.addProduct(p[1].toDouble(), p[2].toDouble(), p[3].toDouble(), p[4].toDouble(), p[5], p[6])
                "T" -> model.addCamera(p[1].toDouble(), p[2].toDouble())
                "I" -> items += MapItem(p[1], p[2], p[3], Vec3(p[4].toFloat(), p[5].toFloat(), p[6].toFloat()), 0.1f, 0.2f, p[7].toInt(), 0L)
            }
        }
        val view = model.estimate()
        for (line in expected.trim().lines()) {
            val p = line.split("\t")
            val l = Listener(Vec3(p[1].toFloat(), p[2].toFloat(), p[3].toFloat()), p[4].toFloat(), p[5].toFloat())
            when (val a = StoreGuide.tapAnswer(view, items, l, groups, 0f)) {
                is jp.devfest.shelfsight.assist.TapAnswer.Sentence -> { assertEquals("sentence", p[6]); assertEquals(p[7], a.text) }
                is jp.devfest.shelfsight.assist.TapAnswer.Around -> { assertEquals("around", p[6]); assertEquals(p[7], a.clauses.joinToString("")) }
                jp.devfest.shelfsight.assist.TapAnswer.Unknown -> assertEquals("unknown", p[6])
            }
            assertEquals(p[8], StoreGuide.context(view, l, items, emptyList(), 0L, "").replace("\n", "|"))
        }
    }
}
