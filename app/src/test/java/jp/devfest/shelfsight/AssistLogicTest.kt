package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.Binaural
import jp.devfest.shelfsight.assist.Command
import jp.devfest.shelfsight.assist.Commands
import jp.devfest.shelfsight.assist.ContextAdjust
import jp.devfest.shelfsight.assist.ContextAdjust.Candidate
import jp.devfest.shelfsight.assist.Listener
import jp.devfest.shelfsight.assist.PlacedShelf
import jp.devfest.shelfsight.assist.RecognizedProduct
import jp.devfest.shelfsight.assist.Scene
import jp.devfest.shelfsight.assist.SceneEvidence
import jp.devfest.shelfsight.assist.SceneJudge
import jp.devfest.shelfsight.assist.ShelfDescriber
import jp.devfest.shelfsight.assist.ShelfMemory
import jp.devfest.shelfsight.assist.VisionProductParser
import jp.devfest.shelfsight.assist.Wav
import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI

class AssistLogicTest {
    // ---------------------------------------------------------------- ① 列ごとのカテゴリ → 棚

    @Test
    fun `runs of the same column category become shelves and no-product columns are dropped`() {
        val cols = listOf("飲み物・お酒", "飲み物・お酒", "商品なし", "日用品・雑貨", "日用品・雑貨", "日用品・雑貨", "お菓子・おつまみ", "お菓子・おつまみ")
        val conf = floatArrayOf(0.9f, 0.7f, 0.8f, 0.6f, 0.6f, 0.9f, 0.5f, 0.7f)
        val shelves = ShelfMemory.shelves(cols, conf)
        assertEquals(listOf("飲み物・お酒", "日用品・雑貨", "お菓子・おつまみ"), shelves.map { it.category })
        assertEquals(0f, shelves[0].left, 1e-6f)
        assertEquals(0.25f, shelves[0].right, 1e-6f)
        assertEquals(0.375f, shelves[1].left, 1e-6f)
        assertEquals(0.75f, shelves[1].right, 1e-6f)
        assertEquals(0.8f, shelves[0].confidence, 1e-6f)
        assertEquals("飲み物とお酒", shelves[0].spoken)
        assertTrue(ShelfMemory.shelves(List(8) { ShelfMemory.NO_PRODUCT }, FloatArray(8) { 1f }).isEmpty())
    }

    @Test
    fun `the walking classifier head has the 7 categories over 8 columns`() {
        val dir = listOf("src/main/assets/coarse", "app/src/main/assets/coarse").map(::File).first { it.exists() }
        for (f in dir.listFiles { x -> x.name.endsWith(".head.json") }!!) {
            val head = jp.devfest.shelfsight.walking.CoarseHead.parse(f.readText())
            assertEquals(f.name, 7, head.classes.size)
            assertEquals(f.name, 8, head.columns)
            assertTrue(ShelfMemory.NO_PRODUCT in head.classes)
        }
    }

    @Test
    fun `the same shelf at the same place is announced once`() {
        val m = ShelfMemory()
        val s = PlacedShelf("冷凍・アイス", 0.6f, 0f, 1f, Vec3(0f, 0f, -2f))
        assertTrue(m.isNew(s, 0))
        m.remember(s, 0)
        assertFalse(m.isNew(s.copy(position = Vec3(0.5f, 0f, -2f)), 1000))
        assertTrue(m.isNew(s.copy(position = Vec3(5f, 0f, -2f)), 1000))       // 別の場所の同じ種類
        assertTrue(m.isNew(s, ShelfMemory.REMEMBER_MS + 1))                    // しばらく経った
    }

    // ---------------------------------------------------------------- ② 周りの商品による補正（products/context.py の自己チェックと同じ）

    private val centers = listOf(0.1f to 0.1f, 0.2f to 0.1f, 0.3f to 0.1f, 0.4f to 0.1f, 0.25f to 0.11f, 0.25f to 0.6f)
    private val genre = mapOf("bread_a" to "パン", "bread_b" to "パン", "wipe" to "日用品・衛生用品", "tea" to "ソフトドリンク", "coffee" to "ソフトドリンク")
    private val fine = mapOf("tea" to "緑茶", "tea2" to "緑茶", "coffee" to "コーヒー")
    private fun c(vararg xs: Pair<String, Float>) = xs.map { Candidate(it.first, it.second) }

    @Test
    fun `a slightly similar odd product among breads is corrected to bread`() {
        val cands = listOf(c("bread_a" to 0.9f), c("bread_b" to 0.9f), c("bread_a" to 0.88f), c("bread_b" to 0.9f),
            c("wipe" to 0.80f, "bread_a" to 0.78f), c("tea" to 0.9f))
        assertEquals("wipe", ContextAdjust.adjust(centers, cands, listOf(ContextAdjust.Level({ genre[it] }, 0.0)))[4].key)
        assertEquals("bread_a", ContextAdjust.adjust(centers, cands, listOf(ContextAdjust.Level({ genre[it] }, 0.05)))[4].key)
    }

    @Test
    fun `fine categories correct within the same genre and other rows matter little`() {
        val cands = listOf(c("tea" to 0.9f), c("tea2" to 0.9f), c("tea" to 0.9f), c("tea2" to 0.9f),
            c("coffee" to 0.80f, "tea" to 0.78f), c("coffee" to 0.9f))
        val g = ContextAdjust.Level({ genre[it] }, 0.05)
        val f = ContextAdjust.Level({ fine[it] }, 0.05)
        assertEquals("coffee", ContextAdjust.adjust(centers, cands, listOf(g))[4].key)
        assertEquals("tea", ContextAdjust.adjust(centers, cands, listOf(g, f))[4].key)
        assertEquals("coffee", ContextAdjust.adjust(centers, cands, listOf(f))[5].key)
    }

    // ---------------------------------------------------------------- ② 段ごとの説明

    private fun p(x: Float, y: Float, fine: String, genre: String = "ソフトドリンク") =
        RecognizedProduct(x - 0.04f, y - 0.06f, x + 0.04f, y + 0.06f, "$fine の商品", genre, fine, 0.8f)

    @Test
    fun `a drink shelf is described row by row from the top`() {
        val shelf = listOf(
            p(0.1f, 0.15f, "緑茶"), p(0.3f, 0.16f, "緑茶"), p(0.5f, 0.15f, "麦茶・ほうじ茶・その他のお茶"),
            p(0.1f, 0.5f, "紅茶"), p(0.3f, 0.5f, "紅茶"), p(0.5f, 0.52f, "コーヒー"),
            p(0.1f, 0.85f, "果汁・野菜ジュース"), p(0.3f, 0.86f, "炭酸飲料"),
        )
        assertEquals(3, ShelfDescriber.rows(shelf).size)
        assertEquals(
            "上の段には緑茶や麦茶などのお茶、真ん中の段には紅茶やコーヒー、下の段にはジュースや炭酸飲料があります。",
            ShelfDescriber.describe(shelf),
        )
    }

    @Test
    fun `a chocolate bar on the bread shelf is not mentioned`() {
        val bread = (0 until 8).map { i -> p(0.1f + 0.1f * i, if (i < 4) 0.2f else 0.6f, if (i % 2 == 0) "菓子パン" else "食パン・調理パン", "パン") }
        val odd = p(0.45f, 0.6f, "チョコレート", "チョコ・ビスケット")
        val text = ShelfDescriber.describe(bread + odd)!!
        assertFalse(text, "チョコ" in text)
        assertTrue(text, "菓子パン" in text)
    }

    @Test
    fun `a genuinely mixed shelf keeps both genres`() {
        val mixed = (0 until 4).map { p(0.1f + 0.2f * it, 0.2f, "菓子パン", "パン") } +
            (0 until 4).map { p(0.1f + 0.2f * it, 0.6f, "チョコレート", "チョコ・ビスケット") }
        val text = ShelfDescriber.describe(mixed)!!
        assertTrue(text, "菓子パン" in text && "チョコレート" in text)
    }

    @Test
    fun `too few products gives no description`() {
        assertNull(ShelfDescriber.describe(listOf(p(0.5f, 0.5f, "緑茶"))))
    }

    @Test
    fun `row names`() {
        assertEquals(listOf("上の段", "下の段"), ShelfDescriber.rowNames(2))
        assertEquals(listOf("いちばん上の段", "上から2段目", "上から3段目", "いちばん下の段"), ShelfDescriber.rowNames(4))
    }

    // ---------------------------------------------------------------- ③ Cloud Vision の応答

    @Test
    fun `maker and product come from web detection`() {
        val json = """{"responses":[{"webDetection":{
            "bestGuessLabels":[{"label":"サントリー 伊右衛門"}],
            "webEntities":[{"description":"Iyemon","score":0.8},{"description":"Green tea","score":0.6}],
            "pagesWithMatchingImages":[{"pageTitle":"伊右衛門 | サントリー"}],
            "fullMatchingImages":[{"url":"x"}]}}]}"""
        val a = VisionProductParser.parse(json, 1000)!!
        assertEquals("サントリー", a.maker)
        assertEquals("伊右衛門", a.product)
        assertEquals("サントリーの、伊右衛門です。", a.spoken())
    }

    @Test
    fun `generic web guesses fall back to the logo and the largest printed text`() {
        val json = """{"responses":[{
            "webDetection":{"bestGuessLabels":[{"label":"snack"}],"webEntities":[{"description":"Food","score":0.9}]},
            "logoAnnotations":[{"description":"Calbee"}],
            "fullTextAnnotation":{"pages":[{"blocks":[{"paragraphs":[{"words":[
              {"symbols":[{"text":"ポ","boundingBox":{"vertices":[{"x":0,"y":100},{"x":50,"y":100},{"x":50,"y":200},{"x":0,"y":200}]}},
                          {"text":"テ","boundingBox":{"vertices":[{"x":50,"y":100},{"x":100,"y":100},{"x":100,"y":200},{"x":50,"y":200}]}},
                          {"text":"ト","boundingBox":{"vertices":[{"x":100,"y":100},{"x":150,"y":100},{"x":150,"y":200},{"x":100,"y":200}]},
                           "property":{"detectedBreak":{"type":"LINE_BREAK"}}}]}]}]}]}]}}]}"""
        val a = VisionProductParser.parse(json, 1000)!!
        assertEquals("カルビー", a.maker)
        assertEquals("ポテト", a.product)
    }

    @Test
    fun `nothing useful gives null`() {
        assertNull(VisionProductParser.parse("""{"responses":[{"webDetection":{"bestGuessLabels":[{"label":"bottle"}]}}]}""", 1000))
    }

    @Test
    fun `the request asks for web, logo and text in one call`() {
        val r = VisionProductParser.request("AAAA")
        assertTrue("WEB_DETECTION" in r && "LOGO_DETECTION" in r && "DOCUMENT_TEXT_DETECTION" in r)
    }

    // ---------------------------------------------------------------- 写っているものの判定

    @Test
    fun `scene is decided from shelves and product boxes and needs two agreeing frames`() {
        val big = floatArrayOf(0.2f, 0.2f, 0.8f, 0.8f)
        val small = { x: Float -> floatArrayOf(x, 0.1f, x + 0.08f, 0.2f) }
        assertEquals(Scene.ONE_PRODUCT, SceneJudge.classify(SceneEvidence(1, listOf(big, small(0f)))))
        assertEquals(Scene.SHELVES, SceneJudge.classify(SceneEvidence(3, (0 until 10).map { small(it * 0.09f) })))
        assertEquals(Scene.ONE_SHELF, SceneJudge.classify(SceneEvidence(1, (0 until 10).map { small(it * 0.09f) })))
        assertEquals(Scene.NONE, SceneJudge.classify(SceneEvidence(0, emptyList())))

        val j = SceneJudge()
        assertFalse(j.update(SceneEvidence(3, emptyList())))
        assertEquals(Scene.NONE, j.scene)
        assertTrue(j.update(SceneEvidence(3, emptyList())))
        assertEquals(Scene.SHELVES, j.scene)
    }

    // ---------------------------------------------------------------- 声の操作

    @Test
    fun `spoken commands`() {
        assertEquals(Command.LOOKUP, Commands.parse("これは何？"))
        assertEquals(Command.DESCRIBE, Commands.parse("何がある？"))
        assertEquals(Command.SHELVES, Commands.parse("ここは何の棚"))
        assertEquals(Command.PAUSE, Commands.parse("ちょっと止めて"))
        assertEquals(Command.REPEAT, Commands.parse("もう一度"))
        assertNull(Commands.parse("こんにちは"))
    }

    // ---------------------------------------------------------------- 立体音響

    @Test
    fun `azimuth is positive to the right and zero ahead`() {
        val l = Listener(Vec3(0f, 0f, 0f), 0f, -1f)  // -z を向いている
        assertEquals(0f, Binaural.azimuth(l, Vec3(0f, 0f, -2f)), 1e-5f)
        assertEquals((PI / 2).toFloat(), Binaural.azimuth(l, Vec3(2f, 0f, 0f)), 1e-5f)
        assertEquals((-PI / 2).toFloat(), Binaural.azimuth(l, Vec3(-2f, 0f, 0f)), 1e-5f)
        assertEquals(3, Binaural.clock(Binaural.azimuth(l, Vec3(2f, 0f, 0f))))
        assertEquals(12, Binaural.clock(0f))
        assertEquals(9, Binaural.clock((-PI / 2).toFloat()))
    }

    @Test
    fun `a sound on the right is louder and earlier in the right ear`() {
        val e = Binaural.params((PI / 2).toFloat(), 1f, 44100)
        assertTrue(e.gainR > e.gainL * 3)
        assertTrue(e.delayL > 0.0005f && e.delayR == 0f)
        assertTrue(e.lowpassL < e.lowpassR)
        val front = Binaural.params(0f, 1f, 44100)
        assertEquals(front.gainL, front.gainR, 1e-6f)
        assertTrue(Binaural.params(PI.toFloat(), 1f, 44100).lowpassL < 1f)  // 後ろはこもらせる
        assertTrue(Binaural.distanceGain(4f) < Binaural.distanceGain(1f))
    }

    @Test
    fun `wav decoding`() {
        val samples = shortArrayOf(0, 16384, -16384, 32767)
        val b = ByteBuffer.allocate(44 + samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + samples.size * 2).put("WAVE".toByteArray())
        b.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1).putInt(22050).putInt(44100).putShort(2).putShort(16)
        b.put("data".toByteArray()).putInt(samples.size * 2)
        samples.forEach { b.putShort(it) }
        val (pcm, rate) = Wav.decode(b.array())
        assertEquals(22050, rate)
        assertEquals(4, pcm.size)
        assertEquals(0.5f, pcm[1], 1e-4f)
        assertNotNull(pcm)
    }
}
