package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.Binaural
import jp.devfest.shelfsight.assist.Command
import jp.devfest.shelfsight.assist.Commands
import jp.devfest.shelfsight.assist.ContextAdjust
import jp.devfest.shelfsight.assist.ContextAdjust.Candidate
import jp.devfest.shelfsight.assist.Focus
import jp.devfest.shelfsight.assist.Listener
import jp.devfest.shelfsight.assist.Narrator
import jp.devfest.shelfsight.assist.PersonFilter
import jp.devfest.shelfsight.assist.Reading
import jp.devfest.shelfsight.assist.RecognizedProduct
import jp.devfest.shelfsight.assist.Scene
import jp.devfest.shelfsight.assist.SceneEvidence
import jp.devfest.shelfsight.assist.SceneJudge
import jp.devfest.shelfsight.assist.ShelfMap
import jp.devfest.shelfsight.assist.ShelfPrior
import jp.devfest.shelfsight.assist.ShelfDescriber
import jp.devfest.shelfsight.assist.VisionProductParser
import jp.devfest.shelfsight.assist.Wav
import jp.devfest.shelfsight.model.NormalizedBox
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

    @Test
    fun `drinking yogurt among gummies is corrected to gummy by the category itself`() {
        val cands = List(4) { c("グミ" to 0.9f) } + listOf(c("飲むヨーグルト" to 0.80f, "グミ" to 0.77f), c("飲むヨーグルト" to 0.9f))
        assertEquals("飲むヨーグルト", ContextAdjust.adjust(centers, cands, emptyList())[4].key)
        assertEquals("グミ", ContextAdjust.adjust(centers, cands, listOf(ContextAdjust.Level({ it }, 0.1)))[4].key)
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

    private val root = listOf(File("."), File("..")).first { File(it, "tools/coarse_categories.json").exists() }
    private val groups = ShelfDescriber.Groups.parse(File(root, "tools/coarse_categories.json").readText())

    @Test
    fun `the app's copy of the 7 groups is the same as the training one`() {
        assertEquals(File(root, "tools/coarse_categories.json").readText(), File(root, "app/src/main/assets/coarse/categories.json").readText())
    }

    @Test
    fun `odd products on the bread shelf are removed even when several were misrecognized`() {
        val bread = (0 until 12).map { i -> p(0.05f + 0.08f * i, if (i < 6) 0.2f else 0.6f, if (i % 2 == 0) "菓子パン" else "食パン・調理パン", "パン") }
        // 間違った商品が 3 つずつ（前は「2 個以下」しか外さなかったので残っていた）
        val odd = (0 until 3).map { p(0.5f, 0.4f, "チョコレート", "チョコ・ビスケット") } + (0 until 3).map { p(0.6f, 0.4f, "緑茶", "ソフトドリンク") }
        val split = ShelfDescriber.split(bread + odd, groups)
        assertEquals("弁当・パン", split.mainGroup)
        assertEquals(12, split.kept.size)
        assertEquals(6, split.removed.size)
        val text = ShelfDescriber.describe(split.kept)!!
        assertFalse(text, "チョコ" in text || "緑茶" in text)
    }

    @Test
    fun `different snack genres on a snack shelf all stay`() {
        val snacks = listOf("スナック菓子" to "ポテトチップス・スナック", "チョコ・ビスケット" to "チョコレート", "グミ・キャンディ" to "グミ")
            .flatMap { (g, f) -> (0 until 4).map { p(0.1f + 0.2f * it, 0.3f, f, g) } }
        val split = ShelfDescriber.split(snacks + p(0.5f, 0.8f, "カップ麺", "カップ麺"), groups)
        assertEquals("お菓子・おつまみ", split.mainGroup)
        assertEquals(12, split.kept.size)
        assertEquals(listOf("カップ麺"), split.removed.map { it.genre })
    }

    @Test
    fun `gummies on the potato chip shelf are removed although both are snacks`() {
        val chips = (0 until 9).map { p(0.05f + 0.1f * it, if (it < 5) 0.2f else 0.6f, "ポテトチップス・スナック", "スナック菓子") }
        val gummies = (0 until 2).map { p(0.4f + 0.1f * it, 0.6f, "グミ", "グミ・キャンディ") }
        val split = ShelfDescriber.split(chips + gummies, groups)
        assertEquals(listOf("ポテトチップス・スナック"), split.mainKinds)
        assertEquals(9, split.kept.size)
        assertEquals(listOf("グミ", "グミ"), split.removed.map { it.fine })
    }

    @Test
    fun `the shelf type from the walking classifier decides the main group`() {
        // 飲み物の方が多く認識されても、目の前の棚がお菓子の棚と分かっていればお菓子を残す
        val drinks = (0 until 6).map { p(0.1f * it, 0.2f, "緑茶") }
        val snacks = (0 until 4).map { p(0.1f * it, 0.6f, "チョコレート", "チョコ・ビスケット") }
        assertEquals("お菓子・おつまみ", ShelfDescriber.split(drinks + snacks, groups, "お菓子・おつまみ").mainGroup)
        assertEquals("飲み物・お酒", ShelfDescriber.split(drinks + snacks, groups).mainGroup)
    }

    @Test
    fun `a genuinely mixed shelf keeps both genres`() {
        val mixed = (0 until 4).map { p(0.1f + 0.2f * it, 0.2f, "菓子パン", "パン") } +
            (0 until 4).map { p(0.1f + 0.2f * it, 0.6f, "チョコレート", "チョコ・ビスケット") }
        val text = ShelfDescriber.describe(ShelfDescriber.split(mixed, groups).kept)!!
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
        // 大きな枠があっても、商品 1 つにはしない
        assertEquals(Scene.ONE_SHELF, SceneJudge.classify(SceneEvidence(1, listOf(big, small(0f)))))
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

    // ---------------------------------------------------------------- 長押しで調べるもの

    @Test
    fun `long press takes the product at the center of the screen`() {
        val center = floatArrayOf(0.4f, 0.4f, 0.6f, 0.65f)
        val left = floatArrayOf(0.05f, 0.4f, 0.25f, 0.6f)
        assertTrue(SceneJudge.centerProduct(listOf(left, center))!!.contentEquals(center))
        // 中央を含む枠がなければ、中心が近い枠
        val near = floatArrayOf(0.52f, 0.52f, 0.66f, 0.66f)
        assertTrue(SceneJudge.centerProduct(listOf(left, near))!!.contentEquals(near))
        // 中央の近くに商品がない
        assertNull(SceneJudge.centerProduct(listOf(left)))
        assertNull(SceneJudge.centerProduct(emptyList()))
    }

    @Test
    fun `long press picks the product, the shelf or asks to step back`() {
        val small = { x: Float, y: Float -> floatArrayOf(x, y, x + 0.08f, y + 0.12f) }
        val shelf = (0 until 4).flatMap { i -> listOf(small(0.05f + 0.22f * i, 0.1f), small(0.05f + 0.22f * i, 0.6f)) }
        assertEquals(Focus.SHELF, SceneJudge.focus(shelf).first)
        // 大きな枠があっても、商品 1 つにはしない
        assertEquals(Focus.STEP_BACK, SceneJudge.focus(listOf(floatArrayOf(0.2f, 0.15f, 0.8f, 0.85f), small(0f, 0f))).first)
        // 少し大きめの商品が 2 つ並んでいる（どちらとも決まらない）
        assertEquals(Focus.STEP_BACK, SceneJudge.focus(listOf(floatArrayOf(0.1f, 0.2f, 0.45f, 0.7f), floatArrayOf(0.5f, 0.2f, 0.85f, 0.7f))).first)
        // 商品が 3 つだけ
        assertEquals(Focus.STEP_BACK, SceneJudge.focus(listOf(small(0.1f, 0.4f), small(0.4f, 0.4f), small(0.7f, 0.4f))).first)
        assertEquals(Focus.NOTHING, SceneJudge.focus(emptyList()).first)
    }

    // ---------------------------------------------------------------- 自分の手を人と間違えない

    @Test
    fun `own hands are not people`() {
        // 画面の下から入ってきた腕
        assertTrue(PersonFilter.isOwnHand(NormalizedBox(0.3f, 0.6f, 0.8f, 1f), 1.0f, 0.4f))
        // 顔の前にかざした手（枠は画面の中、横に広い、深度は近い）
        assertTrue(PersonFilter.isOwnHand(NormalizedBox(0.3f, 0.3f, 0.7f, 0.6f), 1.0f, 0.35f))
        // 縦に長くても、深度が近く、画面の上まで届いていない（商品を持った手）
        assertTrue(PersonFilter.isOwnHand(NormalizedBox(0.4f, 0.35f, 0.6f, 0.9f), 2.0f, 0.5f))
        // 深度が取れなくても、全身が写ったように見えて横に広いものは人ではない
        assertTrue(PersonFilter.isOwnHand(NormalizedBox(0.2f, 0.3f, 0.7f, 0.7f), 1.0f, null))
        // 通路の向こうに立っている人
        assertFalse(PersonFilter.isOwnHand(NormalizedBox(0.45f, 0.3f, 0.55f, 0.7f), 3.0f, 3.5f))
        // 深度が取れない遠くの人
        assertFalse(PersonFilter.isOwnHand(NormalizedBox(0.45f, 0.35f, 0.52f, 0.6f), 2.8f, null))
        // すぐ前に立っている人（画面の上から下まで写る）
        assertFalse(PersonFilter.isOwnHand(NormalizedBox(0.2f, 0f, 0.8f, 1f), 1.3f, 0.9f))
    }

    // ---------------------------------------------------------------- 読み方

    @Test
    fun `position words are written so the speech engine reads them naturally`() {
        assertEquals("うえのだんには緑茶、したのだんにはコーラがあります。", Reading.forSpeech("上段には緑茶、下段にはコーラがあります。"))
        assertEquals("うえのほうにはお茶、まんなかあたりにはコーヒー。", Reading.forSpeech("上の方にはお茶、真ん中あたりにはコーヒー。"))
        assertEquals("うえから2だんめと、いちばんしたのだん", Reading.forSpeech("上から2段目と、最下段"))
        assertEquals("棚のみぎがわ、ひだりには", Reading.forSpeech("棚の右側、左には"))
        // 漢字の言葉の一部の「上」「下」は変えない
        assertEquals("以上のことから、地下に", Reading.forSpeech("以上のことから、地下に"))
        assertEquals("ユニ・チャーム", Reading.forSpeech("ユニ・チャーム"))
    }

    // ---------------------------------------------------------------- LLM への依頼

    @Test
    fun `shelf prompt lists rows top to bottom with kinds and counts, and replies are parsed`() {
        val rows = listOf(
            listOf(p(0.1f, 0.15f, "緑茶").copy(name = "伊藤園 お〜いお茶"), p(0.3f, 0.15f, "緑茶").copy(name = "伊藤園 お〜いお茶")),
            listOf(p(0.1f, 0.8f, "炭酸飲料").copy(name = "コカ・コーラ コカ・コーラ 500ml")),
        )
        val prompt = Narrator.shelfPrompt(rows)
        assertTrue(prompt, "上の段: 伊藤園 お〜いお茶（緑茶）×2" in prompt)
        assertTrue(prompt, "下の段: コカ・コーラ コカ・コーラ 500ml（炭酸飲料）" in prompt)
        assertTrue(prompt.indexOf("上の段:") < prompt.indexOf("下の段:"))
        assertEquals("上のほうにお茶があります。", Narrator.parse("""{"text":" 上のほうにお茶があります。 "}"""))
        assertNull(Narrator.parse("""{"text":""}"""))
        assertTrue(Narrator.questionPrompt("これはいくら？", "- 棚: 飲み物").contains("これはいくら？"))
    }

    // ---------------------------------------------------------------- 地図（壁の上に貯めた商品）

    /** 原点で -z を向いたカメラ（センサー 640x480、正立させるには 90° 回す）。 */
    private val geom = jp.devfest.shelfsight.ar.CaptureGeometry(
        imageWidth = 640, imageHeight = 480, focalX = 500f, focalY = 500f, principalX = 320f, principalY = 240f,
        cameraToWorld = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f),
        uprightRotationDegrees = 90, depth = null, depthU0 = 0f, depthUPerPixel = 0f, depthV0 = 0f, depthVPerPixel = 0f,
    )

    @Test
    fun `a point unprojected from the image projects back to the same place`() {
        for ((u, v) in listOf(0.5f to 0.5f, 0.2f to 0.7f, 0.9f to 0.1f)) {
            val world = jp.devfest.shelfsight.ar.Unprojector.worldAt(u, v, 2f, geom)
            val back = ShelfMap.project(world, geom)!!
            assertEquals(u, back[0], 1e-4f)
            assertEquals(v, back[1], 1e-4f)
            assertEquals(2f, back[2], 1e-4f)
        }
        assertNull(ShelfMap.project(Vec3(0f, 0f, 2f), geom))   // カメラの後ろ
        val (w, h) = ShelfMap.sizeInMeters(0.1f, 0.2f, 2f, geom)
        val (wi, hi) = ShelfMap.sizeInImage(w, h, 2f, geom)
        assertEquals(0.1f, wi, 1e-5f)
        assertEquals(0.2f, hi, 1e-5f)
    }

    @Test
    fun `repeated sightings merge and the most seen name wins at one spot`() {
        val m = ShelfMap()
        val at = Vec3(0f, 0f, -2f)
        repeat(3) { m.add("菓子パン A", "パン", "菓子パン", at, 0.1f, 0.1f, it.toLong()) }
        m.add("チョコ B", "チョコ・ビスケット", "チョコレート", Vec3(0.05f, 0f, -2f), 0.1f, 0.1f, 5)   // 同じ場所の読み違い
        m.add("菓子パン A", "パン", "菓子パン", Vec3(0.1f, 0f, -2f), 0.1f, 0.1f, 6)
        assertEquals(2, m.size)
        val seen = m.visible(geom)
        assertEquals(listOf("菓子パン A"), seen.map { it.name })
        assertEquals(4f, seen[0].sim)
        assertEquals(0.5f, seen[0].cx, 0.02f)
    }

    @Test
    fun `shelf labels gather votes at a place and are read back only from the same side`() {
        val m = jp.devfest.shelfsight.assist.ShelfLabelMap()
        val front = Vec3(0f, 0f, 0f)       // 棚の前（+z 側）から見る
        repeat(4) { m.add("飲み物・お酒", Vec3(0f, 0f, -2f), front, it.toLong()) }
        m.add("お菓子・おつまみ", Vec3(0.1f, 0f, -2f), front, 5)   // 1 枚だけの読み違い
        // 背中合わせの棚の裏側（-z 側から見た）
        repeat(5) { m.add("日用品・雑貨", Vec3(0f, 0f, -2.3f), Vec3(0f, 0f, -4f), it.toLong()) }
        assertEquals(2, m.size)

        val p = m.at(Vec3(0.05f, 0f, -2f), front)!!
        assertEquals("飲み物・お酒", p.label)
        assertEquals(5, p.votes)
        assertEquals(0.8f, p.share, 1e-6f)
        assertEquals("日用品・雑貨", m.at(Vec3(0f, 0f, -2.3f), Vec3(0f, 0f, -4f))!!.label)
        assertNull(m.at(Vec3(5f, 0f, -2f), front))   // まだ見ていない場所
    }

    @Test
    fun `products are grouped under the shelf column they stand in, most seen first`() {
        fun at(x: Float, name: String) = RecognizedProduct(x - 0.02f, 0.4f, x + 0.02f, 0.5f, name, null, null, 0.6f)
        val cols = jp.devfest.shelfsight.assist.ShelfColumns.productsByColumn(
            listOf(at(0.05f, "お茶 A"), at(0.1f, "水 B"), at(0.08f, "水 B"), at(0.99f, "コーラ C")), columns = 8,
        )
        assertEquals(8, cols.size)
        assertEquals(listOf("水 B", "お茶 A"), cols[0])
        assertEquals(listOf("コーラ C"), cols[7])
        assertTrue(cols[3].isEmpty())
    }

    @Test
    fun `a column is corrected by the map only with enough agreeing votes`() {
        val drinks = ShelfPrior("飲み物・お酒", 0.8f, 5, mapOf("飲み物・お酒" to 0.8f, "お菓子・おつまみ" to 0.2f))
        val (cols, fixed) = jp.devfest.shelfsight.assist.ShelfLabelMap.correct(
            listOf("お菓子・おつまみ", "お菓子・おつまみ", "食品", "食品"),
            listOf(drinks, ShelfPrior("飲み物・お酒", 1f, 2, mapOf("飲み物・お酒" to 1f)), ShelfPrior("食品", 1f, 9, mapOf("食品" to 1f)), null),
        )
        assertEquals(listOf("飲み物・お酒", "お菓子・おつまみ", "食品", "食品"), cols)   // 2 票ではまだ直さない
        assertEquals(listOf(true, false, false, false), fixed.toList())
        val segs = jp.devfest.shelfsight.assist.ShelfSegment.from(cols, floatArrayOf(0.9f, 0.7f, 0.8f, 0.6f), fixed)
        assertEquals(listOf("飲み物・お酒", "お菓子・おつまみ", "食品"), segs.map { it.category })
        assertEquals(listOf(true, false, false), segs.map { it.fromMap })
        assertEquals(0.5f, segs[2].left)
        assertEquals(0.7f, segs[2].confidence, 1e-6f)
        // 「商品なし」は棚にしない
        assertTrue(jp.devfest.shelfsight.assist.ShelfSegment.from(listOf("商品なし"), floatArrayOf(0.9f)).isEmpty())
    }

    @Test
    fun `the shelf type lowers names from other groups but does not overturn a clear match`() {
        val shelf = jp.devfest.shelfsight.assist.ShelfPrior("弁当・パン", 1f, 10, mapOf("弁当・パン" to 1f))
        val group = mapOf("choco" to "お菓子・おつまみ", "bread" to "弁当・パン")
        val out = jp.devfest.shelfsight.assist.MapBias.adjust(c("choco" to 0.70f, "bread" to 0.66f), shelf) { group[it] }
        assertEquals("bread", out.first().key)
        val far = jp.devfest.shelfsight.assist.MapBias.adjust(c("choco" to 0.85f, "bread" to 0.66f), shelf) { group[it] }
        assertEquals("choco", far.first().key)
        // 1 枚の判別だけ（1 票）なら弱いので、差が小さくても覆さないことがある
        val weak = shelf.copy(votes = 1)
        assertEquals("choco", jp.devfest.shelfsight.assist.MapBias.adjust(c("choco" to 0.70f, "bread" to 0.66f), weak) { group[it] }.first().key)
    }

    @Test
    fun `products are placed on the wall`() {
        val wall = jp.devfest.shelfsight.assist.Wall(Vec3(0f, 0f, -2f), Vec3(0f, 0f, 1f), 1.5f)
        // 深度の点が壁の少し手前 → 壁の上に寄せる
        val snapped = ShelfMap.place(Vec3(0f, 0f, 0f), Vec3(0f, 0f, -1f), Vec3(0.3f, 0.1f, -1.85f), listOf(wall))!!
        assertEquals(-2f, snapped.z, 1e-5f)
        assertEquals(0.3f, snapped.x, 1e-5f)
        // 深度がない → カメラからの向きと壁が交わる所
        val hit = ShelfMap.place(Vec3(0f, 0f, 0f), Vec3(0.25f, 0f, -1f), null, listOf(wall))!!
        assertEquals(-2f, hit.z, 1e-5f)
        assertEquals(0.5f, hit.x, 1e-5f)
        // 深度も壁もない → 置けない
        assertNull(ShelfMap.place(Vec3(0f, 0f, 0f), Vec3(0f, 0f, -1f), null, emptyList()))
    }
}
