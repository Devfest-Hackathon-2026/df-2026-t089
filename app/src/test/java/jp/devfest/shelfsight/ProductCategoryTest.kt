package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.CategoryJudge
import jp.devfest.shelfsight.assist.CategoryRules
import jp.devfest.shelfsight.assist.Focus
import jp.devfest.shelfsight.assist.ProductAnswer
import jp.devfest.shelfsight.assist.Scene
import jp.devfest.shelfsight.assist.SceneEvidence
import jp.devfest.shelfsight.assist.SceneJudge
import jp.devfest.shelfsight.assist.VisionProductParser
import jp.devfest.shelfsight.cloud.CloudException
import jp.devfest.shelfsight.cloud.CloudFailure
import jp.devfest.shelfsight.cloud.CloudVisionClient
import jp.devfest.shelfsight.ocr.CloudVisionParser
import jp.devfest.shelfsight.ocr.OcrLine
import jp.devfest.shelfsight.ocr.ProductReading
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * 手前の商品のカテゴリの判定（iOS 版と同じ処理）。店で手に持って撮った商品の、Cloud Vision の実際の応答
 * （src/test/resources/lookup/、判定に使う文字と位置だけ残したもの）で確かめる。
 */
class ProductCategoryTest {
    private val rulesText = File("src/main/assets/category_rules.json").readText()
    private val rules = CategoryRules.parse(rulesText)

    private fun lines(name: String): List<OcrLine> {
        val json = javaClass.getResource("/lookup/$name.json")!!.readText()
        val image = JSONObject(json).getJSONObject("image")
        return CloudVisionParser.lines(json, image.getInt("height"), image.getInt("width"))
    }

    private fun judge(name: String, client: CloudVisionClient? = null): CategoryJudge.Outcome {
        val ls = lines(name)
        val reading = ProductReading.from(ls)
        return CategoryJudge(rules, client).judge(ls, if (reading.readable) reading.name else "")
    }

    private val noCall = CloudVisionClient { _, _ -> error("Gemini must not be called") }

    // ---------------------------------------------------------------- 辞書

    @Test
    fun `the dictionary is the same file as the iOS one`() {
        val ios = File("../ios/ShelfSight/Assist/category_rules.json")
        if (ios.exists()) assertEquals(ios.readText().replace("\r\n", "\n"), rulesText.replace("\r\n", "\n"))
    }

    @Test
    fun `every rule names a known category`() {
        assertTrue(rules.rules.size > 100)
        for (r in rules.rules) if (!r.alcoholCue) assertTrue(r.label, r.category in rules.categories)
        assertTrue(rules.alcoholCategories.all { it in rules.categories })
    }

    @Test
    fun `english words match only as whole words`() {
        val tea = rules.score(listOf(OcrLine("STEAK SAUCE", 0.05f, 0.45f, 0.55f, 0.3f, 0.7f)))
        assertTrue(tea.ranked.none { it.category == "紅茶" })
        val gum = rules.score(listOf(OcrLine("Hard Gummy", 0.05f, 0.45f, 0.55f, 0.3f, 0.7f)))
        assertEquals("グミ", gum.top!!.category)
        assertTrue(gum.ranked.none { it.category == "キャンディ・ガム・タブレット" })
    }

    @Test
    fun `full width letters are matched like half width ones`() {
        assertEquals("炭酸飲料", rules.score(listOf(OcrLine("ＷＩＬＫＩＮＳＯＮ", 0.05f, 0.45f, 0.55f, 0.3f, 0.7f))).top!!.category)
    }

    @Test
    fun `text at the edge of the crop counts less than text in the center`() {
        val center = rules.score(listOf(OcrLine("紅茶", 0.05f, 0.45f, 0.55f, 0.45f, 0.55f))).top!!.score
        val edge = rules.score(listOf(OcrLine("紅茶", 0.05f, 0.0f, 0.05f, 0.0f, 0.1f))).top!!.score
        assertTrue(edge < center * 0.6f)
    }

    // ---------------------------------------------------------------- 実際に撮った商品

    @Test
    fun `ninja meshi is a gummy, decided on the phone`() {
        val a = judge("ninja_meshi", noCall).answer!!
        assertEquals("グミ", a.category)
        assertEquals("忍者めし", a.product)
        assertEquals("忍者めし。グミです。", a.spoken())
    }

    @Test
    fun `wilkinson is sparkling water, decided on the phone`() {
        val a = judge("wilkinson", noCall).answer!!
        assertEquals("炭酸飲料", a.category)
        assertEquals("ウィルキンソン。炭酸水です。", a.spoken())
    }

    @Test
    fun `craft boss is coffee even with tea cans behind it`() {
        val a = judge("craft_boss", noCall).answer!!
        assertEquals("コーヒー", a.category)
        assertFalse(a.alcohol)
    }

    @Test
    fun `sapporo black label is said to be alcohol`() {
        val a = judge("sapporo", noCall).answer!!
        assertEquals("ビール類", a.category)
        assertTrue(a.alcohol)
        assertEquals("黒ラベル。お酒、ビールです。", a.spoken())
    }

    @Test
    fun `coca cola is a cola`() {
        val a = judge("coca_cola", noCall).answer!!
        assertEquals("炭酸飲料", a.category)
        assertEquals("コーラ", a.kind)
    }

    @Test
    fun `a soy sauce flavor is not sure on the phone, so gemini decides`() {
        var prompt = ""
        val gemini = CloudVisionClient { task, jpeg ->
            assertNull(jpeg)   // 画像は送らない
            prompt = task.prompt
            """{"category":"せんべい・米菓","kind":"おせんべい","product":"醤油あられ","alcohol":false}"""
        }
        val o = judge("rice_crackers", gemini)
        assertTrue("Soy Sauce Flavored Rice Crackers" in prompt)
        val a = o.answer!!
        assertEquals("Gemini", a.source)
        assertEquals("醤油あられ。おせんべいです。", a.spoken())
        assertNotNull(o.geminiMs)
    }

    @Test
    fun `when gemini fails, the phone's best guess is said softly`() {
        val down = CloudVisionClient { _, _ -> throw CloudException(CloudFailure.NETWORK, "down") }
        val o = judge("rice_crackers", down)
        val a = o.answer!!
        assertEquals("せんべい・米菓", a.category)
        assertFalse(a.sure)
        assertTrue(a.spoken().endsWith("おせんべいのようです。"))
        assertTrue(o.geminiError!!.contains("NETWORK"))
    }

    @Test
    fun `gemini's alcohol answer and an unknown category are handled`() {
        val ls = listOf(OcrLine("よくわからない文字", 0.05f, 0.45f, 0.55f, 0.3f, 0.7f))
        val a = CategoryJudge(rules, CloudVisionClient { _, _ -> """{"category":"不明","kind":"","product":"","alcohol":false}""" }).judge(ls, "よくわからない文字").answer!!
        assertNull(a.category)
        assertEquals("よくわからない文字。種類は分かりませんでした。", a.spoken())
    }

    @Test
    fun `alcohol hints next to a non alcohol category are mentioned`() {
        // 年齢確認の掲示とビールの文字が端にあるが、中央はコーラ
        val ls = listOf(
            OcrLine("コカ・コーラ", 0.05f, 0.45f, 0.55f, 0.3f, 0.7f),
            OcrLine("お酒は20歳になってから", 0.03f, 0.0f, 0.04f, 0.0f, 0.6f),
            OcrLine("生ビール", 0.03f, 0.85f, 0.9f, 0.6f, 1f),
        )
        val a = CategoryJudge(rules, null).judge(ls, "").answer!!
        assertEquals("炭酸飲料", a.category)
        assertTrue(a.maybeAlcohol)
        assertTrue(a.spoken().endsWith("お酒かもしれません。"))
    }

    @Test
    fun `non alcoholic beer is not said to be alcohol`() {
        val ls = listOf(OcrLine("ノンアルコール ビールテイスト", 0.05f, 0.45f, 0.55f, 0.3f, 0.7f))
        val a = CategoryJudge(rules, null).judge(ls, "").answer!!
        assertFalse(a.alcohol)
        assertFalse(a.maybeAlcohol)
        assertTrue(a.kind!!.contains("ノンアル"))
    }

    @Test
    fun `no text gives no answer and does not call gemini`() {
        assertNull(CategoryJudge(rules, noCall).judge(emptyList(), "").answer)
    }

    // ---------------------------------------------------------------- 読み上げと要求

    @Test
    fun `the product name comes first, then the category`() {
        fun a(product: String, kind: String?, sure: Boolean = true, alcohol: Boolean = false) =
            ProductAnswer(product, kind?.let { "x" }, kind, sure, alcohol, false, "", "")
        assertEquals("黒ラベル。お酒、ビールです。", a("黒ラベル", "ビール", alcohol = true).spoken())
        assertEquals("グミのようです。", a("", "グミ", sure = false).spoken())
        assertEquals("ポテト。種類は分かりませんでした。", a("ポテト", null).spoken())
    }

    @Test
    fun `the request asks only for the printed text`() {
        val r = VisionProductParser.request("AAAA")
        assertTrue("DOCUMENT_TEXT_DETECTION" in r)
        assertFalse("WEB_DETECTION" in r || "LOGO_DETECTION" in r)
    }

    @Test
    fun `ocr lines carry their horizontal position`() {
        val l = lines("wilkinson").first { "WILKINSON" in it.text }
        assertTrue(l.left in 0.5f..0.9f && l.right > l.left)
    }

    // ---------------------------------------------------------------- 長押しで何を調べるか

    @Test
    fun `a near product is looked up, a wide shelf is described, otherwise step back`() {
        val near = floatArrayOf(0.3f, 0.3f, 0.7f, 0.7f)
        assertEquals(Focus.PRODUCT to near, SceneJudge.focus(emptyList(), near))
        assertEquals(Focus.NOTHING, SceneJudge.focus(emptyList()).first)
        val shelf = (0 until 6).map { i -> floatArrayOf(i * 0.15f, 0.1f + (i % 2) * 0.4f, i * 0.15f + 0.1f, 0.3f + (i % 2) * 0.4f) }
        assertEquals(Focus.SHELF, SceneJudge.focus(shelf).first)
        // 大きな枠 1 つだけ（近いかどうかは枠の大きさでは決めない）
        assertEquals(Focus.STEP_BACK, SceneJudge.focus(listOf(floatArrayOf(0.1f, 0.1f, 0.9f, 0.9f))).first)
    }

    @Test
    fun `the center box is used only when arcore could not judge the near product`() {
        val center = floatArrayOf(0.35f, 0.3f, 0.65f, 0.7f)
        val boxes = listOf(center, floatArrayOf(0.0f, 0.0f, 0.2f, 0.2f))
        // 深度で測れて、近い商品がない（遠い）→ 中央の枠は使わない（iOS 版と同じく一歩下がって）
        val far = SceneJudge.focusWithFallback(boxes, null, depthMeasured = true)
        assertEquals(Focus.STEP_BACK, far.focus)
        assertFalse(far.byCenterBox)
        // 深度がない・測れなかった → 中央の枠を調べる
        val noDepth = SceneJudge.focusWithFallback(boxes, null, depthMeasured = false)
        assertEquals(Focus.PRODUCT, noDepth.focus)
        assertTrue(noDepth.byCenterBox)
        assertTrue(noDepth.box === center)
        // 深度で近い商品が分かった → その範囲（中央の枠ではない）
        val near = floatArrayOf(0.2f, 0.2f, 0.8f, 0.8f)
        val byDepth = SceneJudge.focusWithFallback(boxes, near, depthMeasured = true)
        assertTrue(byDepth.box === near && !byDepth.byCenterBox)
        // 測れなくても、棚の広い範囲なら棚の説明のまま
        val shelf = (0 until 6).map { i -> floatArrayOf(i * 0.15f, 0.1f + (i % 2) * 0.4f, i * 0.15f + 0.1f, 0.3f + (i % 2) * 0.4f) }
        assertEquals(Focus.SHELF, SceneJudge.focusWithFallback(shelf, null, depthMeasured = false).focus)
    }

    @Test
    fun `a near product makes the scene one product`() {
        val j = SceneJudge()
        repeat(2) { j.update(SceneEvidence(0, emptyList(), near = true)) }
        assertEquals(Scene.ONE_PRODUCT, j.scene)
    }
}
