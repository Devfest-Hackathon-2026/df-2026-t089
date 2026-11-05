package jp.devfest.shelfsight

import java.io.File
import java.io.IOException
import jp.devfest.shelfsight.cloud.CloudConfidence
import jp.devfest.shelfsight.cloud.CloudException
import jp.devfest.shelfsight.cloud.CloudFailure
import jp.devfest.shelfsight.cloud.CloudResponseParser
import jp.devfest.shelfsight.cloud.CloudTask
import jp.devfest.shelfsight.cloud.CloudTasks
import jp.devfest.shelfsight.labeling.DwellDetector
import jp.devfest.shelfsight.cloud.GeminiClient
import jp.devfest.shelfsight.labeling.HandCandidate
import jp.devfest.shelfsight.labeling.HeldProductDetector
import jp.devfest.shelfsight.cloud.HttpResult
import jp.devfest.shelfsight.cloud.HttpTransport
import jp.devfest.shelfsight.labeling.RequestBudget
import jp.devfest.shelfsight.labeling.SceneChangeDetector
import jp.devfest.shelfsight.labeling.SceneSignature
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.ocr.OcrLine
import jp.devfest.shelfsight.ocr.ProductReading
import jp.devfest.shelfsight.model.Vec3
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class CloudLabelingTest {
    private val categories = listOf("カップ麺", "酒", "その他")
    private val task = CloudTask("prompt", """{"type":"object"}""", "high")
    private val completed = """{"status":"completed","steps":[
        {"type":"thought","content":[{"type":"text","text":"hmm"}]},
        {"type":"model_output","content":[{"type":"text","text":"{\"a\":1}"}]}]}"""

    private fun client(vararg responses: Any, sleeps: MutableList<Long> = mutableListOf(), requests: MutableList<String> = mutableListOf(), headers: MutableList<Map<String, String>> = mutableListOf()): GeminiClient {
        val queue = ArrayDeque(responses.toList())
        val transport = HttpTransport { _, h, body ->
            requests += body
            headers += h
            when (val next = queue.removeFirst()) {
                is IOException -> throw next
                is HttpResult -> next
                else -> error("bad fake")
            }
        }
        return GeminiClient("KEY", listOf("test-model", "backup-model"), transport) { sleeps += it }
    }

    @Test
    fun `the request carries the key in a header and asks not to store the photo`() {
        val requests = mutableListOf<String>()
        val headers = mutableListOf<Map<String, String>>()
        val result = client(HttpResult(200, completed), requests = requests, headers = headers).generate(task, byteArrayOf(1, 2))
        assertEquals("""{"a":1}""", result)

        val body = JSONObject(requests.single())
        assertEquals("test-model", body.getString("model"))
        assertEquals(false, body.getBoolean("store"))
        val image = body.getJSONArray("input").getJSONObject(1)
        assertEquals("AQI=", image.getString("data"))
        assertEquals("high", image.getString("resolution"))
        assertEquals("application/json", body.getJSONObject("response_format").getString("mime_type"))
        assertEquals("KEY", headers.single()["x-goog-api-key"])
        assertFalse(requests.single().contains("KEY"))
    }

    @Test
    fun `server errors and network errors are retried, then succeed`() {
        val sleeps = mutableListOf<Long>()
        val result = client(HttpResult(503, ""), IOException("down"), HttpResult(200, completed), sleeps = sleeps).generate(task, ByteArray(1))
        assertEquals("""{"a":1}""", result)
        assertEquals(listOf(300L, 300L), sleeps)
    }

    @Test
    fun `a busy model falls over to the next one`() {
        val requests = mutableListOf<String>()
        client(HttpResult(503, ""), HttpResult(200, completed), requests = requests).generate(task, ByteArray(1))
        assertEquals(listOf("test-model", "backup-model"), requests.map { JSONObject(it).getString("model") })
    }

    @Test
    fun `a bad key is not retried`() {
        val sleeps = mutableListOf<Long>()
        val e = assertThrows(CloudException::class.java) { client(HttpResult(403, ""), sleeps = sleeps).generate(task, ByteArray(1)) }
        assertEquals(CloudFailure.AUTH, e.failure)
        assertTrue(sleeps.isEmpty())
    }

    @Test
    fun `retries stop after the limit`() {
        val e = assertThrows(CloudException::class.java) {
            client(*Array(5) { HttpResult(429, "") }).generate(task, ByteArray(1))
        }
        assertEquals(CloudFailure.RATE_LIMIT, e.failure)
    }

    @Test
    fun `an interaction that did not complete is a failure even with HTTP 200`() {
        val body = """{"status":"failed","errors":[{"code":"x","message":"no"}]}"""
        val e = assertThrows(CloudException::class.java) { client(HttpResult(200, body)).generate(task, ByteArray(1)) }
        assertEquals(CloudFailure.BAD_RESPONSE, e.failure)
    }

    private val parser = CloudResponseParser(categories)

    @Test
    fun `columns are sorted, clamped and overlaps are removed`() {
        val result = parser.parseShelfColumns(
            """{"is_shelf_front":true,"columns":[
              {"x_start":500,"x_end":1200,"category":"酒","confidence":"high","evidence":"ビール"},
              {"x_start":0,"x_end":600,"category":"カップ麺","confidence":"medium","evidence":""}]}""",
        )
        assertTrue(result.isShelfFront)
        assertEquals(listOf("カップ麺", "酒"), result.columns.map { it.category })
        assertEquals(0.5f, result.columns[0].xEnd, 1e-4f)
        assertEquals(1f, result.columns[1].xEnd, 1e-4f)
        assertEquals(CloudConfidence.HIGH, result.columns[1].confidence)
    }

    @Test
    fun `an unknown category becomes other and a sliver is dropped`() {
        val result = parser.parseShelfColumns(
            """{"is_shelf_front":true,"columns":[
              {"x_start":0,"x_end":500,"category":"謎の棚","confidence":"low","evidence":""},
              {"x_start":600,"x_end":605,"category":"酒","confidence":"low","evidence":""}]}""",
        )
        assertEquals("その他", result.columns.single().category)
    }

    @Test
    fun `columns are ignored when the photo is not a shelf front`() {
        val result = parser.parseShelfColumns(
            """{"is_shelf_front":false,"columns":[{"x_start":0,"x_end":500,"category":"酒","confidence":"low","evidence":""}]}""",
        )
        assertFalse(result.isShelfFront)
        assertTrue(result.columns.isEmpty())
    }

    @Test
    fun `a broken output is reported as a bad response`() {
        val e = assertThrows(CloudException::class.java) { parser.parseShelfColumns("not json") }
        assertEquals(CloudFailure.BAD_RESPONSE, e.failure)
    }

    @Test
    fun `the bundled prompts and schemas are filled with every category`() {
        val all = File("src/main/assets/labels.txt").readLines().filter { it.isNotBlank() }
        assertEquals(26, all.size)
        val tasks = CloudTasks({ File("src/main/assets", it).readText() }, all)
        for (t in listOf(tasks.shelfColumns)) {
            assertFalse(t.prompt.contains("{{"))
            assertFalse(t.schemaJson.contains("{{"))
            JSONObject(t.schemaJson)
            all.forEach { assertTrue(it, t.prompt.contains(it)) }
        }
        val columnEnum = JSONObject(tasks.shelfColumns.schemaJson)
            .getJSONObject("properties").getJSONObject("columns").getJSONObject("items")
            .getJSONObject("properties").getJSONObject("category").getJSONArray("enum")
        assertEquals(26, columnEnum.length())
    }

    private val here = Vec3(0f, 0f, 0f)
    private val ahead = Vec3(0f, 0f, -1f)

    @Test
    fun `standing still fires once, and only after the dwell time`() {
        val dwell = DwellDetector(dwellSeconds = 1.5)
        assertFalse(dwell.update(0.0, here, ahead))
        assertFalse(dwell.update(1.0, here, ahead))
        assertTrue(dwell.update(1.6, here, ahead))
        assertFalse(dwell.update(5.0, here, ahead))
    }

    @Test
    fun `walking never fires`() {
        val dwell = DwellDetector(dwellSeconds = 1.5)
        for (i in 0..100) assertFalse(dwell.update(i * 0.1, Vec3(i * 0.1f, 0f, 0f), ahead))
    }

    @Test
    fun `chest sway while standing still does not reset the dwell`() {
        val dwell = DwellDetector(dwellSeconds = 1.5)
        dwell.update(0.0, here, ahead)
        dwell.update(0.8, Vec3(0.05f, 0.03f, 0f), ahead)
        assertTrue(dwell.update(1.6, Vec3(-0.05f, 0f, 0f), ahead))
    }

    @Test
    fun `stopping again at the same spot does not fire, but a new shelf does`() {
        val dwell = DwellDetector(dwellSeconds = 1.5)
        dwell.update(0.0, here, ahead)
        assertTrue(dwell.update(2.0, here, ahead))
        val slightly = Vec3(0.13f, 0f, 0f) // 動いたが、まだ同じ棚の前
        dwell.update(3.0, slightly, ahead)
        assertFalse(dwell.update(5.0, slightly, ahead))
        val nextShelf = Vec3(1f, 0f, 0f)
        dwell.update(6.0, nextShelf, ahead)
        assertTrue(dwell.update(8.0, nextShelf, ahead))
    }

    @Test
    fun `a failed request can fire again without moving`() {
        val dwell = DwellDetector(dwellSeconds = 1.5)
        dwell.update(0.0, here, ahead)
        assertTrue(dwell.update(2.0, here, ahead))
        dwell.retry()
        assertTrue(dwell.update(2.1, here, ahead))
    }

    private fun hand(x: Float, pointing: Boolean = false, distance: Float? = 0.4f) =
        HandCandidate(NormalizedBox(x, 0.4f, x + 0.2f, 0.6f), pointing, distance)

    @Test
    fun `a steady hand fires once until it is lowered`() {
        val detector = HeldProductDetector(holdSeconds = 0.8)
        assertNull(detector.update(0.0, listOf(hand(0.4f))))
        assertNull(detector.update(0.5, listOf(hand(0.41f))))
        assertNotNull(detector.update(0.9, listOf(hand(0.4f))))
        assertNull(detector.update(5.0, listOf(hand(0.4f))))

        assertNull(detector.update(6.0, emptyList()))
        assertNull(detector.update(7.0, listOf(hand(0.4f))))
        assertNotNull(detector.update(8.0, listOf(hand(0.4f))))
    }

    @Test
    fun `a moving hand, a pointing hand and a distant hand do not fire`() {
        val moving = HeldProductDetector(holdSeconds = 0.8)
        for (i in 0..20) assertNull(moving.update(i * 0.2, listOf(hand(0.1f + (i % 2) * 0.3f))))
        val pointing = HeldProductDetector(holdSeconds = 0.8)
        for (i in 0..20) assertNull(pointing.update(i * 0.2, listOf(hand(0.4f, pointing = true))))
        val distant = HeldProductDetector(holdSeconds = 0.8)
        for (i in 0..20) assertNull(distant.update(i * 0.2, listOf(hand(0.4f, distance = 2f))))
    }

    private val room = FloatArray(256) { 100f + (it % 5) }
    private val box = FloatArray(256) { if (it % 2 == 0) 220f else 40f }

    @Test
    fun `a product appearing and staying still fires once`() {
        val scene = SceneChangeDetector(stableSeconds = 0.8)
        assertFalse(scene.update(0.0, room))
        assertFalse(scene.update(0.2, room))
        assertFalse(scene.update(0.4, box))
        assertFalse(scene.update(0.8, box))
        assertTrue(scene.update(1.3, box))
        assertFalse(scene.update(1.5, box))
        assertFalse(scene.update(5.0, box))
    }

    @Test
    fun `a scene that keeps changing never fires`() {
        val scene = SceneChangeDetector(stableSeconds = 0.8)
        scene.update(0.0, room)
        for (i in 1..50) {
            val moving = FloatArray(256) { cell -> if ((cell + i) % 2 == 0) 220f else 40f }
            assertFalse(scene.update(i * 0.2, moving))
        }
    }

    @Test
    fun `small flicker and sensor noise do not fire`() {
        val scene = SceneChangeDetector(stableSeconds = 0.8)
        scene.update(0.0, room)
        for (i in 1..50) assertFalse(scene.update(i * 0.2, FloatArray(256) { room[it] + (i % 3) * 5f }))
    }

    @Test
    fun `a phone laid face down does not fire`() {
        val scene = SceneChangeDetector(stableSeconds = 0.8)
        scene.update(0.0, room)
        for (i in 1..20) assertFalse(scene.update(i * 0.2, FloatArray(256) { 5f }))
    }

    @Test
    fun `moving the camera resets what counts as new`() {
        val scene = SceneChangeDetector(stableSeconds = 0.8)
        scene.update(0.0, room)
        scene.rebaseline(box)
        for (i in 1..20) assertFalse(scene.update(i * 0.2, box))
    }

    @Test
    fun `the signature follows the image and stays inside it`() {
        val width = 64
        val height = 36
        val bright = ByteArray(width * height * 3 / 2) { 200.toByte() }
        assertEquals(200f, SceneSignature.of(bright, width, height).average().toFloat(), 0.01f)
        assertEquals(256, SceneSignature.of(bright, width, height).size)
    }

    @Test
    fun `taking the product away does not read again, but the next product does`() {
        val scene = SceneChangeDetector(stableSeconds = 0.8)
        scene.update(0.0, room)
        scene.update(0.2, box)
        scene.update(0.4, box)
        assertTrue(scene.update(1.3, box))

        scene.update(2.0, room)
        scene.update(2.2, room)
        assertFalse(scene.update(3.1, room))

        val other = FloatArray(256) { if (it % 2 == 0) 40f else 220f }
        scene.update(4.0, other)
        scene.update(4.2, other)
        assertTrue(scene.update(5.1, other))
    }

    /** 実際の箱（2種のぶどうフルーツバー）を真似た行。ロゴの L がいちばん大きい。 */
    private val boxLines = listOf(
        OcrLine("L", 0.03f, 0.31f),
        OcrLine("2種のぶどうフルーツバー", 0.022f, 0.30f),
        OcrLine("Fruits Ice Bar", 0.008f, 0.33f),
        OcrLine("グレープ", 0.006f, 0.44f),
        OcrLine("8本入", 0.008f, 0.49f),
        OcrLine("要冷凍", 0.012f, 0.52f),
    )

    @Test
    fun `the name is the largest real text, not the logo letter`() {
        val product = ProductReading.from(boxLines)
        assertTrue(product.readable)
        assertEquals("2種のぶどうフルーツバー", product.name)
        assertEquals(6, product.ocrText.size)
    }

    @Test
    fun `a name printed on two lines is joined`() {
        val product = ProductReading.from(listOf(OcrLine("特濃", 0.03f, 0.40f), OcrLine("ミルクチョコ", 0.028f, 0.44f), OcrLine("内容量", 0.008f, 0.6f)))
        assertEquals("特濃 ミルクチョコ", product.name)
    }

    @Test
    fun `a short Japanese name is accepted`() {
        assertEquals("お茶", ProductReading.from(listOf(OcrLine("お茶", 0.03f, 0.4f))).name)
    }

    @Test
    fun `small background text or digits alone are not a product`() {
        assertFalse(ProductReading.from(listOf(OcrLine("新商品のお知らせ", 0.005f, 0.1f))).readable)
        assertFalse(ProductReading.from(listOf(OcrLine("500", 0.05f, 0.1f))).readable)
        assertFalse(ProductReading.from(emptyList()).readable)
    }

    @Test
    fun `the budget limits requests per minute and recovers`() {
        val budget = RequestBudget(maxPerMinute = 2)
        assertTrue(budget.tryAcquire(0.0))
        assertTrue(budget.tryAcquire(1.0))
        assertFalse(budget.tryAcquire(2.0))
        assertTrue(budget.tryAcquire(61.0))
    }

}
