package jp.devfest.shelfsight.ocr

import android.graphics.Bitmap
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.Base64
import java.util.concurrent.atomic.AtomicInteger
import jp.devfest.shelfsight.cloud.HttpTransport
import jp.devfest.shelfsight.cloud.UrlConnectionTransport
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import kotlin.math.max

/**
 * Google Cloud Vision API の文字検出（DOCUMENT_TEXT_DETECTION）で読む。ML Kit より日本語のパッケージに強い。
 *
 * 画像は Google に送られるが、同期の要求ではメモリ上で処理され、保存も学習への利用もされない（Vision API のデータ利用の説明による）。
 *
 * @param maxRequests アプリを起動している間に送る上限。無料枠（月 1000 件）を使い切らないためのもの。超えたら読まずに失敗する
 */
internal class CloudVisionOcr(
    private val apiKey: String,
    private val maxRequests: Int = Int.MAX_VALUE,
    private val transport: HttpTransport = UrlConnectionTransport,
) : TextOcr {
    private val sent = AtomicInteger(0)

    override fun read(bitmap: Bitmap): List<OcrLine> {
        check(sent.incrementAndGet() <= maxRequests) { "Cloud Vision request limit ($maxRequests) reached" }
        val jpeg = ByteArrayOutputStream().use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
            out.toByteArray()
        }
        val body = JSONObject().put(
            "requests",
            JSONArray().put(
                JSONObject()
                    .put("image", JSONObject().put("content", Base64.getEncoder().encodeToString(jpeg)))
                    .put("features", JSONArray().put(JSONObject().put("type", "DOCUMENT_TEXT_DETECTION")))
                    .put("imageContext", JSONObject().put("languageHints", JSONArray().put("ja"))),
            ),
        ).toString()
        val result = try {
            transport.post(URL, mapOf("x-goog-api-key" to apiKey, "Content-Type" to "application/json"), body)
        } catch (e: IOException) {
            throw IllegalStateException("Cloud Vision unreachable", e)
        }
        check(result.code in 200..299) { "Cloud Vision HTTP ${result.code}: ${result.body.take(300)}" }
        return CloudVisionParser.lines(result.body, bitmap.height)
    }

    private companion object {
        const val URL = "https://vision.googleapis.com/v1/images:annotate"
        const val JPEG_QUALITY = 90
    }
}

/** Vision API の応答を、ML Kit と同じ形の行に直す。商品名の選び方（[ProductReading]）をそのまま使えるようにする。 */
internal object CloudVisionParser {
    fun lines(json: String, imageHeight: Int): List<OcrLine> {
        val response = try {
            JSONObject(json).getJSONArray("responses").getJSONObject(0)
        } catch (e: JSONException) {
            throw IllegalStateException("Unexpected Cloud Vision response", e)
        }
        response.optJSONObject("error")?.let { error("Cloud Vision error: ${it.optString("message")}") }
        val pages = response.optJSONObject("fullTextAnnotation")?.optJSONArray("pages") ?: return emptyList()
        val out = mutableListOf<OcrLine>()
        val text = StringBuilder()
        val heights = mutableListOf<Int>()
        var top = Int.MAX_VALUE

        fun endLine() {
            if (text.isNotBlank() && heights.isNotEmpty()) {
                // 行全体の枠は、箱が少し傾くだけで高くなるので、ML Kit と同じく 1 文字ずつの高さの中央値を使う。
                val median = heights.sorted()[heights.size / 2]
                out.add(OcrLine(text.toString().trim(), median / imageHeight.toFloat(), top / imageHeight.toFloat()))
            }
            text.clear()
            heights.clear()
            top = Int.MAX_VALUE
        }

        for (symbol in symbols(pages)) {
            text.append(symbol.optString("text"))
            val ys = vertices(symbol).map { it.second }
            if (ys.isNotEmpty()) {
                heights.add(ys.max() - ys.min())
                top = minOf(top, ys.min())
            }
            when (symbol.optJSONObject("property")?.optJSONObject("detectedBreak")?.optString("type")) {
                "SPACE", "SURE_SPACE" -> text.append(' ')
                "EOL_SURE_SPACE", "LINE_BREAK" -> endLine()
                // 改行のハイフンは単語の途中なので行を続ける。
                else -> Unit
            }
        }
        endLine()
        return out
    }

    private fun symbols(pages: JSONArray) = sequence {
        for (page in pages.objects()) for (block in page.array("blocks").objects()) {
            for (paragraph in block.array("paragraphs").objects()) {
                for (word in paragraph.array("words").objects()) yieldAll(word.array("symbols").objects())
            }
        }
    }

    /** 0 の座標は省略されて返る。 */
    private fun vertices(o: JSONObject): List<Pair<Int, Int>> =
        o.optJSONObject("boundingBox")?.optJSONArray("vertices")?.objects()
            ?.map { max(0, it.optInt("x")) to max(0, it.optInt("y")) }?.toList().orEmpty()

    private fun JSONObject.array(name: String) = optJSONArray(name) ?: JSONArray()

    private fun JSONArray.objects() = (0 until length()).asSequence().mapNotNull { optJSONObject(it) }
}

/** クラウドで読めなかったとき（通信できない、上限など）は端末内の OCR で読む。 */
internal class FallbackOcr(private val primary: TextOcr, private val fallback: TextOcr) : TextOcr {
    override fun read(bitmap: Bitmap): List<OcrLine> = try {
        primary.read(bitmap)
    } catch (e: IllegalStateException) {
        Log.w(TAG, "Cloud OCR failed, reading on the device instead: ${e.message}")
        fallback.read(bitmap)
    }

    private companion object {
        const val TAG = "FallbackOcr"
    }
}
