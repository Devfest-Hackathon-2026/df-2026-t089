package jp.devfest.shelfsight.assist

import android.graphics.Bitmap
import android.util.Log
import jp.devfest.shelfsight.cloud.HttpTransport
import jp.devfest.shelfsight.cloud.UrlConnectionTransport
import jp.devfest.shelfsight.ocr.CloudVisionParser
import jp.devfest.shelfsight.ocr.OcrLine
import jp.devfest.shelfsight.ocr.ProductReading
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Base64
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * 手前の商品 1 つを Cloud Vision API で読み、カテゴリと商品名を返す（カテゴリは [CategoryJudge] が決める）。iOS 版の ProductLookup と同じ。
 * 商品に近づいている間（[NearProduct] で近い商品が見え続けている間）は、分かった結果を覚えておき API を呼び直さない。
 * 新しい商品に近づき直したら [newApproach] で忘れる。これで、長押しを何度しても 1 回の接近につき 1 回しか課金されない。
 * 調べるたびに、送った画像・応答・答えを [logDir] に残す（[LookupLog]）。
 */
internal class ProductLookup(
    private val apiKey: String,
    private val judge: CategoryJudge,
    private val logDir: File?,
    private val transport: HttpTransport = UrlConnectionTransport,
) {
    @Volatile private var cached: Result<ProductAnswer?>? = null

    val configured: Boolean get() = apiKey.isNotEmpty()

    /** この接近ですでに調べた結果（まだなら null）。 */
    val cachedResult: Result<ProductAnswer?>? get() = cached

    fun newApproach() {
        cached = null
    }

    /** 調べる。この接近ですでに調べていれば、その結果を返す。重い処理なので main スレッドでは呼ばない。 */
    fun lookup(image: Bitmap): Result<ProductAnswer?> {
        cached?.let { return it }
        val r = runCatching {
            check(configured) { "Cloud Vision API key is not set" }
            val t0 = System.currentTimeMillis()
            val jpeg = ByteArrayOutputStream().use { out ->
                image.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
                out.toByteArray()
            }
            val log = LookupLog(logDir, jpeg, image.width, image.height)
            val body = VisionProductParser.request(Base64.getEncoder().encodeToString(jpeg))
            val res = try {
                transport.post(URL, mapOf("x-goog-api-key" to apiKey, "Content-Type" to "application/json"), body)
            } catch (e: IOException) {
                log.save(null, listOf("result" to "network error: $e", "ms" to "${System.currentTimeMillis() - t0}"))
                throw IllegalStateException("Cloud Vision unreachable", e)
            }
            val visionMs = System.currentTimeMillis() - t0
            if (res.code !in 200..299) {
                log.save(res.body, listOf("result" to "HTTP ${res.code}", "ms" to "$visionMs"))
                error("Cloud Vision HTTP ${res.code}: ${res.body.take(300)}")
            }
            val lines: List<OcrLine> = try {
                CloudVisionParser.lines(res.body, image.height, image.width)
            } catch (e: IllegalStateException) {
                log.save(res.body, listOf("result" to "parse error: ${e.message}", "ms" to "$visionMs"))
                throw e
            }
            val reading = ProductReading.from(lines)
            val outcome = judge.judge(lines, if (reading.readable) reading.name else "")
            val a = outcome.answer
            Log.i(TAG, "lookup: ${a?.describe() ?: "分からない"}, vision ${visionMs}ms, gemini ${outcome.geminiMs?.let { "${it}ms" } ?: "-"}${outcome.geminiError?.let { " ($it)" } ?: ""}")
            log.save(
                res.body,
                listOf(
                    "result" to (if (a == null) "nil" else "ok"),
                    "spoken" to (a?.spoken() ?: "（分からない）"),
                    "category" to (a?.category ?: ""),
                    "kind" to (a?.kind ?: ""),
                    "product" to (a?.product ?: ""),
                    "sure" to (a?.sure?.toString() ?: ""),
                    "alcohol" to (a?.alcohol?.toString() ?: ""),
                    "maybe_alcohol" to (a?.maybeAlcohol?.toString() ?: ""),
                    "source" to (a?.source ?: ""),
                    "ocr_name" to reading.name,
                    "local" to outcome.local.describe(),
                    "hits" to outcome.local.hits.joinToString(" "),
                    "gemini_ms" to (outcome.geminiMs?.toString() ?: ""),
                    "gemini_error" to (outcome.geminiError ?: ""),
                    "vision_ms" to "$visionMs",
                    "ms" to "${System.currentTimeMillis() - t0}",
                ),
            )
            a
        }
        // 分かったときだけ覚える。通信の失敗や「分からない」は覚えない（もう一度長押しすれば、撮り直して調べ直せる）
        if (r.getOrNull() != null) cached = r
        return r
    }

    private companion object {
        const val TAG = "ProductLookup"
        const val URL = "https://vision.googleapis.com/v1/images:annotate"
        const val JPEG_QUALITY = 90
    }
}

/**
 * 調べた記録を残す（失敗したときも。iOS 版の Documents/lookup と同じ書式）。
 * <時刻>.jpg（送った画像）、<時刻>.json（Cloud Vision の応答。通信できなかったときはない）、<時刻>.answer.txt（答え・言った文・点数の内訳・かかった時間）。
 * Android ではアプリの外部ファイルの場所（Android/data/<パッケージ>/files/lookup）に書く。`adb pull` で取り出せる。
 */
internal class LookupLog(private val dir: File?, private val jpeg: ByteArray, private val width: Int, private val height: Int) {
    fun save(response: String?, fields: List<Pair<String, String>>) {
        val d = dir ?: return
        val all = fields + listOf("image" to "${width}x$height", "jpeg_bytes" to "${jpeg.size}")
        WRITER.execute {
            try {
                d.mkdirs()
                val stem = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
                File(d, "$stem.jpg").writeBytes(jpeg)
                if (response != null) File(d, "$stem.json").writeText(response)
                File(d, "$stem.answer.txt").writeText(all.joinToString("\n") { (k, v) -> "$k=${v.replace("\n", " ")}" } + "\n")
            } catch (e: IOException) {
                Log.w("LookupLog", "lookup log failed", e)
            }
        }
    }

    private companion object {
        val WRITER = Executors.newSingleThreadExecutor()
    }
}
