package jp.devfest.shelfsight.assist

import android.graphics.Bitmap
import jp.devfest.shelfsight.cloud.HttpTransport
import jp.devfest.shelfsight.cloud.UrlConnectionTransport
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.Base64

/**
 * 手前の商品 1 つを Cloud Vision API で調べ、社名と商品名を返す。1 回の要求で Web 検索・ロゴ・文字を頼む。
 *
 * 商品に近づいている間（[SceneJudge] が商品 1 つと判定し続けている間）は、結果を覚えておき API を呼び直さない。
 * 新しい商品に近づき直したら [newApproach] で忘れる。これで、長押しを何度しても 1 回の接近につき 1 回しか課金されない。
 */
internal class ProductLookup(
    private val apiKey: String,
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
            val jpeg = ByteArrayOutputStream().use { out ->
                image.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
                out.toByteArray()
            }
            val body = VisionProductParser.request(Base64.getEncoder().encodeToString(jpeg))
            val res = try {
                transport.post(URL, mapOf("x-goog-api-key" to apiKey, "Content-Type" to "application/json"), body)
            } catch (e: IOException) {
                throw IllegalStateException("Cloud Vision unreachable", e)
            }
            check(res.code in 200..299) { "Cloud Vision HTTP ${res.code}: ${res.body.take(300)}" }
            VisionProductParser.parse(res.body, image.height)
        }
        // 通信の失敗は覚えない（もう一度長押しすれば調べ直せる）
        if (r.isSuccess) cached = r
        return r
    }

    private companion object {
        const val URL = "https://vision.googleapis.com/v1/images:annotate"
        const val JPEG_QUALITY = 90
    }
}
