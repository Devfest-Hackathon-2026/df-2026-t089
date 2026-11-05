package jp.devfest.shelfsight.cloud

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.Base64
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/** 画像と指示を渡して、スキーマに沿った JSON 文字列を受け取る。別のサービスへ替えるときはここを実装する。 */
internal fun interface CloudVisionClient {
    /** ネットワーク待ちで止まるので、GL スレッドでは呼ばない。 */
    @Throws(CloudException::class)
    fun generate(task: CloudTask, jpeg: ByteArray): String
}

internal enum class CloudFailure(val messageJa: String) {
    AUTH("API キーを確認してください"),
    RATE_LIMIT("利用上限に達しました。しばらく待ちます"),
    NETWORK("通信できません"),
    SERVER("クラウド側のエラーです"),
    BAD_RESPONSE("判別結果を読み取れませんでした"),
}

internal class CloudException(val failure: CloudFailure, detail: String, cause: Throwable? = null) :
    Exception("$failure: $detail", cause)

internal class HttpResult(val code: Int, val body: String)

internal fun interface HttpTransport {
    @Throws(IOException::class)
    fun post(url: String, headers: Map<String, String>, body: String): HttpResult
}

internal object UrlConnectionTransport : HttpTransport {
    private const val CONNECT_TIMEOUT_MS = 10_000

    /** 立ち止まっている間に答えが欲しいので、待ち続けずに次のモデルへ切り替える。 */
    private const val READ_TIMEOUT_MS = 12_000

    override fun post(url: String, headers: Map<String, String>, body: String): HttpResult {
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = CONNECT_TIMEOUT_MS
            connection.readTimeout = READ_TIMEOUT_MS
            connection.doOutput = true
            headers.forEach { (name, value) -> connection.setRequestProperty(name, value) }
            connection.outputStream.use { it.write(body.toByteArray()) }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            return HttpResult(code, stream?.use { it.readBytes().decodeToString() }.orEmpty())
        } finally {
            connection.disconnect()
        }
    }
}

/**
 * Gemini の Interactions API を HTTP で直接呼ぶ。
 *
 * モデルは混雑で頻繁に 503 を返すので、失敗したら待つだけでなく次のモデルに切り替える。
 * 店内では立ち止まっている間に答えを返したいので、待ち時間は短くする。
 *
 * @param models 試す順に並べたモデル
 */
internal class GeminiClient(
    private val apiKey: String,
    private val models: List<String>,
    private val transport: HttpTransport = UrlConnectionTransport,
    private val sleep: (Long) -> Unit = Thread::sleep,
) : CloudVisionClient {

    override fun generate(task: CloudTask, jpeg: ByteArray): String {
        val headers = mapOf(
            "x-goog-api-key" to apiKey,
            "Content-Type" to "application/json",
            "Api-Revision" to API_REVISION,
        )
        var attempt = 0
        while (true) {
            val model = models[attempt % models.size]
            try {
                return handle(transport.post(ENDPOINT, headers, requestBody(task, jpeg, model)))
            } catch (e: IOException) {
                if (attempt >= RETRY_DELAYS_MS.size) throw CloudException(CloudFailure.NETWORK, "request failed", e)
            } catch (e: CloudException) {
                val retryable = e.failure == CloudFailure.RATE_LIMIT || e.failure == CloudFailure.SERVER
                if (!retryable || attempt >= RETRY_DELAYS_MS.size) throw e
            }
            sleep(RETRY_DELAYS_MS[attempt])
            attempt++
        }
    }

    private fun requestBody(task: CloudTask, jpeg: ByteArray, model: String): String = JSONObject()
        .put("model", model)
        // 店内の写真には来店者が写ることがあるので、応答を返したら保存させない。
        .put("store", false)
        .put(
            "input",
            JSONArray()
                .put(JSONObject().put("type", "text").put("text", task.prompt))
                .put(
                    JSONObject()
                        .put("type", "image")
                        .put("data", Base64.getEncoder().encodeToString(jpeg))
                        .put("mime_type", "image/jpeg")
                        .put("resolution", task.imageResolution),
                ),
        )
        .put(
            "response_format",
            JSONObject()
                .put("type", "text")
                .put("mime_type", "application/json")
                .put("schema", JSONObject(task.schemaJson)),
        )
        .toString()

    private fun handle(response: HttpResult): String {
        when {
            response.code == 401 || response.code == 403 -> throw CloudException(CloudFailure.AUTH, "HTTP ${response.code}")
            response.code == 429 -> throw CloudException(CloudFailure.RATE_LIMIT, "HTTP 429")
            response.code >= 500 -> throw CloudException(CloudFailure.SERVER, "HTTP ${response.code}")
            // 本文は Google のエラーメッセージで、キーや画像を含まない。原因を調べられるようログ用に残す。
            response.code !in 200..299 ->
                throw CloudException(CloudFailure.BAD_RESPONSE, "HTTP ${response.code}: ${response.body.take(MAX_ERROR_CHARS)}")
        }
        return extractText(response.body)
    }

    /** 応答の `steps` から、モデルの出力の文章を取り出す。思考のステップは読み飛ばす。 */
    private fun extractText(body: String): String {
        val root = try {
            JSONObject(body)
        } catch (e: JSONException) {
            throw CloudException(CloudFailure.BAD_RESPONSE, "not JSON", e)
        }
        val status = root.optString("status")
        if (status != "completed") {
            throw CloudException(CloudFailure.BAD_RESPONSE, "status=$status errors=${root.opt("errors")}")
        }
        val steps = root.optJSONArray("steps")
        val text = StringBuilder()
        for (i in 0 until (steps?.length() ?: 0)) {
            val step = steps!!.optJSONObject(i) ?: continue
            if (step.optString("type") != "model_output") continue
            val content = step.optJSONArray("content") ?: continue
            for (j in 0 until content.length()) {
                val part = content.optJSONObject(j) ?: continue
                if (part.optString("type") == "text") text.append(part.optString("text"))
            }
        }
        if (text.isEmpty()) throw CloudException(CloudFailure.BAD_RESPONSE, "no text in response")
        return text.toString()
    }

    private companion object {
        const val ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
        const val API_REVISION = "2026-05-20"
        const val MAX_ERROR_CHARS = 500
        val RETRY_DELAYS_MS = longArrayOf(300L, 300L, 1_000L, 3_000L)
    }
}
