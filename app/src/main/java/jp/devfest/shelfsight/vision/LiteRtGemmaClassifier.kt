package jp.devfest.shelfsight.vision

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import java.util.concurrent.Executors
import jp.devfest.shelfsight.model.ShelfDetection
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext

/** LiteRT-LM を使い、Gemma 4 を端末内で実行する分類器。画像は端末の外に送信しない。 */
class LiteRtGemmaClassifier(
    private val modelPath: String,
    private val cacheDir: String,
) : ShelfClassifier {

    // GPU コンテキストはスレッドに紐づくことがあるため、エンジンの操作は常に同じスレッドで行う。
    private val executor = Executors.newSingleThreadExecutor { Thread(it, "gemma-engine") }
    private val dispatcher = executor.asCoroutineDispatcher()
    private var engine: Engine? = null

    override suspend fun prepare() = withContext(dispatcher) {
        if (engine != null) return@withContext
        engine = try {
            createEngine(Backend.GPU())
        } catch (e: Exception) {
            Log.w(TAG, "GPU backend unavailable, falling back to CPU", e)
            createEngine(Backend.CPU())
        }
    }

    private fun createEngine(backend: Backend): Engine {
        val created = Engine(
            EngineConfig(
                modelPath = modelPath,
                backend = backend,
                visionBackend = backend,
                cacheDir = cacheDir,
            ),
        )
        try {
            created.initialize()
        } catch (e: Exception) {
            created.close()
            throw e
        }
        return created
    }

    override suspend fun classify(jpeg: ByteArray): List<ShelfDetection> = withContext(dispatcher) {
        val readyEngine = checkNotNull(engine) { "prepare() must complete before classify()" }
        // 毎回新しい会話にして、前のフレームの内容が判定に影響しないようにする。
        readyEngine.createConversation(conversationConfig).use { conversation ->
            val reply = conversation.sendMessage(
                Contents.of(Content.ImageBytes(jpeg), Content.Text(ShelfPrompt.text)),
            )
            val text = reply.contents.contents
                .filterIsInstance<Content.Text>()
                .joinToString("") { it.text }
            Log.d(TAG, "Gemma response: $text")
            GemmaResponseParser.parse(text)
        }
    }

    override fun close() {
        executor.execute {
            engine?.close()
            engine = null
        }
        executor.shutdown()
    }

    private companion object {
        const val TAG = "LiteRtGemmaClassifier"
        val conversationConfig = ConversationConfig(
            samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0),
        )
    }
}
