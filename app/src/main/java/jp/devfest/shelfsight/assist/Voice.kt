package jp.devfest.shelfsight.assist

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.util.LruCache
import jp.devfest.shelfsight.model.Vec3
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * 端末の音声合成（TTS）で話す。合成した音を [SpatialMixer] で鳴らすので、棚の種類などを、その棚の方向から聞かせられる。
 * 同じ言葉は合成し直さない（棚の種類のような短い言葉は、2 回目からすぐ鳴る）。
 */
class Voice(context: Context, private val mixer: SpatialMixer) : AutoCloseable {
    private val ready = CountDownLatch(1)
    private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { status ->
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.JAPAN
            tts.setSpeechRate(SPEECH_RATE)
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(id: String) = Unit
                override fun onDone(id: String) { waiting.remove(id)?.countDown() }
                @Deprecated("Deprecated in Java")
                override fun onError(id: String) { waiting.remove(id)?.countDown() }
                override fun onError(id: String, errorCode: Int) { waiting.remove(id)?.countDown() }
            })
        } else {
            Log.e(TAG, "TextToSpeech init failed: $status")
        }
        ready.countDown()
    }
    private val waiting = java.util.concurrent.ConcurrentHashMap<String, CountDownLatch>()
    private val cache = LruCache<String, FloatArray>(CACHE_SIZE)
    private val worker = Executors.newSingleThreadExecutor()
    private val generation = AtomicInteger()
    private val dir = File(context.cacheDir, "tts").apply { mkdirs() }
    private var nextId = 0

    @Volatile var lastSpoken: String = ""
        private set

    /**
     * 話す。[position] があれば、その位置から聞こえる。
     * [interrupt] なら、話している途中の声と、まだ話していない声をやめる（長押しの「調べています」など、すぐ伝えたいとき）。
     */
    fun say(text: String, position: (() -> Vec3?)? = null, interrupt: Boolean = false, remember: Boolean = true, before: SpatialMixer.Sound? = null, onDone: (() -> Unit)? = null) {
        if (text.isBlank()) return
        val spoken = Reading.forSpeech(text)
        val gen = if (interrupt) generation.incrementAndGet().also { mixer.stopSpeech() } else generation.get()
        if (remember) lastSpoken = text
        worker.execute {
            if (gen != generation.get()) return@execute
            val pcm = synth(spoken) ?: return@execute
            if (gen != generation.get()) return@execute
            before?.let(mixer::speak)   // 声と同じ順番の列に入れて、合成を待つ間にずれないようにする
            mixer.speak(SpatialMixer.Sound(pcm, position, onDone = onDone))
        }
    }

    /** 分けて話した文の全体を「もう一度」で言えるように覚える。 */
    fun remember(text: String) {
        lastSpoken = text
    }

    /** よく使う短い言葉を先に合成しておき、言うときにすぐ鳴るようにする。 */
    fun prepare(texts: List<String>) {
        worker.execute { for (t in texts) synth(Reading.forSpeech(t)) }
    }

    /** 話している声と、まだ話していない声をやめる。 */
    fun stop() {
        generation.incrementAndGet()
        mixer.stopSpeech()
    }

    val speaking: Boolean get() = mixer.speaking

    private fun synth(text: String): FloatArray? {
        cache.get(text)?.let { return it }
        if (!ready.await(5, TimeUnit.SECONDS)) return null
        val id = "u${nextId++}"
        val file = File(dir, "$id.wav")
        val done = CountDownLatch(1)
        waiting[id] = done
        if (tts.synthesizeToFile(text, Bundle(), file, id) != TextToSpeech.SUCCESS) {
            waiting.remove(id)
            return null
        }
        if (!done.await(10, TimeUnit.SECONDS)) return null
        return try {
            val (pcm, rate) = Wav.decode(file.readBytes())
            SpatialMixer.resample(pcm, rate).also { cache.put(text, it) }
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "Bad TTS output", e)
            null
        } finally {
            file.delete()
        }
    }

    override fun close() {
        worker.shutdownNow()
        tts.shutdown()
    }

    private companion object {
        const val TAG = "Voice"
        const val SPEECH_RATE = 1.15f
        const val CACHE_SIZE = 64
    }
}

/** WAV（16 ビット PCM）をモノラルの float にする。 */
object Wav {
    fun decode(bytes: ByteArray): Pair<FloatArray, Int> {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size >= 12 && String(bytes, 0, 4) == "RIFF" && String(bytes, 8, 4) == "WAVE") { "not a WAV file" }
        var pos = 12
        var channels = 1
        var rate = 0
        var bits = 16
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4)
            val size = b.getInt(pos + 4)
            val body = pos + 8
            when (id) {
                "fmt " -> {
                    val format = b.getShort(body).toInt() and 0xFFFF
                    require(format == 1 || format == 0xFFFE) { "not PCM" }
                    channels = b.getShort(body + 2).toInt()
                    rate = b.getInt(body + 4)
                    bits = b.getShort(body + 14).toInt()
                }
                "data" -> {
                    require(bits == 16 && rate > 0) { "unsupported WAV format" }
                    // 合成中の WAV はサイズが 0 や不正なことがあるので、ファイルの終わりまでを使う
                    val end = if (size <= 0 || body + size > bytes.size) bytes.size else body + size
                    val frames = (end - body) / (2 * channels)
                    val out = FloatArray(frames) { f ->
                        var s = 0f
                        for (c in 0 until channels) s += b.getShort(body + (f * channels + c) * 2) / 32768f
                        s / channels
                    }
                    return out to rate
                }
            }
            pos = body + size + (size and 1)
            if (size < 0) break
        }
        throw IllegalArgumentException("no data chunk")
    }
}

/** 端末の音声認識で、1 回だけ聞き取る。main スレッドで使う。 */
class Listening(context: Context) : AutoCloseable {
    private val recognizer: SpeechRecognizer? =
        if (SpeechRecognizer.isRecognitionAvailable(context)) SpeechRecognizer.createSpeechRecognizer(context) else null

    val available get() = recognizer != null

    /** 聞き取った文（聞き取れなければ null）を [onResult] に渡す。 */
    fun listen(onReady: () -> Unit, onResult: (String?) -> Unit) {
        val r = recognizer ?: return onResult(null)
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) = onReady()
            override fun onBeginningOfSpeech() = Unit
            override fun onRmsChanged(rmsdB: Float) = Unit
            override fun onBufferReceived(buffer: ByteArray?) = Unit
            override fun onEndOfSpeech() = Unit
            override fun onError(error: Int) = onResult(null)
            override fun onResults(results: Bundle?) =
                onResult(results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull())
            override fun onPartialResults(partialResults: Bundle?) = Unit
            override fun onEvent(eventType: Int, params: Bundle?) = Unit
        })
        r.startListening(
            Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ja-JP")
                .putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1),
        )
    }

    fun cancel() = recognizer?.cancel()

    override fun close() {
        recognizer?.destroy()
    }
}
