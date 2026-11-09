package jp.devfest.shelfsight.assist

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import jp.devfest.shelfsight.model.Vec3
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sin

/**
 * 声と効果音を、測定した頭部インパルス応答（[Hrtf]）で聞く人の両耳へ畳み込んで鳴らす。
 *
 * 位置を持つ音は、鳴っている間も毎ブロック（約 12ms）方向を計算し直すので、スマホの向きを変えても
 * 音は店の中の同じ場所にとどまる。声は 1 つずつ順番に、効果音（人の接近）は重ねて鳴らす。
 */
class SpatialMixer(private val hrtf: Hrtf, private val listener: () -> Listener?) : AutoCloseable {
    /** 鳴らす音。[pcm] はミキサーのサンプリング周波数のモノラル。[position] が null なら頭の中（正面）で鳴る。 */
    class Sound(
        val pcm: FloatArray,
        val position: (() -> Vec3?)? = null,
        val gain: Float = 1f,
        val onDone: (() -> Unit)? = null,
    ) {
        internal var frame = 0
        internal var direction = -1
        internal var lastGain = 0f
        internal var doneCalled = false
        internal val length get() = pcm.size + Hrtf.TAPS
    }

    /** 繰り返し鳴る音（接近してくる人）。間隔と位置は外から更新する。 */
    class Emitter(@Volatile var position: Vec3, @Volatile var intervalSec: Float, @Volatile var gain: Float) {
        internal var nextFrame = 0L
    }

    private val track: AudioTrack = AudioTrack.Builder()
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
        )
        .setAudioFormat(
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(RATE)
                .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                .build(),
        )
        .setTransferMode(AudioTrack.MODE_STREAM)
        .setBufferSizeInBytes(BLOCK * 2 * 4 * 4)
        .build()

    private val speechQueue = java.util.concurrent.LinkedBlockingDeque<Sound>()
    @Volatile private var speech: Sound? = null
    private val effects = CopyOnWriteArrayList<Sound>()
    private val emitters = ConcurrentHashMap<Int, Emitter>()
    private val tick = makeTick()
    /** 記録モードで、実際に AudioTrack へ渡す音を保存する先。処理は呼び出し側で短く保つ。 */
    @Volatile var outputSink: ((FloatArray) -> Unit)? = null
    @Volatile private var running = true
    private var time = 0L

    private val thread = Thread({ loop() }, "SpatialMixer").apply {
        priority = Thread.MAX_PRIORITY
        start()
    }

    /** 声が鳴っている、または待っている。 */
    val speaking: Boolean get() = speech != null || speechQueue.isNotEmpty()

    /** 声を鳴らす。[interrupt] なら、鳴っている声と待っている声をやめてから鳴らす。 */
    fun speak(sound: Sound, interrupt: Boolean = false) {
        if (interrupt) stopSpeech()
        speechQueue.add(sound)
    }

    fun stopSpeech() {
        speechQueue.clear()
        speech?.let { it.frame = it.length }
    }

    fun play(sound: Sound) {
        effects += sound
    }

    /** 繰り返し鳴る音を入れ替える。[ids] にない音は止める。 */
    fun setEmitters(updates: Map<Int, Triple<Vec3, Float, Float>>) {
        emitters.keys.retainAll(updates.keys)
        for ((id, u) in updates) {
            val e = emitters[id]
            if (e == null) {
                emitters[id] = Emitter(u.first, u.second, u.third).also { it.nextFrame = time }
            } else {
                e.position = u.first; e.intervalSec = u.second; e.gain = u.third
            }
        }
    }

    private fun loop() {
        val left = FloatArray(BLOCK)
        val right = FloatArray(BLOCK)
        val out = FloatArray(BLOCK * 2)
        track.play()
        while (running) {
            left.fill(0f); right.fill(0f)
            val l = listener()

            // 繰り返しの音: 時刻が来たら 1 回ぶんを足す
            for (e in emitters.values) {
                if (time >= e.nextFrame) {
                    effects += Sound(tick, { e.position }, e.gain)
                    e.nextFrame = time + (e.intervalSec * RATE).toLong()
                }
            }

            val s = speech ?: speechQueue.poll()?.also { speech = it }
            if (s != null) {
                render(s, left, right, l, 1f)
                if (s.frame >= s.length) {
                    speech = null
                    finish(s)
                }
            }
            val duck = if (s != null) DUCK else 1f
            for (e in effects) {
                render(e, left, right, l, duck)
                if (e.frame >= e.length) { effects.remove(e); finish(e) }
            }
            for (k in 0 until BLOCK) {
                out[2 * k] = softClip(left[k])
                out[2 * k + 1] = softClip(right[k])
            }
            outputSink?.let { sink -> try { sink(out) } catch (e: RuntimeException) { Log.w(TAG, "audio recording failed", e) } }
            track.write(out, 0, out.size, AudioTrack.WRITE_BLOCKING)
            time += BLOCK
        }
    }

    private fun finish(s: Sound) {
        if (s.doneCalled) return
        s.doneCalled = true
        try { s.onDone?.invoke() } catch (e: RuntimeException) { Log.w(TAG, "onDone failed", e) }
    }

    private fun render(s: Sound, outL: FloatArray, outR: FloatArray, l: Listener?, extraGain: Float) {
        val pos = s.position?.invoke()
        if (pos == null || l == null) {   // 位置がない声は頭の中（正面）で鳴らす
            for (k in 0 until BLOCK) {
                if (s.frame >= s.length) return
                val x = s.pcm.getOrElse(s.frame) { 0f } * s.gain * extraGain
                outL[k] += x
                outR[k] += x
                s.frame++
            }
            return
        }
        val distance = Binaural.horizontalDistance(l, pos)
        val to = hrtf.nearest(Binaural.azimuth(l, pos), atan2(pos.y - l.position.y, distance))
        val gainTo = Binaural.distanceGain(distance) * s.gain * extraGain
        val from = if (s.direction < 0) to else s.direction
        val gainFrom = if (s.direction < 0) gainTo else s.lastGain
        s.direction = to
        s.lastGain = gainTo
        for (k in 0 until BLOCK) {
            if (s.frame >= s.length) return
            val t = k.toFloat() / BLOCK
            val g = lerp(gainFrom, gainTo, t)
            var left0 = hrtf.convolve(s.pcm, s.frame, to, 0)
            var right0 = hrtf.convolve(s.pcm, s.frame, to, 1)
            if (from != to) {   // 方向が変わったブロックは、前の方向から滑らかに移す
                left0 = lerp(hrtf.convolve(s.pcm, s.frame, from, 0), left0, t)
                right0 = lerp(hrtf.convolve(s.pcm, s.frame, from, 1), right0, t)
            }
            outL[k] += left0 * g
            outR[k] += right0 * g
            s.frame++
        }
    }

    override fun close() {
        running = false
        thread.join(500)
        track.stop()
        track.release()
    }

    companion object {
        private const val TAG = "SpatialMixer"
        const val RATE = 44100
        private const val BLOCK = 512
        /** 声が鳴っている間の効果音の大きさ。 */
        private const val DUCK = 0.5f

        private fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t

        private fun sample(pcm: FloatArray, at: Float): Float {
            if (at < 0f) return 0f
            val i = floor(at).toInt()
            if (i >= pcm.size) return 0f
            val f = at - i
            val a = pcm[i]
            val b = if (i + 1 < pcm.size) pcm[i + 1] else 0f
            return a + (b - a) * f
        }

        private fun softClip(x: Float): Float = x.coerceIn(-1f, 1f)

        /** 「ピコ」: 短く澄んだ音。方向が分かるよう、弱い倍音を足す。 */
        fun makeTick(): FloatArray {
            val len = (RATE * 0.09f).toInt()
            return FloatArray(len) { t ->
                val s = t.toFloat() / RATE
                val env = min(1f, s / 0.004f) * exp(-s / 0.025f) * min(1f, (len - t) / (RATE * 0.005f))
                0.45f * env * (sin(2 * PI * 1320 * s).toFloat() + 0.25f * sin(2 * PI * 2640 * s).toFloat())
            }
        }

        /** 「ポン」: 棚の方向を知らせる、丸く低めの音。 */
        fun makePon(): FloatArray {
            val len = (RATE * 0.16f).toInt()
            return FloatArray(len) { t ->
                val s = t.toFloat() / RATE
                val env = min(1f, s / 0.003f) * exp(-s / 0.05f) * min(1f, (len - t) / (RATE * 0.005f))
                0.6f * env * (sin(2 * PI * 620 * s).toFloat() + 0.35f * sin(2 * PI * 1240 * s).toFloat())
            }
        }

        /** [rate] のモノラルを、ミキサーの周波数へ線形補間で変換する。 */
        fun resample(pcm: FloatArray, rate: Int): FloatArray {
            if (rate == RATE) return pcm
            val n = (pcm.size.toLong() * RATE / rate).toInt()
            val step = rate.toFloat() / RATE
            return FloatArray(n) { sample(pcm, it * step) }
        }
    }
}
