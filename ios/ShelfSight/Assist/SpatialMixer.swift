import AVFoundation

/**
 * 声と効果音を、聞く人の向きに合わせて左右の耳へ振り分けて鳴らす。
 *
 * 位置を持つ音は、鳴っている間も 512 サンプルごと（約 12ms）に方向を計算し直すので、スマホの向きを変えても
 * 音は店の中の同じ場所にとどまる。声は 1 つずつ順番に、効果音（人の接近）は重ねて鳴らす。
 * AVAudioEngine の AVAudioSourceNode に毎回書き込む（Android 版の AudioTrack に書き続けるスレッドにあたる）。
 */
final class SpatialMixer {
    /// 鳴らす音。pcm はミキサーのサンプリング周波数のモノラル。position が nil なら頭の中（正面）で鳴る。
    final class Sound {
        let pcm: [Float]
        let position: (() -> Vec3?)?
        let gain: Float
        let onDone: (() -> Void)?
        var frame = 0
        var lpL: Float = 0
        var lpR: Float = 0
        var last: EarParams?
        var doneCalled = false
        var length: Int { pcm.count + SpatialMixer.maxDelay }

        init(_ pcm: [Float], position: (() -> Vec3?)? = nil, gain: Float = 1, onDone: (() -> Void)? = nil) {
            self.pcm = pcm
            self.position = position
            self.gain = gain
            self.onDone = onDone
        }
    }

    /// 繰り返し鳴る音（接近してくる人）。間隔と位置は外から更新する。
    final class Emitter {
        var position: Vec3
        var intervalSec: Float
        var gain: Float
        var nextFrame: Int64 = 0

        init(position: Vec3, intervalSec: Float, gain: Float) {
            self.position = position
            self.intervalSec = intervalSec
            self.gain = gain
        }
    }

    static let rate = 44100
    private static let block = 512
    static let maxDelay = 64
    /// 声が鳴っている間の効果音の大きさ。
    private static let duck: Float = 0.5
    private static let center = EarParams(gainL: 1, gainR: 1, delayL: 0, delayR: 0, lowpassL: 1, lowpassR: 1)

    private let listener: () -> Listener?
    private let engine = AVAudioEngine()
    private var source: AVAudioSourceNode!
    private let lock = NSLock()
    private var speechQueue: [Sound] = []
    private var speech: Sound?
    private var effects: [Sound] = []
    private var emitters: [Int: Emitter] = [:]
    private let tick = SpatialMixer.makeTick()
    private var time: Int64 = 0
    private var left = [Float](repeating: 0, count: SpatialMixer.block)
    private var right = [Float](repeating: 0, count: SpatialMixer.block)

    init(listener: @escaping () -> Listener?) {
        self.listener = listener
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(SpatialMixer.rate), channels: 2)!
        source = AVAudioSourceNode(format: format) { [unowned self] _, _, frameCount, bufferList -> OSStatus in
            let buffers = UnsafeMutableAudioBufferListPointer(bufferList)
            let n = Int(frameCount)
            guard buffers.count >= 2,
                  let l = buffers[0].mData?.assumingMemoryBound(to: Float.self),
                  let r = buffers[1].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            self.fill(l, r, n)
            return noErr
        }
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: format)
        // イヤホンの抜き差しや、音声認識でマイクを使い始めたときに止まるので、動かし直す
        NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            self?.start()
        }
        start()
    }

    func start() {
        if engine.isRunning { return }
        do {
            try engine.start()
        } catch {
            logInfo("SpatialMixer", "could not start: \(error)")
        }
    }

    /// 声が鳴っている、または待っている。
    var speaking: Bool {
        lock.lock(); defer { lock.unlock() }
        return speech != nil || !speechQueue.isEmpty
    }

    /// 声を鳴らす。interrupt なら、鳴っている声と待っている声をやめてから鳴らす。
    func speak(_ sound: Sound, interrupt: Bool = false) {
        if interrupt { stopSpeech() }
        lock.lock(); speechQueue.append(sound); lock.unlock()
    }

    func stopSpeech() {
        lock.lock()
        speechQueue.removeAll()
        if let s = speech { s.frame = s.length }
        lock.unlock()
    }

    func play(_ sound: Sound) {
        lock.lock(); effects.append(sound); lock.unlock()
    }

    /// 繰り返し鳴る音を入れ替える（id → 位置、間隔、大きさ）。updates にない音は止める。
    func setEmitters(_ updates: [Int: (Vec3, Float, Float)]) {
        lock.lock(); defer { lock.unlock() }
        // NaN や無限大はオーディオのスレッドで整数にするときに落ちるので、ここで捨てる
        let updates = updates.filter { $0.value.0.isFinite && $0.value.1.isFinite && $0.value.1 > 0 && $0.value.2.isFinite }
        emitters = emitters.filter { updates[$0.key] != nil }
        for (id, u) in updates {
            if let e = emitters[id] {
                e.position = u.0
                e.intervalSec = u.1
                e.gain = u.2
            } else {
                let e = Emitter(position: u.0, intervalSec: u.1, gain: u.2)
                e.nextFrame = time
                emitters[id] = e
            }
        }
    }

    // ------------------------------------------------------------ 音を作る（オーディオのスレッド）

    private func fill(_ outL: UnsafeMutablePointer<Float>, _ outR: UnsafeMutablePointer<Float>, _ n: Int) {
        let l = listener()
        var finished: [Sound] = []
        lock.lock()
        var done = 0
        while done < n {
            let m = min(Self.block, n - done)
            for k in 0..<m { left[k] = 0; right[k] = 0 }

            // 繰り返しの音: 時刻が来たら 1 回ぶんを足す
            for e in emitters.values where time >= e.nextFrame {
                effects.append(Sound(tick, position: { e.position }, gain: e.gain))
                e.nextFrame = time + Self.frames(e.intervalSec)
            }

            if speech == nil, !speechQueue.isEmpty { speech = speechQueue.removeFirst() }
            let s = speech
            if let s {
                render(s, m, l, 1)
                if s.frame >= s.length {
                    speech = nil
                    finished.append(s)
                }
            }
            let duck: Float = s != nil ? Self.duck : 1
            var still: [Sound] = []
            for e in effects {
                render(e, m, l, duck)
                if e.frame >= e.length { finished.append(e) } else { still.append(e) }
            }
            effects = still
            for k in 0..<m {
                outL[done + k] = left[k].clamped(-1, 1)
                outR[done + k] = right[k].clamped(-1, 1)
            }
            time += Int64(m)
            done += m
        }
        lock.unlock()
        for s in finished where !s.doneCalled {
            s.doneCalled = true
            if let cb = s.onDone { DispatchQueue.global().async { cb() } }
        }
    }

    private func render(_ s: Sound, _ n: Int, _ l: Listener?, _ extraGain: Float) {
        var target = Self.center
        if let pos = s.position?(), pos.isFinite, let l, l.position.isFinite, l.forwardX.isFinite, l.forwardZ.isFinite {
            let p = Binaural.params(Binaural.azimuth(l, pos), Binaural.horizontalDistance(l, pos), Self.rate)
            if p.isFinite { target = p }
        }
        let from = s.last ?? target
        s.last = target
        let g = s.gain * extraGain
        let rate = Float(Self.rate)
        for k in 0..<n {
            if s.frame >= s.length { return }
            let t = Float(k) / Float(n)
            let dl = lerp(from.delayL, target.delayL, t) * rate
            let dr = lerp(from.delayR, target.delayR, t) * rate
            let xl = Self.sample(s.pcm, Float(s.frame) - dl)
            let xr = Self.sample(s.pcm, Float(s.frame) - dr)
            s.lpL += lerp(from.lowpassL, target.lowpassL, t) * (xl - s.lpL)
            s.lpR += lerp(from.lowpassR, target.lowpassR, t) * (xr - s.lpR)
            left[k] += s.lpL * lerp(from.gainL, target.gainL, t) * g
            right[k] += s.lpR * lerp(from.gainR, target.gainR, t) * g
            s.frame += 1
        }
    }

    private func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }

    /// 秒 → サンプル数。NaN や無限大を Int64 にすると Swift が止めるので、0.05〜10 秒に収める。
    private static func frames(_ seconds: Float) -> Int64 {
        let s = seconds.isFinite ? min(max(seconds, 0.05), 10) : 1
        return Int64(s * Float(rate))
    }

    private static func sample(_ pcm: [Float], _ at: Float) -> Float {
        // NaN を Int にすると Swift が止める（NaN は at < 0 にも当たらない）
        if !at.isFinite || at < 0 { return 0 }
        let i = Int(at)
        if i >= pcm.count { return 0 }
        let f = at - Float(i)
        let a = pcm[i]
        let b = i + 1 < pcm.count ? pcm[i + 1] : 0
        return a + (b - a) * f
    }

    /// 「ピコ」: 短く澄んだ音。方向が分かるよう、弱い倍音を足す。
    static func makeTick() -> [Float] {
        let r = Float(rate)
        let len = Int(r * 0.09)
        return (0..<len).map { t in
            let s = Float(t) / r
            let env = min(1, s / 0.004) * exp(-s / 0.025) * min(1, Float(len - t) / (r * 0.005))
            return 0.45 * env * (sin(2 * Float.pi * 1320 * s) + 0.25 * sin(2 * Float.pi * 2640 * s))
        }
    }

    /// rate のモノラルを、ミキサーの周波数へ線形補間で変換する。
    static func resample(_ pcm: [Float], _ fromRate: Int) -> [Float] {
        if fromRate == rate || fromRate <= 0 { return pcm }
        let n = Int(Int64(pcm.count) * Int64(rate) / Int64(fromRate))
        let step = Float(fromRate) / Float(rate)
        return (0..<n).map { sample(pcm, Float($0) * step) }
    }
}
