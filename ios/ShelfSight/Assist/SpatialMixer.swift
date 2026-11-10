import AVFoundation

/**
 * 声と効果音を、測定した頭部インパルス応答（Hrtf）で聞く人の両耳へ畳み込んで鳴らす。
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
        var direction = -1
        var lastGain: Float = 0
        var doneCalled = false
        var length: Int { pcm.count + Hrtf.taps }

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
    /// 声が鳴っている間の効果音の大きさ。
    private static let duck: Float = 0.5

    private let hrtf: Hrtf?
    private let listener: () -> Listener?
    private let engine = AVAudioEngine()
    private var source: AVAudioSourceNode!
    private let lock = NSLock()
    private var speechQueue: [Sound] = []
    private var speech: Sound?
    private var effects: [Sound] = []
    private var emitters: [Int: Emitter] = [:]
    private let approachAlert = SpatialMixer.makeApproachAlert()
    private var time: Int64 = 0
    private var left = [Float](repeating: 0, count: SpatialMixer.block)
    private var right = [Float](repeating: 0, count: SpatialMixer.block)

    init(hrtf: Hrtf?, listener: @escaping () -> Listener?) {
        self.hrtf = hrtf
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
                effects.append(Sound(approachAlert, position: { e.position }, gain: e.gain))
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
        // 位置がない声と、方向が NaN になった音は頭の中（正面）で鳴らす
        guard let hrtf, let l, l.position.isFinite, l.forwardX.isFinite, l.forwardZ.isFinite, let pos = s.position?(), pos.isFinite else {
            for k in 0..<n {
                if s.frame >= s.length { return }
                let x = (s.frame < s.pcm.count ? s.pcm[s.frame] : 0) * s.gain * extraGain
                left[k] += x
                right[k] += x
                s.frame += 1
            }
            return
        }
        let horizontal = Binaural.horizontalDistance(l, pos)
        let height = pos.y - l.position.y
        let to = hrtf.nearest(azimuth: Binaural.azimuth(l, pos), elevation: atan2(height, horizontal))
        let gainTo = Binaural.distanceGain((horizontal * horizontal + height * height).squareRoot()) * s.gain * extraGain
        let from = s.direction < 0 ? to : s.direction
        let gainFrom = s.direction < 0 ? gainTo : s.lastGain
        s.direction = to
        s.lastGain = gainTo
        s.pcm.withUnsafeBufferPointer { pcm in
            for k in 0..<n {
                if s.frame >= s.length { return }
                let t = Float(k) / Float(n)
                let g = lerp(gainFrom, gainTo, t)
                var l0 = hrtf.convolve(pcm, frame: s.frame, direction: to, ear: 0)
                var r0 = hrtf.convolve(pcm, frame: s.frame, direction: to, ear: 1)
                if from != to {   // 方向が変わったブロックは、前の方向から滑らかに移す
                    l0 = lerp(hrtf.convolve(pcm, frame: s.frame, direction: from, ear: 0), l0, t)
                    r0 = lerp(hrtf.convolve(pcm, frame: s.frame, direction: from, ear: 1), r0, t)
                }
                left[k] += l0 * g
                right[k] += r0 * g
                s.frame += 1
            }
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

    /// 毎回同じ音にするための、種の決まった乱数（-1..1）。
    private struct Noise {
        var state: UInt32
        mutating func next() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state >> 8) / Float(1 << 23) - 1
        }
    }

    private static func tone(_ s: Float, _ hz: Float) -> Float { sin(2 * Float.pi * hz * s) }

    private static func normalized(_ out: [Float]) -> [Float] {
        let peak = max(out.map { abs($0) }.max() ?? 1, 1e-6)
        return out.map { $0 / peak * 0.6 }
    }

    /// 「ピコ」: 短く澄んだ音。方向が分かるよう、弱い倍音を足す。
    static func makeTick() -> [Float] {
        let r = Float(rate)
        let len = Int(r * 0.09)
        return (0..<len).map { t in
            let s = Float(t) / r
            let env = min(1, s / 0.004) * exp(-s / 0.025) * min(1, Float(len - t) / (r * 0.005))
            return 0.45 * env * (tone(s, 1320) + 0.25 * tone(s, 2640))
        }
    }

    /// 人が近づいてくるときの 175 ms の警告音。音程が上がる短い 3 連で、棚のチャイムと聞き分けられるようにする。
    static func makeApproachAlert() -> [Float] {
        let r = Float(rate)
        let pulse: Float = 0.045, gap: Float = 0.02
        var out = [Float](repeating: 0, count: Int(r * (3 * pulse + 2 * gap)))
        var noise = Noise(state: 11)
        var last: Float = 0
        for (i, base) in [Float(1400), 1800, 2300].enumerated() {
            let start = Int(Float(i) * (pulse + gap) * r)
            let len = min(Int(pulse * r), out.count - start)
            for t in 0..<len {
                let s = Float(t) / r
                let envelope = min(1, s / 0.001) * min(1, Float(len - t) / (r * 0.003))
                var saw: Float = 0
                for h in 1...9 { saw += tone(s, base * Float(h)) / Float(h) }
                let white = noise.next()
                let bright = white - last   // 差分で低域を落とす
                last = white
                out[start + t] = envelope * (saw + 0.6 * bright)
            }
        }
        return normalized(out)
    }

    /// 周波数比と強さ。金属的に響くよう整数倍から少しずらし、4 kHz より上にも重みを置く。
    private static let chimePartials: [(Float, Float)] = [(1, 0.45), (2.01, 0.3), (4.02, 0.3), (6.03, 0.3), (8.1, 0.28), (10.7, 0.25), (13.3, 0.2)]

    /// 棚の方向と立体音響のテストで鳴らす、360 ms の 2 連チャイム。方向の手がかりが出る高域まで、倍音とノイズで鳴らす。
    static func makeChime() -> [Float] {
        let r = Float(rate)
        var out = [Float](repeating: 0, count: Int(r * 0.36))
        var noise = Noise(state: 7)
        var last: Float = 0
        for (startS, base) in [(Float(0), Float(1047)), (Float(0.2), Float(1397))] {
            let start = Int(startS * r)
            let len = min(Int(0.16 * r), out.count - start)
            for t in 0..<len {
                let s = Float(t) / r
                let attack = min(1, s / 0.002)
                let tail = min(1, Float(len - t) / (r * 0.005))
                var sum: Float = 0
                for (ratio, amp) in chimePartials { sum += amp * tone(s, base * ratio) }
                let white = noise.next()
                let bright = white - last   // 差分で低域を落とす
                last = white
                out[start + t] = attack * tail * (exp(-s / 0.05) * sum + 0.5 * exp(-s / 0.03) * bright)
            }
        }
        return normalized(out)
    }

    /// fromRate のモノラルを、ミキサーの周波数へ線形補間で変換する。
    static func resample(_ pcm: [Float], _ fromRate: Int) -> [Float] {
        if fromRate == rate || fromRate <= 0 { return pcm }
        let n = Int(Int64(pcm.count) * Int64(rate) / Int64(fromRate))
        let step = Float(fromRate) / Float(rate)
        return (0..<n).map { sample(pcm, Float($0) * step) }
    }
}
