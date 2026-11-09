import AVFoundation
import Speech

/**
 * 端末の音声合成（AVSpeechSynthesizer）で話す。合成した音を SpatialMixer で鳴らすので、位置を付ければその方向から聞かせられる。
 * 同じ言葉は合成し直さない（短い言葉は、2 回目からすぐ鳴る）。
 */
final class Voice: NSObject, AVSpeechSynthesizerDelegate {
    private static let speechRate: Float = 1.15
    private static let cacheSize = 64
    private static let timeout: DispatchTimeInterval = .seconds(10)

    private let mixer: SpatialMixer
    private let synthesizer = AVSpeechSynthesizer()
    private let voice = AVSpeechSynthesisVoice(language: "ja-JP")
    private let worker = DispatchQueue(label: "voice", qos: .userInitiated, autoreleaseFrequency: .workItem)
    @Locked private var generation = 0
    @Locked private var finishWaiting: (() -> Void)? = nil
    /// worker のキューだけで使う。
    private var cache: [String: [Float]] = [:]
    private var cacheOrder: [String] = []

    @Locked private(set) var lastSpoken = ""

    init(mixer: SpatialMixer) {
        self.mixer = mixer
        super.init()
        synthesizer.delegate = self
    }

    /**
     * 話す。position があれば、その位置から聞こえる。
     * interrupt なら、話している途中の声と、まだ話していない声をやめる（長押しの「調べています」など、すぐ伝えたいとき）。
     */
    func say(_ text: String, position: (() -> Vec3?)? = nil, interrupt: Bool = false, remember: Bool = true, onDone: (() -> Void)? = nil) {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return }
        let spoken = Reading.forSpeech(text)
        let gen: Int
        if interrupt {
            generation += 1
            gen = generation
            mixer.stopSpeech()
        } else {
            gen = generation
        }
        if remember { lastSpoken = text }
        worker.async { [self] in
            guard gen == generation, let pcm = synth(spoken), gen == generation else { return }
            mixer.speak(SpatialMixer.Sound(pcm, position: position, onDone: onDone))
        }
    }

    /// よく使う短い言葉を先に合成しておき、言うときにすぐ鳴るようにする。
    func prepare(_ texts: [String]) {
        worker.async { [self] in for t in texts { _ = synth(Reading.forSpeech(t)) } }
    }

    /// 話している声と、まだ話していない声をやめる。
    func stop() {
        generation += 1
        mixer.stopSpeech()
    }

    var speaking: Bool { mixer.speaking }

    /// 合成した音（ミキサーの周波数のモノラル）。worker のキューで呼ぶ。
    private func synth(_ text: String) -> [Float]? {
        if let c = cache[text] { return c }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        utterance.rate = min(AVSpeechUtteranceMaximumSpeechRate, AVSpeechUtteranceDefaultSpeechRate * Self.speechRate)
        let lock = NSLock()
        var samples: [Float] = []
        var rate = 0
        var finished = false
        let done = DispatchSemaphore(value: 0)
        let finish = {
            lock.lock()
            let first = !finished
            finished = true
            lock.unlock()
            if first { done.signal() }
        }
        finishWaiting = finish
        synthesizer.write(utterance) { buffer in
            guard let pcm = buffer as? AVAudioPCMBuffer else { return }
            let frames = Int(pcm.frameLength)
            if frames == 0 { finish(); return }
            let channels = Int(pcm.format.channelCount)
            var mono = [Float](repeating: 0, count: frames)
            if let f = pcm.floatChannelData {
                for c in 0..<channels { for i in 0..<frames { mono[i] += f[c][i] / Float(channels) } }
            } else if let s = pcm.int16ChannelData {
                for c in 0..<channels { for i in 0..<frames { mono[i] += Float(s[c][i]) / 32768 / Float(channels) } }
            } else if let s = pcm.int32ChannelData {
                for c in 0..<channels { for i in 0..<frames { mono[i] += Float(s[c][i]) / 2_147_483_648 / Float(channels) } }
            }
            lock.lock()
            rate = Int(pcm.format.sampleRate)
            samples += mono
            lock.unlock()
        }
        let waited = done.wait(timeout: .now() + Self.timeout)
        finishWaiting = nil
        if waited == .timedOut {
            synthesizer.stopSpeaking(at: .immediate)
            return nil
        }
        lock.lock()
        let pcm = SpatialMixer.resample(samples, rate)
        lock.unlock()
        if pcm.isEmpty { return nil }
        cache[text] = pcm
        cacheOrder.append(text)
        if cacheOrder.count > Self.cacheSize { cache[cacheOrder.removeFirst()] = nil }
        return pcm
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        finishWaiting?()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        finishWaiting?()
    }
}

/**
 * 端末の音声認識で、1 回だけ聞き取る。メインスレッドで使う。
 * 話し終わって少し黙ったら（SILENCE 秒）、そこまでに聞き取った文を返す。
 */
final class Listening {
    private static let silence: TimeInterval = 1.2
    private static let firstWordTimeout: TimeInterval = 6
    private static let maxDuration: TimeInterval = 12

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
    private let engine = AVAudioEngine()
    private var task: SFSpeechRecognitionTask?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var timer: Timer?
    private var deadline: Timer?
    private var onResult: ((String?) -> Void)?
    private var best: String?

    var available: Bool {
        recognizer?.isAvailable == true && SFSpeechRecognizer.authorizationStatus() == .authorized
    }

    /// マイクと音声認識の許可を求める（初回だけダイアログが出る）。
    static func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { _ in }
        AVAudioSession.sharedInstance().requestRecordPermission { _ in }
    }

    /// 聞き取った文（聞き取れなければ nil）を onResult に渡す。
    func listen(onReady: @escaping () -> Void, onResult: @escaping (String?) -> Void) {
        cancel()
        guard let recognizer, available else { onResult(nil); return }
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition { req.requiresOnDeviceRecognition = true }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else { onResult(nil); return }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in req.append(buffer) }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            onResult(nil)
            return
        }
        request = req
        self.onResult = onResult
        best = nil
        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            DispatchQueue.main.async {
                guard let self else { return }
                if let result {
                    self.best = result.bestTranscription.formattedString
                    if result.isFinal { self.finish() } else { self.waitSilence(Self.silence) }
                } else if error != nil {
                    self.finish()
                }
            }
        }
        waitSilence(Self.firstWordTimeout)
        deadline = Timer.scheduledTimer(withTimeInterval: Self.maxDuration, repeats: false) { [weak self] _ in self?.finish() }
        onReady()
    }

    private func waitSilence(_ seconds: TimeInterval) {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in self?.finish() }
    }

    private func finish() {
        guard let cb = onResult else { return }
        onResult = nil
        let text = best?.trimmingCharacters(in: .whitespacesAndNewlines)
        stopAudio()
        cb((text?.isEmpty ?? true) ? nil : text)
    }

    func cancel() {
        onResult = nil
        stopAudio()
    }

    private func stopAudio() {
        timer?.invalidate()
        timer = nil
        deadline?.invalidate()
        deadline = nil
        if engine.isRunning {
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
        }
        request?.endAudio()
        request = nil
        task?.cancel()
        task = nil
    }
}
