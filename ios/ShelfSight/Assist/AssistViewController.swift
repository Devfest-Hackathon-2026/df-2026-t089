import ARKit
import AVFoundation
import SceneKit
import UIKit

/**
 * 視覚障害のある人のための、コンビニの棚と商品の案内。胸の前にスマホを立てて持ち、イヤホンで聞く（Android 版の AssistActivity）。
 *
 * ② 棚が 1 つ写っている: 商品を認識し、LLM で何があるかを短く自然な文にして話す（場違いな商品は言わない）
 * ③ スマホのすぐ前に商品がある（画面の中央の深度だけで決める、NearProduct）: 長押しで Cloud Vision API で文字を読み（1 回の接近で 1 回だけ呼ぶ）、
 *    その文字からカテゴリを決めて（端末の辞書で決めきれなければ Gemini、CategoryJudge）、商品名、カテゴリの順に言う
 * ④ 人が近づいてくる: その人の方向からピコピコ鳴らす（productOnly の間も動く）
 *
 * 棚の種類の判別（歩行中の 8 列 × 7 分類）と 3D の地図は入れていない（Android 版も止めている）。
 *
 * 操作
 * - 長押し: 「調べています」と言い、手前の商品 1 つなら ③、棚の広い範囲なら ②、その間なら「一歩下がって」と頼む
 * - 2 回たたく: 質問する（決まった言い方はその操作、それ以外は LLM が画像を見て答える）
 * VoiceOver が有効なときは、ダブルタップが質問、ダブルタップ長押しが長押し（「調べる」のアクションでもよい）。
 */
final class Assist: NSObject, ARSessionDelegate {
    // ------------------------------------------------------------ 定数（Android 版と同じ）
    /**
     * 商品の解析を、前の解析が終わったらすぐ次を回し、その時間を計る（Android 版の PRODUCT_ONLY と同じ）。
     * 記録・面の検出（地図）は止める。人の検出（④）・長押し・質問・棚の説明は動く。
     */
    static let productOnly = true
    /// 商品の解析の間隔（productOnly でないとき）。
    private static let sceneIntervalMs: Int64 = 2000
    /// これより速く動いている（歩く・振り向く）間は、商品の解析と棚の判別を休む。画像がぶれて役に立たない。
    private static let fastMps: Float = 1.0
    private static let fastDps: Float = 60
    private static let motionSmoothing: Float = 0.2
    private static let personIntervalMs: Int64 = 200
    private static let sceneLongSide = 1280
    private static let personLongSide = 640
    private static let lookupLongSide = 1600
    private static let maxCameraImageWidth: CGFloat = 1920
    private static let narrateLongSide = 768
    private static let questionLongSide = 1280
    private static let forgetDescriptionMs: Int64 = 3000
    private static let moveToRedescribeM: Float = 1.5
    /// 棚の説明を試してから、次に試すまでの間（説明できなかったときも含む）。
    private static let describeRetryMs: Int64 = 1500
    /// 残りのメモリがこれより少ないときは、商品の解析を休む（OS に止められないための安全装置）。
    private static let minAvailableMB: Double = 400
    /// 1 つだけ特に近い商品がこれだけ見えなければ、次に見えたときを新しい接近とする。
    private static let nearGapMs: Int64 = 2000
    private static let personHoldS = 1.0
    private static let approachAlertM: Float = 6
    private static let closeAlertM: Float = 1.5

    // ------------------------------------------------------------ AR と画面の表示
    private let session: ARSession
    let debug = AssistDebug()
    /// 画面の大きさ（画面の表示用。メインスレッドで書く）。
    @Locked var viewSize = CGSize.zero
    private var greeted = false

    // ------------------------------------------------------------ 音
    @Locked private var listener: Listener? = nil
    private lazy var mixer = SpatialMixer { [weak self] in self?.listener }
    private lazy var voice = Voice(mixer: mixer)
    private let listening = Listening()

    // ------------------------------------------------------------ キュー（Android 版のスレッド）
    // どのキューも、1 回の処理が終わるたびに一時的なオブジェクト（autorelease されたもの）を捨てる（.workItem）。
    // 既定のままだと、休まず回り続けるキューでは捨てる機会がなく、カメラの画像や ONNX Runtime の出力がたまって、
    // 20 秒ほどでメモリが足りなくなり OS に終了させられた
    /// ARFrame を受け取るキュー（Android 版の GL スレッド）。
    private let arQueue = DispatchQueue(label: "ar", qos: .userInteractive, autoreleaseFrequency: .workItem)
    /// 商品の解析。商品だけを計るときは譲る相手がいないので、優先度を下げない。
    private let sceneWorker = DispatchQueue(label: "scene", qos: Assist.productOnly ? .userInitiated : .utility, autoreleaseFrequency: .workItem)
    private let personWorker = DispatchQueue(label: "person", qos: .userInitiated, autoreleaseFrequency: .workItem)
    private let lookupWorker = DispatchQueue(label: "lookup", qos: .userInitiated, autoreleaseFrequency: .workItem)
    private let sceneBusy = AtomicFlag()
    private let personBusy = AtomicFlag()
    /// 商品の解析にかかった時間（productOnly の計測用）。
    private let timing = SceneTiming()

    // ------------------------------------------------------------ 認識
    /// 商品認識。CPU 側のメモリが増えたら、解析を止めずに裏で作り直す（RecognizerPool）。
    private let recognizer = RecognizerPool()
    private let personDetector: PersonDetector = VisionPersonDetector()
    private let tracker = PersonTracker()
    private let lookup = ProductLookup(
        apiKey: Secrets.value("CLOUD_VISION_API_KEY", fallback: "GCP_API_KEY"),
        judge: CategoryJudge(rules: .fromBundle(), client: Assist.geminiClient())
    )
    private let narrator = Narrator(client: Assist.geminiClient())
    private let judge = SceneJudge()
    private let recorder = Once<Recorder> { Recorder() }
    /// 商品のジャンル（26 種類）→ 7 つのまとまり。場違いな商品を外すのに使う（categories.json）。
    private let groups = ShelfDescriber.Groups.fromBundle()

    // ------------------------------------------------------------ 状態
    @Locked private var pendingCapture: ((CapturedFrame) -> Void)? = nil
    @Locked private var speedMps: Float = 0
    @Locked private var turnDps: Float = 0
    @Locked private var skipped: [(String, Int)] = []
    @Locked private var auto = true
    /// ② を話した場所と、その棚の種類（細分類）。ここから離れるか、別の種類の棚に向くか、棚 1 つの場面を離れるまでは繰り返さない。
    @Locked private var describedAt: Vec3? = nil
    @Locked private var describedKinds: Set<String>? = nil
    /// 自動の棚の説明を作っている間（LLM を待っている間）は、次を頼まない。
    @Locked private var describing = false
    /// 質問に答えるときに LLM に渡す、最後に話した説明と、最後に調べた商品。
    @Locked private var lastDescription = ""
    @Locked private var lastProduct = ""
    // AR のキューだけで使う
    private var lastPose: [Float]?
    private var lastPoseAt: Int64 = 0
    private var lastScene: Int64 = 0
    private var lastPerson: Int64 = 0
    // sceneWorker だけで使う
    private var leftOneShelfAt: Int64 = 0
    private var lastDescribeTry: Int64 = 0
    /// 説明した棚から別の棚に移ったか（1.5m 動かなくても、隣の棚に向きを変えたら説明し直す）。
    private var shelfChange = ShelfChange()
    /// 1 つだけ特に近い商品（NearProduct）を最後に見た時刻。しばらく見えなければ、次に見えたときを新しい接近とする。
    private var nearSeenAt: Int64 = 0

    // ------------------------------------------------------------ 始める・止める

    init(session: ARSession) {
        self.session = session
        super.init()
        // 声は A2DP（ステレオ）のまま鳴らしたいので、HFP は使わない（マイクは本体のもの）
        let audio = AVAudioSession.sharedInstance()
        try? audio.setCategory(.playAndRecord, mode: .default, options: [.allowBluetoothA2DP, .defaultToSpeaker])
        try? audio.setActive(true)
        _ = mixer
        _ = voice
        session.delegate = self
        session.delegateQueue = arQueue
        updateSettings()
        Listening.requestPermissions()
    }

    func resume() {
        guard ARWorldTrackingConfiguration.isSupported else {
            voice.say("このスマホでは、カメラの位置合わせが使えません。")
            debug.tracking = "ARKit が使えません"
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.isAutoFocusEnabled = true
        // 商品だけを計るときは、地図と棚のラベルにしか使わない面の検出を止める
        if !Self.productOnly { config.planeDetection = [.vertical] }
        // 深度は、すぐ前の商品の判定（NearProduct）に使うので、商品だけを計るときも使う（LiDAR のある機種だけ）
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        } else {
            debug.near = "深度が取れない機種（LiDAR なし）なので、近い商品を判定できません"
        }
        // 商品の文字を読めるよう、30fps 以上で幅 1920 までのいちばん大きいカメラ画像を使う
        if let best = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter({ $0.imageResolution.width <= Self.maxCameraImageWidth && $0.framesPerSecond >= 30 })
            .max(by: { $0.imageResolution.width * $0.imageResolution.height < $1.imageResolution.width * $1.imageResolution.height }) {
            config.videoFormat = best
        }
        session.run(config)
        sceneWorker.async { [recognizer] in _ = recognizer.get }   // 優先度を下げたキューで読み込む
        if !greeted {
            greeted = true
            voice.say("準備ができました。長押しで、商品や棚を調べます。2回たたくと、質問できます。")
            voice.prepare(["調べています"])
        }
    }

    func pause() {
        listening.cancel()
        session.pause()
    }

    // ------------------------------------------------------------ 毎フレーム（AR のキュー）

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // 毎フレーム（60 回/秒）の一時的なオブジェクトを、このフレームのうちに捨てる
        autoreleasepool { onFrame(frame) }
    }

    private func onFrame(_ frame: ARFrame) {
        let size = viewSize
        if size.width > 0 {
            // 画面の表示用: センサー画像の正規化座標が、画面のどこに来るか
            let t = frame.displayTransform(for: .portrait, viewportSize: size)
            let p = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 0, y: 1)].map { $0.applying(t) }
            debug.corners = p.flatMap { [Float($0.x), Float($0.y)] }
        }
        let camera = frame.camera
        switch camera.trackingState {
        case .normal: debug.tracking = "追跡中"
        case .limited(let reason): debug.tracking = "一時停止（\(reason)）"
        case .notAvailable: debug.tracking = "停止"
        }
        guard case .normal = camera.trackingState else { return }

        // 聞く人の向き: スマホを胸の前に立てて持つので、カメラの前（-z）を水平にしたもの
        let m = camera.transform
        let z = m.columns.2
        let fx = -z.x
        let fz = -z.z
        let len = (fx * fx + fz * fz).squareRoot()
        if len > 0.3 {
            listener = Listener(position: Vec3(m.columns.3.x, m.columns.3.y, m.columns.3.z), forwardX: fx / len, forwardZ: fz / len)
            debug.headingDeg = atan2(fx, -fz) * 180 / .pi
        }

        let now = nowMs()
        updateMotion(m, now)
        if let target = pendingCapture, let f = FrameCapture.capture(frame, depthEnabled: true, fullResolution: true) {
            pendingCapture = nil
            lookupWorker.async { target(f) }
        }
        // ④ 人の検出は、商品だけを計るときも回す（地図・面の検出・記録は止めたまま）
        if now - lastPerson >= Self.personIntervalMs && personBusy.tryAcquire() {
            if let f = FrameCapture.capture(frame, depthEnabled: true, usualLongSide: Self.personLongSide) {
                lastPerson = now
                personWorker.async { [self] in runPeople(f); personBusy.release() }
            } else {
                personBusy.release()
            }
        }
        if Self.productOnly {
            // 商品の解析を、前の解析が終わったらすぐ次を回す（記録と面の検出は止める）。
            // 残りのメモリが少ないときは休む（増え続ける原因がほかにあっても、OS に止められないように）
            let available = Memory.availableMB()
            if available < Self.minAvailableMB {
                if sceneBusy.tryAcquire() {
                    debug.sceneMs = String(format: "メモリが少ないので解析を休み中（残り %.0fMB）", available)
                    sceneBusy.release()
                }
            } else if sceneBusy.tryAcquire() {
                let t0 = nowMs()
                if let f = FrameCapture.capture(frame, depthEnabled: true, usualLongSide: Self.sceneLongSide) {
                    let captureMs = nowMs() - t0
                    let heading = debug.headingDeg
                    sceneWorker.async { [self] in runScene(f, 0, heading, captureMs: captureMs); sceneBusy.release() }
                } else {
                    sceneBusy.release()
                }
            }
            return
        }
        let fast = speedMps > Self.fastMps || turnDps > Self.fastDps
        if now - lastScene >= Self.sceneIntervalMs && fast {
            // 速く動いている間は画像がぶれて役に立たない
            lastScene = now
            skip("動きが速い")
        }
        if now - lastScene >= Self.sceneIntervalMs && sceneBusy.tryAcquire() {
            if let f = FrameCapture.capture(frame, depthEnabled: true, usualLongSide: Self.sceneLongSide) {
                lastScene = now
                let w = wallCount(frame)
                let heading = debug.headingDeg
                sceneWorker.async { [self] in runScene(f, w, heading); sceneBusy.release() }
            } else {
                sceneBusy.release()
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        logInfo("Assist", "AR session failed: \(error)")
        debug.tracking = "ARKit のエラー: \(error.localizedDescription)"
    }

    // ------------------------------------------------------------ ①② 写っているものの判定

    /// 商品の枠を出して写っているものを判定し、商品に名前を付ける。productOnly でなければ画像と姿勢と結果を記録する。
    private func runScene(_ f: CapturedFrame, _ walls: Int, _ heading: Float?, captureMs: Int64 = 0) {
        guard let recognizer = recognizer.get else { debug.sceneMs = "商品認識のモデルを読み込めません"; return }
        let t0 = nowMs()
        let image = ImageUtil.scaled(f.image, longSide: Self.sceneLongSide)
        let convertMs = nowMs() - t0
        do {
            let now = nowMs()
            let boxes = try recognizer.detect(image).map(\.0)
            let detectMs = nowMs() - now
            // スマホのすぐ前の商品（枠は使わず、画面の中央の深度だけで決める）
            let reading = NearProduct.measure(f.geometry)
            let near = reading.pick
            // 枠ごとの距離は、画面に出すためだけに測る
            let distances = boxDistances(boxes, f.geometry)
            judge.update(SceneEvidence(shelves: 0, productBoxes: boxes, near: near != nil))
            // 新しく近い商品が見えたら、前に調べた結果を忘れる（次の長押しで Cloud Vision を 1 回呼ぶ）
            if near != nil {
                if nearSeenAt == 0 || now - nearSeenAt > Self.nearGapMs { lookup.newApproach() }
                nearSeenAt = now
            }
            debug.boxDistances = distances
            debug.nearBox = near?.box
            debug.nearDistance = near?.distanceM
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {   // LiDAR がなければ、深度が取れないという表示のまま
                debug.near = reading.describe()
            }
            let products = try recognizer.recognize(image, boxes)
            // ONNX Runtime が中に取っておいたメモリが増えていれば、裏で作り直して返させる
            self.recognizer.reloadIfNeeded()
            let recognizeMs = nowMs() - now - detectMs
            if Self.productOnly {
                timing.add(captureMs: captureMs, convertMs: convertMs, detectMs: detectMs, nameMs: recognizeMs, boxes: boxes.count, named: products.count)
                logInfo("Assist", "product only: \(timing.lastLine())")
            } else {
                recorder.get.record(
                    Int64(Date().timeIntervalSince1970 * 1000), image, f.geometry, headingDeg: heading, walls: walls,
                    products: products.map { Recorder.Product(p: $0, position: nil) }
                )
            }
            debug.productBoxes = boxes
            debug.products = products
            debug.scene = judge.scene
            debug.sceneMs = Self.productOnly ? "【商品だけ計測】\n\(timing.summary())" : "枠 \(detectMs)ms / 名前 \(recognizeMs)ms / 壁 \(walls)"
            debug.sceneAt = now
            debug.motion = String(format: "%.1fm/s %.0f°/s", speedMps, turnDps)
            let known = distances.compactMap { $0 }
            let nearLog = "near " + reading.describe() +
                ", box depth \(known.count)/\(boxes.count) [" + known.sorted().prefix(5).map { String(format: "%.2f", $0) }.joined(separator: ",") + "]"
            logInfo("Assist", "scene: detect \(detectMs)ms, name \(recognizeMs)ms, boxes \(boxes.count), named \(products.count), \(nearLog)")
            let camera = cameraPosition(f)
            if judge.scene != .oneShelf && leftOneShelfAt == 0 { leftOneShelfAt = now }
            if judge.scene == .oneShelf { leftOneShelfAt = 0 }
            if leftOneShelfAt != 0 && now - leftOneShelfAt > Self.forgetDescriptionMs { forgetDescription() }

            // 説明し直すのは、説明した場所から moveToRedescribeM 離れたとき（まだ説明していなければすぐ）か、別の種類の棚に移ったとき
            let kinds = judge.scene == .oneShelf ? Set(ShelfDescriber.split(products, groups).mainKinds) : []
            let otherShelf = shelfChange.update(kinds, described: describedKinds, now: now)
            let moved = (describedAt.map { $0.distanceTo(camera) } ?? .greatestFiniteMagnitude) > Self.moveToRedescribeM
            debug.shelfChange = shelfChange.describe(now) + (describedAt.map { String(format: "、説明した場所から %.1fm", $0.distanceTo(camera)) } ?? "")
            if auto && !voice.speaking && !describing && judge.scene == .oneShelf && now - lastDescribeTry > Self.describeRetryMs && (moved || otherShelf) {
                lastDescribeTry = now
                describing = true
                shelfChange.reset()
                logInfo("Assist", "describe trigger: \(moved ? "moved" : "other shelf"), \(debug.shelfChange)")
                // LLM を待つ間も記録を止めないよう、別のキューで話す
                let jpg = ImageUtil.jpeg(image, longSide: Self.narrateLongSide)
                lookupWorker.async { [self] in
                    defer { describing = false }
                    describe(products, jpg, boxes.count, camera, requested: false)
                }
            }
        } catch {
            logInfo("Assist", "scene analysis failed: \(error)")
        }
    }

    /// 前に説明した棚を忘れる（次に棚 1 つになったら、場所や種類に関係なく説明する）。
    private func forgetDescription() {
        describedAt = nil
        describedKinds = nil
    }

    /**
     * ② 棚 1 つの商品を説明する。products はその場で認識したもの。
     * LLM で短い自然な文にする（使えなければ決まった言い方の文）。LLM を待つので、呼んだキューはしばらく止まる。
     */
    private func describe(_ products: [RecognizedProduct], _ jpg: Data, _ boxCount: Int, _ camera: Vec3?, requested: Bool) {
        let t1 = nowMs()
        let split = ShelfDescriber.split(products, groups)
        debug.kept = split.kept
        debug.removed = split.removed
        debug.unnamed = max(boxCount - products.count, 0)
        debug.mainGroup = "\(split.mainGroup ?? "null")（\(split.mainKinds.joined(separator: "・"))）"
        debug.productsAt = t1
        let llm = narrator.available ? narrator.describeShelf(split.kept, jpg) : nil
        let text = llm ?? ShelfDescriber.describe(split.kept)
        let t2 = nowMs()
        debug.description = text ?? "（商品が少なく説明なし）"
        debug.describeInfo = "文 \(llm != nil ? "LLM" : "定型") \(t2 - t1)ms"
        logInfo("Assist", "describe: \(products.count) products, kept \(split.kept.count), removed \(split.removed.count) -> \(text ?? "nil")")
        if let text {
            if let camera { describedAt = camera }
            describedKinds = Set(split.mainKinds)
            lastDescription = text
            voice.say(text, interrupt: requested)
        } else if requested {
            voice.say("商品がよく分かりませんでした。棚の正面から、棚全体を写してください。")
        }
    }

    // ------------------------------------------------------------ 長押し: 商品か棚を調べる

    func onLongPressed() {
        listening.cancel()
        voice.say("調べています", interrupt: true, remember: false)
        pendingCapture = { [weak self] f in self?.runLongPress(f) }
    }

    /// 撮った画像で、手前の商品 1 つ（③）か、棚の広い範囲（②）か、その間（一歩下がってもらう）かを決める。
    private func runLongPress(_ f: CapturedFrame) {
        guard let recognizer = recognizer.get else { voice.say("商品認識のモデルを読み込めません。"); return }
        let image = ImageUtil.scaled(f.image, longSide: Self.lookupLongSide)
        do {
            let boxes = try recognizer.detect(image).map(\.0)
            // 商品 1 つかどうかは、枠を使わず、画面の中央の深度だけで決める
            let reading = NearProduct.measure(f.geometry)
            let near = reading.pick
            let distances = boxDistances(boxes, f.geometry)
            let (focus, productBox) = SceneJudge.focus(boxes, nearBox: near?.box)
            let nearText = near?.describe() ?? "近い商品\(reading.describe())"
            let distanceText = distances.map { $0.map { String(format: "%.2f", $0) } ?? "-" }.joined(separator: ",")
            logInfo("Assist", "long press: \(boxes.count) boxes, \(nearText), distances \(distanceText) -> \(focus)")
            debug.longPress = "\(focus.rawValue)（枠 \(boxes.count) 個、\(nearText)）"
            debug.boxDistances = distances
            debug.nearBox = near?.box
            debug.focusBox = productBox
            debug.focusAt = nowMs()
            debug.productBoxes = boxes
            debug.sceneAt = debug.focusAt
            switch focus {
            case .product:
                lookupProduct(image, productBox!)
            case .shelf:
                let products = try recognizer.recognize(image, boxes)
                describe(products, ImageUtil.jpeg(image, longSide: Self.narrateLongSide), boxes.count, nil, requested: true)
            case .stepBack:
                voice.say("棚は一歩下がって全体を、商品はスマホに近づけて写してください。")
            case .nothing:
                voice.say("商品が見えません。棚か商品の方にスマホを向けてください。")
            }
        } catch {
            logInfo("Assist", "long press failed: \(error)")
            voice.say("調べられませんでした。")
        }
    }

    /// 枠ごとのスマホからの距離（分からなければ nil）。近い商品の判定には使わず、画面に出すためだけ。
    private func boxDistances(_ boxes: [[Float]], _ g: CaptureGeometry) -> [Float?] {
        boxes.map { Unprojector.medianDepthMeters(NormalizedBox(left: $0[0], top: $0[1], right: $0[2], bottom: $0[3]), g) }
    }

    /// ③ 手前の商品を Cloud Vision API で読み、カテゴリと商品名を言う。この接近ですでに調べていれば、API を呼ばずにその結果を言う。
    private func lookupProduct(_ image: RGBAImage, _ box: [Float]) {
        if let cached = lookup.cachedResult {
            debug.lookup = "（この接近で調べた結果を使う）"
            speakLookup(cached)
            return
        }
        if !lookup.configured {
            voice.say("商品を調べる設定がされていません。")
            return
        }
        let crop = cropWithMargin(image, box, 0.08)
        let t0 = nowMs()
        let result = lookup.lookup(crop)
        let ms = nowMs() - t0
        switch result {
        case .success(let a):
            debug.lookup = (a?.describe() ?? "分からない") + "（全体 \(ms)ms）"
        case .failure(let e):
            logInfo("Assist", "lookup failed: \(e)")
            debug.lookup = "失敗: \(e)（Cloud Vision \(ms)ms）"
        }
        speakLookup(result)
    }

    private func speakLookup(_ result: Result<ProductAnswer?, Error>) {
        switch result {
        case .success(let a):
            if let a { lastProduct = a.spoken() }
            voice.say(a?.spoken() ?? "商品が分かりませんでした。商品の正面を、カメラに向けてください。")
        case .failure:
            voice.say("調べられませんでした。通信を確認してください。")
        }
    }

    // ------------------------------------------------------------ ④ 人の接近

    private func runPeople(_ f: CapturedFrame) {
        let small = ImageUtil.scaled(f.image, longSide: Self.personLongSide)
        guard let image = ImageUtil.cgImage(small) else { return }
        let w = Float(image.width)
        let h = Float(image.height)
        let detected = personDetector.detect(image)
        // 自分の手や、持っている商品を人と間違えたものを外す
        let judged = detected.map { box -> (NormalizedBox, Bool, Float?) in
            let aspect = (box.bottom - box.top) * h / max((box.right - box.left) * w, 1)
            let depth = Unprojector.medianDepthMeters(box, f.geometry)
            return (box, PersonFilter.isOwnHand(box, aspect: aspect, depthM: depth), depth)
        }
        let boxes = judged.filter { !$0.1 }.map(\.0)
        let t = ProcessInfo.processInfo.systemUptime
        let camera = cameraPosition(f)
        let observations = boxes.map { box -> PersonObservation in
            let d = PersonDistance.estimate(box, f.geometry)
            let pos = d.map { Unprojector.worldAt((box.left + box.right) / 2, (box.top + box.bottom) / 2, $0, f.geometry) }
            return PersonObservation(box: box, distanceM: d, position: pos)
        }
        let tracked = tracker.update(t, observations, camera: camera)
        // 近づいてくる人と、すぐ近くにいる人から鳴らす。近いほど速く
        var sounding: [Int: (Vec3, Float, Float)] = [:]
        for m in tracker.markers() where t - m.lastSeenSeconds < Self.personHoldS {
            let d = Self.horizontal(m.position, camera)
            // NaN だと下の比較がすべて false になり、鳴らす側に入ってしまう
            guard m.position.isFinite, d.isFinite else { continue }
            let approaching = m.motion == .approaching && d < Self.approachAlertM
            if !approaching && d >= Self.closeAlertM { continue }
            sounding[m.id] = (m.position, Self.beepInterval(d), approaching ? 1 : 0.6)
        }
        mixer.setEmitters(sounding)
        func meters(_ x: Float?) -> String { x.map { String(format: "%.1fm", $0) } ?? "?" }
        debug.people = judged.filter(\.1).map { AssistDebug.PersonView(box: $0.0, ownHand: true, label: "手 深度\(meters($0.2))") } +
            tracked.map { AssistDebug.PersonView(box: $0.box, ownHand: false, label: "人 \(meters($0.distanceM)) \($0.motion.labelJa)") }
        debug.peopleInfo = "検出 \(detected.count)（手として外した \(detected.count - boxes.count)）/ 追跡 \(tracker.markers().count) / 鳴らしている \(sounding.count)"
        debug.peopleAt = nowMs()
    }

    // ------------------------------------------------------------ 声の操作

    func startListening() {
        if !listening.available {
            voice.say("このスマホでは、音声認識が使えません。設定でマイクと音声認識を許可してください。", interrupt: true)
            return
        }
        voice.stop()
        debug.voice = "聞いています…"
        listening.listen(onReady: { [weak self] in
            self?.mixer.play(SpatialMixer.Sound(SpatialMixer.makeTick(), gain: 0.6))
        }) { [weak self] text in
            guard let self else { return }
            logInfo("Assist", "heard: \(text ?? "nil")")
            let command = text.flatMap(Commands.parse)
            let shown = text == nil ? "-" : (command?.rawValue ?? "LLM に質問")
            debug.voice = "聞き取り「\(text ?? "（聞き取れず）")」→ \(shown)"
            handle(command, text)
        }
    }

    private func handle(_ command: Command?, _ heard: String?) {
        switch command {
        case .shelves: ask(heard ?? "")   // 棚の種類の判定は止めているので、画像を見て LLM が答える
        case .describe, .lookup: onLongPressed()
        case .pause:
            auto = false
            updateSettings()
            voice.say("お知らせを止めます。人が近づいたときの音は鳴ります。", interrupt: true)
        case .resume:
            auto = true
            updateSettings()
            forgetDescription()
            voice.say("お知らせを再開します。", interrupt: true)
        case .repeatLast: voice.say(voice.lastSpoken.isEmpty ? "まだ何も話していません。" : voice.lastSpoken, interrupt: true)
        case .help: voice.say(Commands.helpText, interrupt: true)
        case nil:
            if let heard, !heard.trimmingCharacters(in: .whitespaces).isEmpty { ask(heard) } else { voice.say("聞き取れませんでした。", interrupt: true) }
        }
    }

    /// 決まった言い方でない質問は、今のカメラの画像と分かっていることを LLM に渡して答えてもらう。
    private func ask(_ question: String) {
        if !narrator.available {
            voice.say("質問に答える設定がされていません。", interrupt: true)
            return
        }
        voice.say("調べています", interrupt: true, remember: false)
        pendingCapture = { [weak self] f in
            guard let self else { return }
            var context = ""
            if !lastDescription.isEmpty { context += "- 最後に伝えた棚の説明: \(lastDescription)\n" }
            if !lastProduct.isEmpty { context += "- 最後に調べた商品: \(lastProduct)\n" }
            let t0 = nowMs()
            let answer = narrator.answer(question, context, ImageUtil.jpeg(f.image, longSide: Self.questionLongSide))
            debug.voice = "質問「\(question)」→「\(answer ?? "失敗")」（\(nowMs() - t0)ms）"
            voice.say(answer ?? "答えられませんでした。通信を確認してください。")
        }
    }

    // ------------------------------------------------------------ 補助

    /// Gemini（キーがなければ nil）。話す文を作る・カテゴリを決めるだけなので、速い lite のモデルから試す（flash は 6〜9 秒、lite は 2 秒ほど）。
    private static func geminiClient() -> GeminiClient? {
        let key = Secrets.value("GEMINI_API_KEY")
        if key.isEmpty { return nil }
        let models = Secrets.value("GEMINI_MODELS", default: "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            .stableSorted { $0.contains("lite") && !$1.contains("lite") }
        return models.isEmpty ? nil : GeminiClient(apiKey: key, models: models)
    }

    /// 壁（ARKit の縦の平面）の数（記録用）。
    private func wallCount(_ frame: ARFrame) -> Int {
        frame.anchors.compactMap { $0 as? ARPlaneAnchor }.filter { $0.alignment == .vertical }.count
    }

    /// カメラの動きの速さを、前のフレームとの差からならして求める。
    private func updateMotion(_ m: simd_float4x4, _ now: Int64) {
        let z = m.columns.2
        let cur: [Float] = [m.columns.3.x, m.columns.3.y, m.columns.3.z, -z.x, -z.y, -z.z]
        let dt = Float(now - lastPoseAt) / 1000
        if let prev = lastPose, dt >= 0.005, dt <= 0.5 {
            let dx = prev[0] - cur[0], dy = prev[1] - cur[1], dz = prev[2] - cur[2]
            let v = (dx * dx + dy * dy + dz * dz).squareRoot() / dt
            let dot = (prev[3] * cur[3] + prev[4] * cur[4] + prev[5] * cur[5]).clamped(-1, 1)
            let w = acos(dot) * 180 / .pi / dt
            speedMps += (v - speedMps) * Self.motionSmoothing
            turnDps += (w - turnDps) * Self.motionSmoothing
        }
        lastPose = cur
        lastPoseAt = now
    }

    private func skip(_ reason: String) {
        var list = skipped
        if let i = list.firstIndex(where: { $0.0 == reason }) { list[i].1 += 1 } else { list.append((reason, 1)) }
        skipped = list
        debug.skipInfo = list.map { "\($0.0) \($0.1)" }.joined(separator: "、")
    }

    private func updateSettings() {
        func ok(_ b: Bool) -> String { b ? "○" : "×" }
        debug.settings = "Gemini\(ok(narrator.available)) Vision\(ok(lookup.configured)) 自動のお知らせ\(ok(auto))"
    }

    private func cameraPosition(_ f: CapturedFrame) -> Vec3 {
        let m = f.geometry.cameraToWorld
        return Vec3(m[12], m[13], m[14])
    }

    private func cropWithMargin(_ image: RGBAImage, _ b: [Float], _ margin: Float) -> RGBAImage {
        let W = image.width, H = image.height
        let w = b[2] - b[0]
        let h = b[3] - b[1]
        let l = Int((b[0] - w * margin) * Float(W)).clamped(0, W - 1)
        let t = Int((b[1] - h * margin) * Float(H)).clamped(0, H - 1)
        let r = Int((b[2] + w * margin) * Float(W)).clamped(l + 1, W)
        let btm = Int((b[3] + h * margin) * Float(H)).clamped(t + 1, H)
        return ImageUtil.crop(image, l, t, r, btm) ?? image
    }

    private static func horizontal(_ a: Vec3, _ b: Vec3) -> Float {
        let dx = a.x - b.x
        let dz = a.z - b.z
        return (dx * dx + dz * dz).squareRoot()
    }

    /// 鳴らす間隔（秒）: 近いほど短く。
    private static func beepInterval(_ d: Float) -> Float { (0.1 + 0.2 * d).clamped(0.15, 1.0) }
}

/// 画面（カメラの映像、認識の状況、操作を受ける面）。処理は Assist にある。
final class AssistViewController: UIViewController {
    private let sceneView = ARSCNView()
    private lazy var assist = Assist(session: sceneView.session)
    private lazy var overlay: DebugOverlay = {
        DebugOverlay(debug: assist.debug)
    }()
    private let touch = TouchSurface()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        sceneView.frame = view.bounds
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.scene = SCNScene()
        sceneView.automaticallyUpdatesLighting = false
        sceneView.isAccessibilityElement = false
        sceneView.accessibilityElementsHidden = true
        view.addSubview(sceneView)

        overlay.frame = view.bounds
        overlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(overlay)

        // 画面全体を 1 つのボタンにする（どこを触ってもよい）
        touch.frame = view.bounds
        touch.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        touch.onDoubleTap = { [weak self] in self?.assist.startListening() }
        touch.onLongPress = { [weak self] in self?.assist.onLongPressed() }
        view.addSubview(touch)
        UIApplication.shared.isIdleTimerDisabled = true
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        assist.viewSize = view.bounds.size
    }

    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .portrait }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        assist.resume()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        assist.pause()
    }
}

/**
 * 画面全体を覆う、操作を受ける面。長押しで調べる、2 回たたくと質問。
 * VoiceOver では、ダブルタップ（accessibilityActivate）が質問、「調べる」のアクションかダブルタップ長押しが長押し。
 */
final class TouchSurface: UIView {
    var onDoubleTap: (() -> Void)?
    var onLongPress: (() -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        let double = UITapGestureRecognizer(target: self, action: #selector(doubleTapped))
        double.numberOfTapsRequired = 2
        addGestureRecognizer(double)
        let long = UILongPressGestureRecognizer(target: self, action: #selector(longPressed(_:)))
        long.minimumPressDuration = 0.5
        addGestureRecognizer(long)
        isAccessibilityElement = true
        accessibilityLabel = "長押しで調べる。2回たたくと質問"
        accessibilityTraits = [.button]
        accessibilityCustomActions = [UIAccessibilityCustomAction(name: "調べる") { [weak self] _ in
            self?.onLongPress?()
            return true
        }]
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    @objc private func doubleTapped() { onDoubleTap?() }

    @objc private func longPressed(_ g: UILongPressGestureRecognizer) {
        if g.state == .began { onLongPress?() }
    }

    override func accessibilityActivate() -> Bool {
        onDoubleTap?()
        return true
    }
}
