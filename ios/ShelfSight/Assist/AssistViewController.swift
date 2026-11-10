import ARKit
import AVFoundation
import CoreLocation
import SceneKit
import UIKit

/**
 * 視覚障害のある人のための、コンビニの棚と商品の案内。胸の前にスマホを立てて持ち、イヤホンで聞く（Android 版の AssistActivity）。
 *
 * 話すのはタップしたときだけ（人が近づいたときの音は別）。歩いている間は、認識した商品で店のモデルと地図を作り続ける。
 * 商品の位置や距離（Android 版の Depth API を使うところ）は、すべて LiDAR の深度（ARKit の sceneDepth）で測る。
 *
 * 操作
 * - 1 回たたく: 棚に向かっていれば、その棚の主な商品と、上・真ん中・下の商品。近くに棚がいくつかあれば、方向ごとにチャイムを鳴らして主な商品を
 * - 2 回たたく: 手前の商品を Cloud Vision API に聞き、商品名とカテゴリを言う（1 回の接近で 1 回だけ呼ぶ）
 * - 長押し: 話しかける。「〇〇はどこ？」などに、店の地図・今いる場所・調べた商品を知っている LLM が答える
 * - 人が近づいてくる: その人の方向から警告音を鳴らす
 * VoiceOver が有効なときは、ダブルタップが 1 回たたく、ダブルタップ長押しが長押し、「商品名を調べる」のアクションが 2 回たたくになる。
 */
final class Assist: NSObject, ARSessionDelegate, CLLocationManagerDelegate {
    // ------------------------------------------------------------ 定数（Android 版と同じ）
    private static let sceneIntervalMs: Int64 = 1000
    private static let personIntervalMs: Int64 = 200
    private static let sceneLongSide = 1280
    private static let personLongSide = 640
    private static let lookupLongSide = 1600
    private static let maxCameraImageWidth: CGFloat = 1920
    private static let questionLongSide = 1280
    private static let maxLookedUp = 20
    private static let approachGapMs: Int64 = 2000
    private static let approachMoveM: Float = 0.6
    private static let personHoldS = 1.0
    private static let approachAlertM: Float = 6
    private static let storeEstimateMs: Int64 = 3000
    private static let demoTickS = 0.05
    private static let backendUploadMs: Int64 = 60_000   // 送るたびに店の検索（Places API）と OR-Tools が 1 回ずつ動く
    /// 判定待ちの切り抜きは、メモリを食わないよう数を抑える。iOS は Core ML で判定が速く計算に余裕があるので、Android 版（240）より多く持つ。
    private static let maxPendingCrops = 480
    private static let identifyIdleS = 0.03
    private static let deviceHeightM: Float = 1.3
    /// スマホ（胸の高さ）から見たレジの天板の高さ。
    private static let counterBelowDeviceM: ClosedRange<Float> = -0.6 ... -0.1
    private static let minCounterAreaM2: Float = 0.1
    /// 残りのメモリがこれより少ないときは、商品の解析を休む（OS に止められないための安全装置）。
    private static let minAvailableMB: Double = 400
    private static let demoListener = Listener(position: .zero, forwardX: 0, forwardZ: -1)

    // ------------------------------------------------------------ AR と画面の表示
    private let session: ARSession
    let debug = AssistDebug()
    /// 画面の大きさ（画面の表示用。メインスレッドで書く）。
    @Locked var viewSize = CGSize.zero
    private var greeted = false
    /// モデルの説明の文章ができたときに呼ぶ（メインスレッド）。
    var onModelText: ((String) -> Void)?

    // ------------------------------------------------------------ 音
    @Locked private var listener: Listener? = nil
    /// 棚や商品の説明を始めてから 90 度より大きく向きを変えたら、説明をやめる（向いている棚が変わったため）。
    private let turnGuard = TurnGuard()
    /// 立体音響のテストのタブを開いている。
    @Locked private var showTest = false
    private lazy var mixer = SpatialMixer(hrtf: Hrtf.load()) { [weak self] in
        guard let self else { return nil }
        return listener ?? (showTest ? Self.demoListener : nil)
    }
    private lazy var voice = Voice(mixer: mixer)
    private let listening = Listening()
    private let chime = SpatialMixer.makeChime()

    // ------------------------------------------------------------ キュー（Android 版のスレッド）
    // どのキューも、1 回の処理が終わるたびに一時的なオブジェクト（autorelease されたもの）を捨てる（.workItem）。
    // 既定のままだと、休まず回り続けるキューでは捨てる機会がなく、カメラの画像や ONNX Runtime の出力がたまって落ちる
    private let arQueue = DispatchQueue(label: "ar", qos: .userInteractive, autoreleaseFrequency: .workItem)
    private let sceneWorker = DispatchQueue(label: "scene", qos: .utility, autoreleaseFrequency: .workItem)
    private let identifyWorker = DispatchQueue(label: "identify", qos: .utility, autoreleaseFrequency: .workItem)
    private let personWorker = DispatchQueue(label: "person", qos: .userInitiated, autoreleaseFrequency: .workItem)
    private let lookupWorker = DispatchQueue(label: "lookup", qos: .userInitiated, autoreleaseFrequency: .workItem)
    private let backendWorker = DispatchQueue(label: "backend", qos: .utility, autoreleaseFrequency: .workItem)
    private let sceneBusy = AtomicFlag()
    private let personBusy = AtomicFlag()
    private let backendBusy = AtomicFlag()

    // ------------------------------------------------------------ 認識
    /// 商品認識。CPU 側のメモリが増えたら、解析を止めずに裏で作り直す（RecognizerPool）。
    private let recognizer = RecognizerPool()
    private let personDetector: PersonDetector = YoloPersonDetector()
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

    // ------------------------------------------------------------ 地図
    /// 点ごとの切り抜き。identifyLoop が優先度の高いものから MobileCLIP で判定する。
    private let cropQueue = CropQueue<RGBAImage>(capacity: Assist.maxPendingCrops)
    /// 判定した確からしさを、地図で近くの判定との一致で付け直す。
    private let confidence = NeighborConfidence.fromBundle()
    /// 認識の結果を、AR 空間に貯めたもの（アプリを閉じると消える）。
    private let shelfMap = ShelfMap()
    /// 認識した商品の位置から推定する店のモデル。sceneWorker からだけ触る。
    private lazy var storeModel = StoreModel(themes: Assist.loadThemes())
    @Locked private var storeView: StoreView? = nil
    private var lastStoreEstimate: Int64 = 0
    private var lastBackendUpload: Int64 = 0
    private let backend = BackendClient(baseUrl: Secrets.value("BACKEND_URL"), apiKey: Secrets.value("BACKEND_API_KEY"))
    private let locationManager = CLLocationManager()
    @Locked private var location: CLLocation? = nil
    /// この店の、前回までに保存された地図。
    @Locked private var savedView: StoreView? = nil
    @Locked private var people: [PersonMarker] = []
    @Locked private var sceneGeometry: CaptureGeometry? = nil

    // ------------------------------------------------------------ 状態
    @Locked private var pendingCapture: ((CapturedFrame) -> Void)? = nil
    @Locked private var deviceAt: Vec3? = nil
    @Locked private var devicePose: simd_float4x4? = nil
    /// 調べた商品の記録（質問に答える LLM に渡す）。
    @Locked private var lookedUp: [LookedUp] = []
    @Locked var topView = false
    @Locked var storeLayers = Set(StoreLayer.allCases)
    // AR のキューだけで使う
    private var lastScene: Int64 = 0
    private var lastPerson: Int64 = 0
    // sceneWorker だけで使う: 今の接近で、すぐ前の商品を最後に見た時刻と、そのときの場所
    private var approachSeenAt: Int64 = 0
    private var approachFrom: Vec3?
    // テストの音源（メインスレッドで動かし、基準の場所はオーディオのスレッドからも読む）
    private var demoTimer: Timer?
    @Locked private var demoStart: Int64 = 0
    private var demoSlot: Int?
    @Locked private var demoOrigin: Listener? = nil

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
        locationManager.delegate = self
        updateSettings()
        Listening.requestPermissions()
        identifyWorker.async { [weak self] in self?.identifyLoop() }
    }

    func resume() {
        guard ARWorldTrackingConfiguration.isSupported else {
            voice.say("このスマホでは、カメラの位置合わせが使えません。")
            debug.tracking = "ARKit が使えません"
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.isAutoFocusEnabled = true
        config.planeDetection = [.horizontal, .vertical]
        // 商品の位置と距離は LiDAR の深度で測る（LiDAR のある機種だけ。ない機種では壁の平面と画面の中央の枠で代わりにする）
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        } else {
            debug.near = "LiDAR がない機種なので、深度を測れません"
        }
        // 商品の文字を読めるよう、30fps 以上で幅 1920 までのいちばん大きいカメラ画像を使う
        if let best = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter({ $0.imageResolution.width <= Self.maxCameraImageWidth && $0.framesPerSecond >= 30 })
            .max(by: { $0.imageResolution.width * $0.imageResolution.height < $1.imageResolution.width * $1.imageResolution.height }) {
            config.videoFormat = best
        }
        session.run(config)
        sceneWorker.async { [recognizer] in _ = recognizer.get }   // 優先度を下げたキューで読み込む
        requestLocation()
        if !greeted {
            greeted = true
            voice.say("準備ができました。1回たたくと棚の案内、2回たたくと商品の名前、長押しで質問ができます。")
            voice.prepare(["調べています"])
        }
    }

    func pause() {
        listening.cancel()
        stopDemo()
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
        let position = Vec3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        deviceAt = position
        devicePose = camera.viewMatrix(for: .portrait).inverse
        let z = m.columns.2
        let fx = -z.x
        let fz = -z.z
        let len = (fx * fx + fz * fz).squareRoot()
        if len > 0.3 {
            let l = Listener(position: position, forwardX: fx / len, forwardZ: fz / len)
            listener = l
            if turnGuard.turned(l, turn: voice.turn, speaking: voice.speaking, now: nowMs()) {
                voice.stop()
                debug.description = "（向きが 90 度より大きく変わったので、説明をやめました）"
            }
            debug.headingDeg = atan2(fx, -fz) * 180 / .pi
        }

        let now = nowMs()
        if let target = pendingCapture, let f = FrameCapture.capture(frame, depthEnabled: true, fullResolution: true) {
            pendingCapture = nil
            lookupWorker.async { target(f) }
        }
        if now - lastPerson >= Self.personIntervalMs && personBusy.tryAcquire() {
            if let f = FrameCapture.capture(frame, depthEnabled: true, usualLongSide: Self.personLongSide) {
                lastPerson = now
                personWorker.async { [self] in runPeople(f); personBusy.release() }
            } else {
                personBusy.release()
            }
        }
        guard now - lastScene >= Self.sceneIntervalMs else { return }
        // 残りのメモリが少ないときは休む（増え続ける原因がほかにあっても、OS に止められないように）
        let available = Memory.availableMB()
        if available < Self.minAvailableMB {
            debug.sceneMs = String(format: "メモリが少ないので解析を休み中（残り %.0fMB）", available)
            return
        }
        if sceneBusy.tryAcquire() {
            if let f = FrameCapture.capture(frame, depthEnabled: true, usualLongSide: Self.sceneLongSide) {
                lastScene = now
                let planes = frame.anchors.compactMap { $0 as? ARPlaneAnchor }
                let walls = Self.walls(planes)
                let counters = Self.counterPoints(planes, deviceY: position.y)
                let heading = debug.headingDeg
                sceneWorker.async { [self] in runScene(f, walls, counters, heading); sceneBusy.release() }
            } else {
                sceneBusy.release()
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        logInfo("Assist", "AR session failed: \(error)")
        debug.tracking = "ARKit のエラー: \(error.localizedDescription)"
    }

    private static func worldPoint(_ a: ARPlaneAnchor, _ x: Float, _ z: Float) -> Vec3 {
        let w = a.transform * SIMD4<Float>(a.center.x + x, a.center.y, a.center.z + z, 1)
        return Vec3(w.x, w.y, w.z)
    }

    /// ARKit の縦の平面を壁にする。
    private static func walls(_ planes: [ARPlaneAnchor]) -> [Wall] {
        planes.filter { $0.alignment == .vertical }.map { a in
            let n = a.transform.columns.1
            return Wall(center: worldPoint(a, 0, 0), normal: Vec3(n.x, n.y, n.z), radius: max(a.planeExtent.width, a.planeExtent.height) / 2)
        }
    }

    /// 腰の高さの水平面（レジの天板）の上の点。商品の棚の段は商品に隠れるので、ほぼレジだけが残る。
    private static func counterPoints(_ planes: [ARPlaneAnchor], deviceY: Float) -> [Vec3] {
        planes.filter {
            $0.alignment == .horizontal && counterBelowDeviceM.contains(worldPoint($0, 0, 0).y - deviceY) &&
                $0.planeExtent.width * $0.planeExtent.height >= minCounterAreaM2
        }.flatMap { a -> [Vec3] in
            let x = a.planeExtent.width / 2, z = a.planeExtent.height / 2
            return [(0, 0), (x, z), (-x, z), (x, -z), (-x, -z)].map { worldPoint(a, $0.0, $0.1) }
        }
    }

    // ------------------------------------------------------------ ①② 写っているものの判定

    /// 1 秒ごと: 商品の枠を出して写っているものを判定し、枠ごとの切り抜きを判定待ちに入れ、画像と姿勢を記録する。
    private func runScene(_ f: CapturedFrame, _ walls: [Wall], _ counters: [Vec3], _ heading: Float?) {
        guard let recognizer = recognizer.get else { debug.sceneMs = "商品認識のモデルを読み込めません"; return }
        let image = ImageUtil.scaled(f.image, longSide: Self.sceneLongSide)
        do {
            let now = nowMs()
            let detected = try recognizer.detect(image)
            let boxes = detected.map(\.0)
            let detectMs = nowMs() - now
            debug.productBoxes = boxes  // 名前付けに時間がかかるので、枠は先に出す
            debug.sceneAt = nowMs()
            // スマホのすぐ前の商品（枠は使わず、画面の中央の LiDAR の深度だけで決める）
            let reading = NearProduct.measure(f.geometry)
            let near = reading.pick
            let switched = judge.update(SceneEvidence(shelves: 0, productBoxes: boxes, near: near != nil))
            queueCrops(image, detected, f, walls, now, recognizer)
            self.recognizer.reloadIfNeeded()
            let queueMs = nowMs() - now - detectMs
            updateStoreModel(boxes, counters, f, now)
            recorder.get.record(Int64(Date().timeIntervalSince1970 * 1000), image, f.geometry, headingDeg: heading, walls: walls.count, products: [])
            debug.boxDistances = boxDistances(boxes, f.geometry)
            debug.nearBox = near?.box
            debug.nearDistance = near?.distanceM
            if f.geometry.depth != nil { debug.near = reading.describe() }
            debug.scene = judge.scene
            debug.sceneMs = "枠 \(detectMs)ms / 切り抜き \(queueMs)ms・判定待ち \(cropQueue.size) / 壁 \(walls.count)"
            sceneGeometry = f.geometry
            debug.products = shelfMap.visible(f.geometry)

            // 新しい商品に近づいたら、前に調べた結果を忘れる（次の 2 回たたくで API を 1 回呼ぶ）
            let camera = cameraPosition(f)
            let gone = approachSeenAt != 0 && now - approachSeenAt > Self.approachGapMs
            let moved = (approachFrom.map { Self.horizontal($0, camera) } ?? 0) > Self.approachMoveM
            if (switched && judge.scene == .oneProduct) || gone || moved {
                lookup.newApproach()
                approachSeenAt = 0
                approachFrom = nil
            }
            if near != nil || SceneJudge.mainProduct(boxes) != nil {
                approachSeenAt = now
                if approachFrom == nil { approachFrom = camera }
            }
        } catch {
            logInfo("Assist", "scene analysis failed: \(error)")
        }
    }

    /// 正立画像の枠の 3D の位置と実際の大きさ。LiDAR の深度も壁もなければ nil。
    private func locate(_ b: [Float], _ f: CapturedFrame, _ walls: [Wall]) -> (position: Vec3, widthM: Float, heightM: Float)? {
        let g = f.geometry
        let camera = cameraPosition(f)
        let cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2
        let depth = Unprojector.medianDepthMeters(NormalizedBox(left: b[0], top: b[1], right: b[2], bottom: b[3]), g)
        let along = Unprojector.worldAt(cx, cy, 1, g)
        guard let pos = ShelfMap.place(camera: camera, along: along, depthPoint: depth.map { Unprojector.worldAt(cx, cy, $0, g) }, walls: walls) else { return nil }
        let (w, h) = ShelfMap.sizeInMeters(b[2] - b[0], b[3] - b[1], pos.distanceTo(camera), g)
        return (pos, w, h)
    }

    /// まだ判定していない点の切り抜きを、判定待ちに入れる。判定済みの場所は見えたことだけ記録する。
    private func queueCrops(_ image: RGBAImage, _ boxes: [([Float], Float)], _ f: CapturedFrame, _ walls: [Wall], _ now: Int64, _ recognizer: ProductRecognizer) {
        let camera = cameraPosition(f)
        for (b, confidence) in boxes {
            guard let at = locate(b, f, walls) else { continue }
            if shelfMap.touch(at.position, at.widthM, at.heightM, now, camera) { continue }
            guard let crop = recognizer.cropOf(image, b) else { continue }
            let (added, dropped) = cropQueue.add(CropQueue.Entry(crop: crop, box: b, position: at.position, camera: camera, widthM: at.widthM, heightM: at.heightM,
                                                                 area: SceneJudge.area(b), time: now, confidence: confidence))
            if let dropped { shelfMap.removePending(dropped) }
            // 確からしさが低かった場所は、十分大きく写るまで茶色の点も出さない
            if added { shelfMap.addPending(at.position) }
        }
    }

    /// 判定待ちの切り抜きを、優先度の高いものから 1 枚ずつ判定して地図に置く。
    private func identifyLoop() {
        while true {
            // このキューは回り続けて戻らないので、1 枚ごとに一時的なオブジェクトを捨てる
            autoreleasepool { identifyOne() }
        }
    }

    private func identifyOne() {
        guard Memory.availableMB() >= Self.minAvailableMB, let recognizer = recognizer.get,
              let e = cropQueue.take() else {
            Thread.sleep(forTimeInterval: Self.identifyIdleS)
            return
        }
        do {
            let (judged, top) = try recognizer.judge(e.crop, e.box)
            let p = confidence.adjust(e.position, sizeM: max(e.widthM, e.heightM), top: top) >= ProductRecognizer.knownProb ? judged : nil
            guard let p else {
                // MobileCLIP の確からしさが低いものは無視する（地図にも店のモデルにも入れない。棚の形は枠の位置から別に入っている）
                cropQueue.reject(e)
                shelfMap.removePending(e.position)
                return
            }
            shelfMap.add(p.name, p.genre, p.fine, e.position, e.widthM, e.heightM, e.time, e.camera)
            if let g = sceneGeometry { debug.products = shelfMap.visible(g) }   // 分かったものから順に出す
            sceneWorker.async { [self] in
                storeModel.addProduct(Double(e.camera.x), Double(e.camera.z), Double(e.position.x), Double(e.position.z), p.name, p.genre ?? "")
            }
        } catch {
            logInfo("Assist", "crop identification failed: \(error)")
        }
    }

    /// 店のモデルに今の 1 枚を足す。枠はまず位置だけ使う（名前は判定が終わってから identifyOne が足す）。
    private func updateStoreModel(_ boxes: [[Float]], _ counters: [Vec3], _ f: CapturedFrame, _ now: Int64) {
        let cam = cameraPosition(f)
        let cx = Double(cam.x), cz = Double(cam.z)
        storeModel.addCamera(cx, cz)
        for b in boxes {
            guard let depth = Unprojector.medianDepthMeters(NormalizedBox(left: b[0], top: b[1], right: b[2], bottom: b[3]), f.geometry), depth <= ShelfMap.maxDepthM else { continue }
            let p = Unprojector.worldAt((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, depth, f.geometry)
            storeModel.addProduct(cx, cz, Double(p.x), Double(p.z), "", "")
        }
        for c in counters { storeModel.addCounter(Double(c.x), Double(c.z)) }
        if now - lastStoreEstimate < Self.storeEstimateMs { return }
        lastStoreEstimate = now
        let view = storeModel.estimate()
        storeView = view
        if backend.configured, let at = location, now - lastBackendUpload >= Self.backendUploadMs, backendBusy.tryAcquire() {
            lastBackendUpload = now
            let body = storeModel.requestBody()
            backendWorker.async { [self] in
                backend.upload(lat: at.coordinate.latitude, lng: at.coordinate.longitude, body: body)
                backendBusy.release()
            }
        }
        let description = StoreModel.text(view)
        DispatchQueue.main.async { [weak self] in self?.onModelText?(description) }
    }

    /// 枠ごとのスマホからの距離（分からなければ nil）。画面に出すためだけ。
    private func boxDistances(_ boxes: [[Float]], _ g: CaptureGeometry) -> [Float?] {
        boxes.map { Unprojector.medianDepthMeters(NormalizedBox(left: $0[0], top: $0[1], right: $0[2], bottom: $0[3]), g) }
    }

    // ------------------------------------------------------------ 1 回たたく: 棚の案内

    func onTap() {
        listening.cancel()
        guard let l = listener else {
            voice.say("位置を合わせています。少し周りを見回してから、もう一度たたいてください。", interrupt: true)
            return
        }
        switch StoreGuide.tapAnswer(storeView, shelfMap.snapshot(), l, groups, floorY: l.position.y - Self.deviceHeightM) {
        case .sentence(let text):
            debug.description = text
            voice.say(text, interrupt: true)
        case .around(let clauses, let shelves):
            let all = clauses.joined()
            debug.description = all
            voice.remember(all)
            for (i, clause) in clauses.enumerated() {
                let at = shelves[i].at
                voice.say(clause, interrupt: i == 0, remember: false, before: SpatialMixer.Sound(chime, position: { at }))
            }
        case .unknown:
            voice.say("まだ棚が分かりません。棚の方を向いて、少し歩いてください。", interrupt: true)
        }
        turnGuard.start(l, turn: voice.turn, now: nowMs())
    }

    // ------------------------------------------------------------ 2 回たたく: 商品の名前

    func onProductTap() {
        listening.cancel()
        voice.say("調べています", interrupt: true, remember: false)
        pendingCapture = { [weak self] f in self?.runProductTap(f) }
    }

    /// 撮った画像の手前の商品を調べる。LiDAR で近い商品が測れればその範囲を、測れなければ画面の中央の枠を Cloud Vision に渡す。
    private func runProductTap(_ f: CapturedFrame) {
        guard let recognizer = recognizer.get else { voice.say("商品認識のモデルを読み込めません。"); return }
        let image = ImageUtil.scaled(f.image, longSide: Self.lookupLongSide)
        do {
            let boxes = try recognizer.detect(image).map(\.0)
            let reading = NearProduct.measure(f.geometry)
            let box = reading.pick?.box ?? SceneJudge.productTapBox(boxes)
            logInfo("Assist", "product tap: \(boxes.count) boxes, \(reading.describe())")
            debug.longPress = "枠 \(boxes.count) 個、\(reading.describe())"
            debug.focusBox = box
            debug.focusAt = nowMs()
            debug.nearBox = reading.pick?.box
            debug.productBoxes = boxes
            debug.sceneAt = debug.focusAt
            lookupProduct(image, box, cameraPosition(f))
        } catch {
            logInfo("Assist", "product tap failed: \(error)")
            voice.say("調べられませんでした。")
        }
    }

    /// ③ 手前の商品を Cloud Vision API で読み、商品名とカテゴリを言う。この接近ですでに調べていれば、API を呼ばずにその結果を言う。
    private func lookupProduct(_ image: RGBAImage, _ box: [Float], _ at: Vec3) {
        if let cached = lookup.cachedResult {
            debug.lookup = "（この接近で調べた結果を使う）"
            speakLookup(cached, at)
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
        speakLookup(result, at)
    }

    private func speakLookup(_ result: Result<ProductAnswer?, Error>, _ at: Vec3) {
        switch result {
        case .success(let a):
            if let a {
                let name = a.spoken()
                var list = lookedUp
                if list.last?.name != name { list.append(LookedUp(name: name, at: at, time: nowMs())) }
                if list.count > Self.maxLookedUp { list.removeFirst() }
                lookedUp = list
            }
            voice.say(a?.spoken() ?? "商品が分かりませんでした。商品の正面を、カメラに向けてください。")
        case .failure:
            voice.say("調べられませんでした。通信を確認してください。")
        }
        if let l = listener { turnGuard.start(l, turn: voice.turn, now: nowMs()) }
    }

    // ------------------------------------------------------------ ④ 人の接近

    private func runPeople(_ f: CapturedFrame) {
        let small = ImageUtil.scaled(f.image, longSide: Self.personLongSide)
        let w = Float(small.width)
        let h = Float(small.height)
        let detected = personDetector.detect(small)
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
        let visible = tracker.markers().filter { t - $0.lastSeenSeconds < Self.personHoldS }
        people = visible
        // 近づいてくる人の場所から鳴らす。近いほど速く
        var sounding: [Int: (Vec3, Float, Float)] = [:]
        for m in visible {
            let d = Self.horizontal(m.position, camera)
            // NaN だと下の比較がすべて false になり、鳴らす側に入ってしまう
            guard m.position.isFinite, d.isFinite, m.motion == .approaching, d < Self.approachAlertM else { continue }
            sounding[m.id] = (m.position, Self.beepInterval(d), 1)
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
        case .repeatLast: voice.say(voice.lastSpoken.isEmpty ? "まだ何も話していません。" : voice.lastSpoken, interrupt: true)
        case .help: voice.say(Commands.helpText, interrupt: true)
        case nil:
            if let heard, !heard.trimmingCharacters(in: .whitespaces).isEmpty { ask(heard) } else { voice.say("聞き取れませんでした。", interrupt: true) }
        }
    }

    /// 決まった言い方でない質問は、今のカメラの画像と、店の地図や調べた商品を LLM に渡して答えてもらう。
    private func ask(_ question: String) {
        if !narrator.available {
            voice.say("質問に答える設定がされていません。", interrupt: true)
            return
        }
        voice.say("調べています", interrupt: true, remember: false)
        pendingCapture = { [weak self] f in
            guard let self else { return }
            let context = StoreGuide.context(storeView, listener, shelfMap.snapshot(), lookedUp, nowMs(), voice.lastSpoken)
            let t0 = nowMs()
            let answer = narrator.answer(question, context, ImageUtil.jpeg(f.image, longSide: Self.questionLongSide))
            debug.voice = "質問「\(question)」→「\(answer ?? "失敗")」（\(nowMs() - t0)ms）"
            voice.say(answer ?? "答えられませんでした。通信を確認してください。")
        }
    }

    // ------------------------------------------------------------ 店の地図（GPS とバックエンド）

    /// GPS から店を求め、保存済みの地図があれば読み込む。店の地図は任意の機能なので、断られてもカメラと音声は使える。
    private func requestLocation() {
        guard backend.configured, location == nil else { return }
        switch locationManager.authorizationStatus {
        case .notDetermined: locationManager.requestWhenInUseAuthorization()
        case .authorizedWhenInUse, .authorizedAlways: locationManager.requestLocation()
        default: break
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) { requestLocation() }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard location == nil, let found = locations.last else { return }
        location = found
        backendWorker.async { [self] in
            guard let map = backend.lookup(lat: found.coordinate.latitude, lng: found.coordinate.longitude) else { return }
            savedView = map.view
            if map.view != nil { voice.say("\(map.name)の地図を読み込みました。") }
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        logInfo("Assist", "location failed: \(error)")
    }

    // ------------------------------------------------------------ 画面に渡すもの

    func modelContent() -> StoreCanvas.Content {
        .model(StoreCanvas.ModelState(
            view: storeView, saved: savedView, phone: devicePose, floorY: (deviceAt?.y ?? 0) - Self.deviceHeightM,
            items: shelfMap.snapshot(), pending: shelfMap.pendingPoints(), people: people, topView: topView, layers: storeLayers
        ))
    }

    func demoContent() -> StoreCanvas.Content {
        let seconds = demoSeconds()
        return .demo(StoreCanvas.DemoState(listener: listener ?? Self.demoListener, source: demoPosition(), sounding: SpatialDemo.chimeSlot(seconds) != nil))
    }

    /// 店のモデルの全体が見える視点の中心と半径。まだモデルがなければ nil。
    func modelFit(aspect: CGFloat) -> (Vec3, Float)? {
        guard let v = storeView, let r = v.room else { return nil }
        let cu = (r[0] + r[2]) / 2, cv = (r[1] + r[3]) / 2
        let cs = cos(v.yaw), sn = sin(v.yaw)
        // 縦長の画面は横が狭いので、棚のジャンルの文字が画面に入るまで引いて見る
        return (Vec3(Float(cu * cs - cv * sn), (deviceAt?.y ?? 0) - Self.deviceHeightM, Float(cu * sn + cv * cs)), Float(max(r[2] - r[0], r[3] - r[1]) / 2 / Double(max(aspect, 0.1))))
    }

    // ------------------------------------------------------------ 立体音響のテスト（メインスレッド）

    func setTestTab(_ open: Bool) {
        showTest = open
        stopDemo()
        if open { startDemo() }
    }

    private func startDemo() {
        listening.cancel()
        demoStart = nowMs()
        demoSlot = nil
        demoOrigin = nil
        demoTimer = Timer.scheduledTimer(withTimeInterval: Self.demoTickS, repeats: true) { [weak self] _ in
            guard let self else { return }
            let slot = SpatialDemo.chimeSlot(demoSeconds())
            if let slot, slot != demoSlot {
                mixer.play(SpatialMixer.Sound(chime, position: { [weak self] in self?.demoPosition() }))
            }
            demoSlot = slot
        }
    }

    private func stopDemo() {
        demoTimer?.invalidate()
        demoTimer = nil
    }

    private func demoSeconds() -> Double { Double(nowMs() - demoStart) / 1000 }

    /// ARKit が追跡していなくても試せるよう、そのときは原点で正面を向いていることにする。
    private func demoPosition() -> Vec3 {
        let current = listener
        var origin = demoOrigin
        // まだ基準がないか、仮の基準のあとで追跡が始まったら、今の場所と向きを基準にする
        if origin == nil || (current != nil && origin!.position == .zero && origin!.forwardZ == -1) {
            origin = current ?? Self.demoListener
            demoOrigin = origin
        }
        return SpatialDemo.world(origin!, demoSeconds())
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

    private static func loadThemes() -> [Theme] {
        guard let url = Bundle.main.url(forResource: "store_themes", withExtension: "json"), let data = try? Data(contentsOf: url) else { return [] }
        return StoreModel.parseThemes(data)
    }

    private func updateSettings() {
        func ok(_ b: Bool) -> String { b ? "○" : "×" }
        debug.settings = "Gemini\(ok(narrator.available)) Vision\(ok(lookup.configured)) 店の地図\(ok(backend.configured))"
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
    private static func beepInterval(_ d: Float) -> Float { (0.1 + 0.2 * d).clamped(0.3, 1.2) }
}

/// 画面（カメラの映像、認識の状況、3D の店、操作を受ける面）。処理は Assist にある。
final class AssistViewController: UIViewController {
    private let sceneView = ARSCNView()
    private lazy var assist = Assist(session: sceneView.session)
    private lazy var overlay = DebugOverlay(debug: assist.debug)
    private let touch = TouchSurface()
    private let orbit = OrbitCamera()
    /// テストの人を、少し上の後ろから見る。
    private let testOrbit: OrbitCamera = {
        let o = OrbitCamera()
        o.fit(Vec3(0, 0.9, 0), radius: 3.9)
        o.setView(yaw: 0, pitch: 0.75)
        return o
    }()
    private lazy var modelCanvas = StoreCanvas(orbit: orbit)
    private lazy var testCanvas = StoreCanvas(orbit: testOrbit)
    private let tabBar = UIStackView()
    private var tabs: [UIButton] = []
    private let topViewButton = UIButton(type: .system)
    private lazy var legend = StoreLegend { [weak self] in self?.assist.storeLayers = $0 }
    private let modelText = UITextView()

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
        touch.onTap = { [weak self] in self?.assist.onTap() }
        touch.onDoubleTap = { [weak self] in self?.assist.onProductTap() }
        touch.onLongPress = { [weak self] in self?.assist.startListening() }
        view.addSubview(touch)

        modelCanvas.accessibilityLabel = "店のモデルの 3D 表示"
        modelCanvas.content = { [weak self] in self?.assist.modelContent() }
        testCanvas.accessibilityLabel = "立体音響のテスト。同じ場所から5回鳴らしたあと、ゆっくり別の場所へ移ります"
        testCanvas.content = { [weak self] in self?.assist.demoContent() }
        for canvas in [modelCanvas, testCanvas] {
            canvas.frame = view.bounds
            canvas.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            view.addSubview(canvas)
        }

        modelText.isEditable = false
        modelText.backgroundColor = uiColor(0, alpha: 0.67)
        modelText.textColor = .white
        modelText.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        modelText.text = "歩くと店のモデルができます"
        view.addSubview(modelText)
        assist.onModelText = { [weak self] in self?.modelText.text = $0 }

        tabBar.axis = .horizontal
        tabBar.distribution = .fillEqually
        for (i, name) in ["案内", "3D", "立体音響"].enumerated() {
            let b = barButton(name)
            b.tag = i
            b.addTarget(self, action: #selector(tabTapped(_:)), for: .touchUpInside)
            tabs.append(b)
            tabBar.addArrangedSubview(b)
        }
        view.addSubview(tabBar)

        topViewButton.setTitle("2D", for: .normal)
        topViewButton.setTitleColor(.white, for: .normal)
        topViewButton.backgroundColor = uiColor(0, alpha: 0.63)
        topViewButton.addTarget(self, action: #selector(topViewTapped), for: .touchUpInside)
        view.addSubview(topViewButton)
        view.addSubview(legend)

        select(0)
        UIApplication.shared.isIdleTimerDisabled = true
    }

    private func barButton(_ title: String) -> UIButton {
        let b = UIButton(type: .system)
        b.setTitle(title, for: .normal)
        b.setTitleColor(.white, for: .normal)
        b.backgroundColor = uiColor(0, alpha: 0.63)
        return b
    }

    @objc private func tabTapped(_ b: UIButton) { select(b.tag) }

    @objc private func topViewTapped() {
        assist.topView.toggle()
        let top = assist.topView
        topViewButton.setTitle(top ? "3D" : "2D", for: .normal)
        legend.showTopViewRows(top)
    }

    private func select(_ index: Int) {
        let model = index == 1
        for (i, b) in tabs.enumerated() {
            b.alpha = i == index ? 1 : 0.55
            b.accessibilityTraits = i == index ? [.button, .selected] : [.button]
        }
        overlay.isHidden = index != 0
        touch.isHidden = index != 0
        modelCanvas.isHidden = !model
        testCanvas.isHidden = index != 2
        modelText.isHidden = !model
        topViewButton.isHidden = !model
        legend.isHidden = !model
        assist.setTestTab(index == 2)
        if model, let (center, radius) = assist.modelFit(aspect: view.bounds.width / max(view.bounds.height, 1)) {
            orbit.fit(center, radius: radius)
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        assist.viewSize = view.bounds.size
        let w = view.bounds.width, h = view.bounds.height, top = view.safeAreaInsets.top
        tabBar.frame = CGRect(x: 0, y: top, width: w, height: 44)
        topViewButton.frame = CGRect(x: w - 56, y: top + 48, width: 52, height: 44)
        legend.frame = CGRect(x: 0, y: top + 48, width: 130, height: legend.systemLayoutSizeFitting(UIView.layoutFittingCompressedSize).height)
        modelText.frame = CGRect(x: 0, y: h * 2 / 3, width: w, height: h / 3)
        for canvas in [modelCanvas, testCanvas] {
            canvas.topInset = top + 44
            canvas.bottomInset = canvas === modelCanvas ? h / 3 : 0
        }
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
 * 画面全体を覆う、操作を受ける面。1 回たたくと棚の案内、2 回たたくと商品の名前、長押しで質問。
 * VoiceOver では、ダブルタップ（accessibilityActivate）が 1 回たたく、「商品名を調べる」のアクションが 2 回たたく、ダブルタップ長押しが長押し。
 */
final class TouchSurface: UIView {
    var onTap: (() -> Void)?
    var onDoubleTap: (() -> Void)?
    var onLongPress: (() -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        let double = UITapGestureRecognizer(target: self, action: #selector(doubleTapped))
        double.numberOfTapsRequired = 2
        addGestureRecognizer(double)
        let single = UITapGestureRecognizer(target: self, action: #selector(tapped))
        single.require(toFail: double)
        addGestureRecognizer(single)
        let long = UILongPressGestureRecognizer(target: self, action: #selector(longPressed(_:)))
        long.minimumPressDuration = 0.5
        addGestureRecognizer(long)
        isAccessibilityElement = true
        accessibilityLabel = "1回たたくと棚の案内。2回たたくと商品の名前。長押しして質問"
        accessibilityTraits = [.button]
        accessibilityCustomActions = [UIAccessibilityCustomAction(name: "商品名を調べる") { [weak self] _ in
            self?.onDoubleTap?()
            return true
        }]
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    @objc private func tapped() { onTap?() }
    @objc private func doubleTapped() { onDoubleTap?() }

    @objc private func longPressed(_ g: UILongPressGestureRecognizer) {
        if g.state == .began { onLongPress?() }
    }

    override func accessibilityActivate() -> Bool {
        onTap?()
        return true
    }
}
