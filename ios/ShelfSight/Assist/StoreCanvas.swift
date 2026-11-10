import UIKit
import simd

/// 店のモデルの画面で表示を切り替えられる項目。color は描画の色（RGB）で、凡例にも使う。
enum StoreLayer: CaseIterable {
    case shelf, glass, floor, point, sight, category, genre, person, phone, mirror, ransac, saved

    var label: String {
        switch self {
        case .shelf: return "棚"
        case .glass: return "ガラス棚"
        case .floor: return "床"
        case .point: return "商品の点"
        case .sight: return "視線"
        case .category: return "商品カテゴリ"
        case .genre: return "棚のジャンル"
        case .person: return "人"
        case .phone: return "スマホ"
        case .mirror: return "ガラス補正"
        case .ransac: return "RANSAC"
        case .saved: return "保存マップ"
        }
    }

    var color: Int {
        switch self {
        case .shelf: return 0xc8ccd2
        case .glass: return 0x4fd1c5
        case .floor: return 0x565d68
        case .point: return 0xff9800
        case .sight: return 0x63e6ff
        case .category: return 0xffffff
        case .genre: return 0x1b6fd0
        case .person: return 0x5ad1e6
        case .phone: return 0xf0efec
        case .mirror: return 0xff8c00
        case .ransac: return 0xff4fd8
        case .saved: return 0xb388ff
        }
    }

    var topViewOnly: Bool { self == .ransac }
}

func uiColor(_ hex: Int, alpha: CGFloat = 1) -> UIColor {
    UIColor(red: CGFloat(hex >> 16 & 255) / 255, green: CGFloat(hex >> 8 & 255) / 255, blue: CGFloat(hex & 255) / 255, alpha: alpha)
}

/// 地図を見るための視点。タッチ操作はメインスレッドで使う。
final class OrbitCamera {
    private static let defaultPitch: Float = 1.1   // 約 63 度。店内を斜め上から見下ろす
    private static let minPitch: Float = 0.05
    private static let maxPitch: Float = 1.55
    private static let minDistance: Float = 0.5
    private static let minFitDistance: Float = 4
    private static let maxDistance: Float = 200
    private static let radiansPerPoint: Float = 0.015
    private static let metersPerPointAtUnitDistance: Float = 0.004

    private var yaw: Float = 0
    private var pitch = OrbitCamera.defaultPitch
    private var distance: Float = 10
    private(set) var target = Vec3.zero

    func fit(_ center: Vec3, radius: Float) {
        target = center
        distance = (radius * 2.2).clamped(Self.minFitDistance, Self.maxDistance)
        pitch = Self.defaultPitch
    }

    func setView(yaw: Float, pitch: Float) { self.yaw = yaw; self.pitch = pitch }

    func rotate(_ dx: Float, _ dy: Float) {
        yaw -= dx * Self.radiansPerPoint
        pitch = (pitch + dy * Self.radiansPerPoint).clamped(Self.minPitch, Self.maxPitch)
    }

    func zoom(_ factor: Float) {
        if factor > 0 { distance = (distance / factor).clamped(Self.minDistance, Self.maxDistance) }
    }

    /// 指の動きに地図が付いてくるよう、注視点を床と平行に動かす。
    func pan(_ dx: Float, _ dy: Float) {
        let scale = distance * Self.metersPerPointAtUnitDistance
        let rightX = cos(yaw), rightZ = -sin(yaw), forwardX = -sin(yaw), forwardZ = -cos(yaw)
        target = Vec3(target.x - rightX * dx * scale + forwardX * dy * scale, target.y, target.z - rightZ * dx * scale + forwardZ * dy * scale)
    }

    func eye() -> Vec3 {
        Vec3(target.x + distance * cos(pitch) * sin(yaw), target.y + distance * sin(pitch), target.z + distance * cos(pitch) * cos(yaw))
    }
}

/// 立体音響のテストの音源。同じ場所で chimes 回鳴らし、次の場所へゆっくり移る、を繰り返す。
enum SpatialDemo {
    static let chimes = 5
    private static let chimeS = 0.7
    private static let holdS = Double(chimes) * chimeS
    private static let moveS = 3.0
    private static let cycleS = holdS + moveS
    /// 回る場所（右が正の方位 °、上が正の仰角 °、距離 m）。隣どうしは少しずつ離し、最初は正面にする。
    private static let spots: [(Double, Double, Double)] = [
        (0, 0, 2), (40, 25, 1.5), (100, -20, 2.5), (170, 10, 1.5), (-120, 35, 3), (-50, -25, 1), (-10, 40, 2.5), (20, -30, 3.5),
    ]

    /// 今鳴らす番号（鳴らす回ごとに変わる）。移動中は nil。
    static func chimeSlot(_ seconds: Double) -> Int? {
        let cycle = floor(seconds / cycleS)
        let phase = seconds - cycle * cycleS
        return phase < holdS ? Int(cycle) * chimes + Int(phase / chimeS) : nil
    }

    /// 最初に向いていた方向を基準にした音源の位置（x が右、y が上、z が正面、m）。
    static func offset(_ seconds: Double) -> Vec3 {
        let cycle = floor(seconds / cycleS)
        let phase = seconds - cycle * cycleS
        let from = spots[Int(cycle) % spots.count], to = spots[(Int(cycle) + 1) % spots.count]
        let s = ((phase - holdS) / moveS).clamped(0, 1)
        let t = s * s * (3 - 2 * s)   // 動き始めと止まる前をゆっくりに
        let turn = (to.0 - from.0 + 540).truncatingRemainder(dividingBy: 360) - 180   // 近い方の回りで移る
        let azimuth = (from.0 + turn * t) * .pi / 180
        let elevation = (from.1 + (to.1 - from.1) * t) * .pi / 180
        let distance = from.2 + (to.2 - from.2) * t
        let horizontal = distance * cos(elevation)
        return Vec3(Float(horizontal * sin(azimuth)), Float(distance * sin(elevation)), Float(horizontal * cos(azimuth)))
    }

    /// origin の場所と向きを基準にした世界座標。
    static func world(_ origin: Listener, _ seconds: Double) -> Vec3 {
        let o = offset(seconds), p = origin.position
        return Vec3(p.x - origin.forwardZ * o.x + origin.forwardX * o.z, p.y + o.y, p.z + origin.forwardX * o.x + origin.forwardZ * o.z)
    }
}

/**
 * 推定した店（StoreView）を線画で、認識した商品を点で、自分を向き付きのスマホで描く（Android 版の StoreModelRenderer と StoreLabelsView）。
 * 線画だけで面数が少ないので、毎回 Core Graphics で描く。
 */
final class StoreCanvas: UIView {
    struct ModelState {
        let view: StoreView?
        let saved: StoreView?
        /// 端末の姿勢（画面の右 +x、上 +y、カメラの向き -z）。
        let phone: simd_float4x4?
        let floorY: Float
        let items: [MapItem]
        let pending: [Vec3]
        let people: [PersonMarker]
        let topView: Bool
        let layers: Set<StoreLayer>
    }

    struct DemoState {
        let listener: Listener
        let source: Vec3
        let sounding: Bool
    }

    enum Content {
        case model(ModelState)
        case demo(DemoState)
    }

    private struct Pen: Hashable {
        let color: Int
        let width: CGFloat
    }

    private struct Label {
        let at: CGPoint
        let text: String
        let shelf: Bool
    }

    private static let background = uiColor(0x14171c)
    private static let line: CGFloat = 1.2
    private static let thin: CGFloat = 0.6
    private static let ransacLine: CGFloat = 3
    private static let pointSize: CGFloat = 4
    private static let nameRangeM: Float = 3
    private static let shelfLabelLiftM: Float = 0.1
    private static let shelfLabelReach: Float = 1.6
    private static let personLabelY: Float = 1.9
    private static let headY: Float = 1.6
    private static let headR: Float = 0.12
    private static let pendingColor = 0x6e4a12
    private static let rawColor = 0xffd9a8
    private static let arrowColor = 0xff7a45
    private static let approachingColor = 0xff4d4d
    private static let meColor = StoreLayer.phone.color
    private static let movingColor = 0xaeb4bd
    private static let mirrorPointColor = 0xff5a36
    private static let body: [[Float]] = [
        [0, 1.18, 0, 0.42, 0.62, 0.22], [-0.3, 1.16, 0, 0.1, 0.62, 0.12], [0.3, 1.16, 0, 0.1, 0.62, 0.12],
        [-0.11, 0.43, 0, 0.15, 0.86, 0.16], [0.11, 0.43, 0, 0.15, 0.86, 0.16], [0, 1.6, -0.15, 0.04, 0.05, 0.08],
    ]

    let orbit: OrbitCamera
    var content: (() -> Content?)?
    /// 上のタブの下端。棚のジャンルがタブに隠れないようにする。
    var topInset: CGFloat = 0
    /// 下に重ねる文章の高さ。棚のジャンルが隠れないようにする。
    var bottomInset: CGFloat = 0

    private var link: CADisplayLink?
    private var viewProjection = matrix_identity_float4x4
    private var paths: [Pen: CGMutablePath] = [:]
    private var dots: [(color: Int, size: CGFloat, path: CGMutablePath)] = []

    init(orbit: OrbitCamera) {
        self.orbit = orbit
        super.init(frame: .zero)
        isOpaque = true
        backgroundColor = Self.background
        contentMode = .redraw
        isAccessibilityElement = true
        let pan = UIPanGestureRecognizer(target: self, action: #selector(panned(_:)))
        pan.maximumNumberOfTouches = 2
        addGestureRecognizer(pan)
        addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinched(_:))))
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        link?.invalidate()
        link = nil
        guard window != nil else { return }
        let l = CADisplayLink(target: self, selector: #selector(tick))
        l.preferredFramesPerSecond = 15
        l.add(to: .main, forMode: .common)
        link = l
    }

    @objc private func tick() { if !isHidden { setNeedsDisplay() } }

    @objc private func panned(_ g: UIPanGestureRecognizer) {
        let t = g.translation(in: self)
        g.setTranslation(.zero, in: self)
        if g.numberOfTouches >= 2 { orbit.pan(Float(t.x), Float(t.y)) } else { orbit.rotate(Float(t.x), Float(t.y)) }
    }

    @objc private func pinched(_ g: UIPinchGestureRecognizer) {
        orbit.zoom(Float(g.scale))
        g.scale = 1
    }

    // ------------------------------------------------------------ 射影

    private static func simdOf(_ v: Vec3) -> SIMD3<Float> { SIMD3(v.x, v.y, v.z) }

    private static func lookAt(_ eye: SIMD3<Float>, _ center: SIMD3<Float>, _ up: SIMD3<Float>) -> simd_float4x4 {
        let f = normalize(center - eye), s = normalize(cross(f, up)), u = cross(s, f)
        return simd_float4x4(SIMD4(s.x, u.x, -f.x, 0), SIMD4(s.y, u.y, -f.y, 0), SIMD4(s.z, u.z, -f.z, 0), SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1))
    }

    private static func perspective(_ fovDeg: Float, _ aspect: Float, _ near: Float, _ far: Float) -> simd_float4x4 {
        let f = 1 / tan(fovDeg * .pi / 360)
        return simd_float4x4(SIMD4(f / aspect, 0, 0, 0), SIMD4(0, f, 0, 0), SIMD4(0, 0, (far + near) / (near - far), -1), SIMD4(0, 0, 2 * far * near / (near - far), 0))
    }

    private static func ortho(_ halfW: Float, _ halfH: Float, _ near: Float, _ far: Float) -> simd_float4x4 {
        simd_float4x4(SIMD4(1 / halfW, 0, 0, 0), SIMD4(0, 1 / halfH, 0, 0), SIMD4(0, 0, -2 / (far - near), 0), SIMD4(0, 0, -(far + near) / (far - near), 1))
    }

    /// 視点を決める。@return 視点の位置
    private func begin(_ fovDeg: Float, topView: Bool) -> Vec3 {
        let aspect = Float(bounds.width / max(bounds.height, 1))
        var e = orbit.eye()
        let c = orbit.target
        if topView {   // 真上からの正射影。ズームは視点の距離、回転は視点の水平の向きをそのまま使う
            let distance = e.distanceTo(c)
            let half = distance * tan(fovDeg * .pi / 360)
            let flat = max(hypot(c.x - e.x, c.z - e.z), 1e-4)
            let up = SIMD3((c.x - e.x) / flat, 0, (c.z - e.z) / flat)
            viewProjection = Self.ortho(half * aspect, half, -200, 200) * Self.lookAt(SIMD3(c.x, c.y + distance, c.z), Self.simdOf(c), up)
            e = Vec3(c.x, c.y + distance, c.z)
        } else {
            viewProjection = Self.perspective(fovDeg, aspect, 0.1, 200) * Self.lookAt(Self.simdOf(e), Self.simdOf(c), SIMD3(0, 1, 0))
        }
        return e
    }

    /// 世界座標の点の画面上の位置。視点の後ろや、画面の reach 倍の範囲より外なら nil。
    private func screen(_ p: SIMD3<Float>, reach: Float = .greatestFiniteMagnitude) -> CGPoint? {
        let c = viewProjection * SIMD4(p.x, p.y, p.z, 1)
        guard c.w > 0.01 else { return nil }
        let sx = c.x / c.w, sy = c.y / c.w
        guard abs(sx) <= reach, abs(sy) <= reach else { return nil }
        return CGPoint(x: CGFloat((sx + 1) / 2) * bounds.width, y: CGFloat((1 - sy) / 2) * bounds.height)
    }

    // ------------------------------------------------------------ 線と点をためる

    private func line(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ color: Int, _ width: CGFloat = StoreCanvas.line) {
        guard let p = screen(a), let q = screen(b) else { return }
        let pen = Pen(color: color, width: width)
        let path = paths[pen] ?? CGMutablePath()
        path.move(to: p)
        path.addLine(to: q)
        paths[pen] = path
    }

    /// 中心が原点の単位立方体の 12 本の辺を、transform で置いて描く。
    private func cube(_ color: Int, _ transform: (SIMD3<Float>) -> SIMD3<Float>) {
        let ends: [Float] = [-0.5, 0.5]
        for a in ends {
            for b in ends {
                line(transform(SIMD3(-0.5, a, b)), transform(SIMD3(0.5, a, b)), color)
                line(transform(SIMD3(a, -0.5, b)), transform(SIMD3(a, 0.5, b)), color)
                line(transform(SIMD3(a, b, -0.5)), transform(SIMD3(a, b, 0.5)), color)
            }
        }
    }

    private func box(_ yaw: Double, _ u0: Double, _ v0: Double, _ u1: Double, _ v1: Double, _ y: Float, _ height: Float, _ color: Int) {
        let cs = cos(yaw), sn = sin(yaw)
        cube(color) { p in
            let u = (u0 + u1) / 2 + Double(p.x) * (u1 - u0), v = (v0 + v1) / 2 + Double(p.z) * (v1 - v0)
            return SIMD3(Float(u * cs - v * sn), y + height / 2 + p.y * height, Float(u * sn + v * cs))
        }
    }

    private func moduleBoxes(_ yaw: Double, _ f: StoreFixture, _ floorY: Float, _ color: Int) {
        let lo = f.c - f.d / 2, hi = f.c + f.d / 2
        for i in 0..<f.n {
            let a = f.moduleStart(i), b = f.moduleEnd(i)
            if f.axis == 0 { box(yaw, lo, a, hi, b, floorY, Float(f.h), color) } else { box(yaw, a, lo, b, hi, floorY, Float(f.h), color) }
        }
    }

    private func dotsOf(_ points: [SIMD3<Float>], _ color: Int, _ size: CGFloat = StoreCanvas.pointSize) {
        let path = CGMutablePath()
        for p in points {
            if let at = screen(p) { path.addEllipse(in: CGRect(x: at.x - size / 2, y: at.y - size / 2, width: size, height: size)) }
        }
        dots.append((color, size, path))
    }

    private func ring(_ center: SIMD3<Float>, _ radius: Float, _ color: Int, plane: Int) {
        let n = 32
        func at(_ k: Int) -> SIMD3<Float> {
            let a = 2 * Float.pi * Float(k) / Float(n)
            let c = cos(a) * radius, s = sin(a) * radius
            switch plane {
            case 0: return center + SIMD3(c, 0, s)
            case 1: return center + SIMD3(c, s, 0)
            default: return center + SIMD3(0, c, s)
            }
        }
        for k in 0..<n { line(at(k), at(k + 1), color) }
    }

    /// 足元が (x, floorY, z) の人の線画。
    private func person(_ x: Float, _ floorY: Float, _ z: Float, _ color: Int) {
        for plane in 0...2 { ring(SIMD3(x, floorY + Self.headY, z), Self.headR, color, plane: plane) }
        for b in Self.body {
            cube(color) { p in SIMD3(x + b[0] + p.x * b[3], floorY + b[1] + p.y * b[4], z + b[2] + p.z * b[5]) }
        }
    }

    /// 実際の位置と傾きで浮かぶスマホと、カメラの向きの矢印。店の中で見えるよう、実物より大きく描く。
    private func phone(_ pose: simd_float4x4) {
        func world(_ p: SIMD3<Float>) -> SIMD3<Float> {
            let w = pose * SIMD4(p.x, p.y, p.z, 1)
            return SIMD3(w.x, w.y, w.z)
        }
        cube(Self.meColor) { world(SIMD3($0.x * 0.24, $0.y * 0.5, $0.z * 0.04)) }
        let tip: Float = -1, back: Float = -0.82, h: Float = 0.108
        line(world(.zero), world(SIMD3(0, 0, tip)), Self.arrowColor)
        for side in [SIMD3<Float>(h, 0, back), SIMD3(-h, 0, back), SIMD3(0, h, back), SIMD3(0, -h, back)] {
            line(world(SIMD3(0, 0, tip)), world(side), Self.arrowColor)
        }
    }

    // ------------------------------------------------------------ 描く

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.setFillColor(Self.background.cgColor)
        ctx.fill(bounds)
        paths = [:]
        dots = []
        guard bounds.width > 0, bounds.height > 0, let content = content?() else { return }
        var labels: [Label] = []
        var caption: String?
        switch content {
        case .model(let s): labels = drawModel(s)
        case .demo(let s): caption = drawDemo(s)
        }
        ctx.setLineCap(.round)
        for (pen, path) in paths {
            ctx.setStrokeColor(uiColor(pen.color).cgColor)
            ctx.setLineWidth(pen.width)
            ctx.addPath(path)
            ctx.strokePath()
        }
        for d in dots {
            ctx.setFillColor(uiColor(d.color).cgColor)
            ctx.addPath(d.path)
            ctx.fillPath()
        }
        drawLabels(ctx, labels)
        if let caption { drawCaption(ctx, caption) }
    }

    private func drawModel(_ s: ModelState) -> [Label] {
        let eye = begin(50, topView: s.topView)
        let layers = s.layers
        if let view = s.view, let r = view.room {
            if layers.contains(.floor) { box(view.yaw, r[0], r[1], r[2], r[3], s.floorY, 0, StoreLayer.floor.color) }
            for f in view.fixtures where layers.contains(f.kind == "glass" ? .glass : .shelf) {
                moduleBoxes(view.yaw, f, s.floorY, f.kind == "glass" ? StoreLayer.glass.color : StoreLayer.shelf.color)
            }
        }
        // shortcut: 保存した地図は前回の ARKit の座標のままで今の場所と合っていない。店の点との位置合わせを入れたら案内にも使う
        if let saved = s.saved, layers.contains(.saved) {
            for f in saved.fixtures { moduleBoxes(saved.yaw, f, s.floorY, StoreLayer.saved.color) }
        }
        if let pose = s.phone, layers.contains(.phone) { phone(pose) }
        if layers.contains(.person) {
            for p in s.people { person(p.position.x, s.floorY, p.position.z, p.motion == .approaching ? Self.approachingColor : StoreLayer.person.color) }
        }
        if layers.contains(.sight) {
            for item in s.items { if let camera = item.cameraAt { sightArrow(item.position, camera) } }
        }
        let correcting = layers.contains(.mirror)
        let restored = s.items.map { correcting ? GlassProjection.corrected($0, s.view) : nil }
        for (item, r) in zip(s.items, restored) {
            if let r { line(Self.simdOf(item.position), Self.simdOf(r), StoreLayer.mirror.color, Self.thin) }
        }
        if s.topView && correcting {
            for m in s.view?.mirror ?? [] {
                line(SIMD3(Float(m[0]), s.floorY, Float(m[1])), SIMD3(Float(m[2]), s.floorY, Float(m[3])), StoreLayer.mirror.color, Self.thin)
            }
        }
        if s.topView && layers.contains(.ransac) {
            for l in s.view?.ransac ?? [] {
                line(SIMD3(Float(l[0]), s.floorY, Float(l[1])), SIMD3(Float(l[2]), s.floorY, Float(l[3])), StoreLayer.ransac.color, Self.ransacLine)
            }
        }
        if layers.contains(.point) { dotsOf(s.pending.map(Self.simdOf), Self.pendingColor) }
        if layers.contains(.point) {
            dotsOf(zip(s.items, restored).filter { $0.1 == nil }.map { Self.simdOf($0.0.position) }, StoreLayer.point.color)
        }
        dotsOf(zip(s.items, restored).filter { $0.1 != nil }.map { Self.simdOf($0.0.position) }, Self.rawColor)
        dotsOf(restored.compactMap { $0 }.map(Self.simdOf), StoreLayer.point.color)
        if s.topView && correcting {
            let moved = s.view?.mirror ?? []
            dotsOf(moved.map { SIMD3(Float($0[0]), s.floorY, Float($0[1])) }, Self.rawColor)
            dotsOf(moved.map { SIMD3(Float($0[2]), s.floorY, Float($0[3])) }, Self.mirrorPointColor)
        }
        return labelsOf(s, eye)
    }

    /// 商品を見た場所から点まで、細い線で視線を描く。
    private func sightArrow(_ point: Vec3, _ camera: Vec3) {
        let dx = point.x - camera.x, dz = point.z - camera.z
        let length = max(hypot(dx, dz), 1e-4)
        let fx = dx / length, fz = dz / length
        let backX = point.x - fx * 0.1, backZ = point.z - fz * 0.1
        let sideX = -fz * 0.07, sideZ = fx * 0.07
        let tip = Self.simdOf(point), color = StoreLayer.sight.color
        line(Self.simdOf(camera), tip, color, Self.thin)
        line(tip, SIMD3(backX + sideX, point.y, backZ + sideZ), color, Self.thin)
        line(tip, SIMD3(backX - sideX, point.y, backZ - sideZ), color, Self.thin)
    }

    private func labelsOf(_ s: ModelState, _ eye: Vec3) -> [Label] {
        var out: [Label] = []
        if s.layers.contains(.person) {
            for p in s.people {
                guard let at = screen(SIMD3(p.position.x, s.floorY + Self.personLabelY, p.position.z), reach: Self.shelfLabelReach) else { continue }
                let metres = s.phone.map { String(format: "%.1fm", hypot(p.position.x - $0.columns.3.x, p.position.z - $0.columns.3.z)) } ?? "?"
                out.append(Label(at: at, text: "人 \(metres) \(p.motion.labelJa)", shelf: true))
            }
        }
        if s.layers.contains(.genre), let view = s.view, view.room != nil {
            let cs = cos(view.yaw), sn = sin(view.yaw)
            for l in StoreLabels.shelves(view) {
                let p = SIMD3(Float(l.u * cs - l.v * sn), s.floorY + Float(l.top) + Self.shelfLabelLiftM, Float(l.u * sn + l.v * cs))
                if let at = screen(p, reach: Self.shelfLabelReach) { out.append(Label(at: at, text: l.text, shelf: true)) }
            }
        }
        // 近づいた点だけ名前を出す
        if s.layers.contains(.category) {
            for item in s.items.filter({ $0.position.distanceTo(eye) < Self.nameRangeM }).stableSorted(by: { $0.seen > $1.seen }) {
                if let at = screen(Self.simdOf(item.position), reach: 1) {
                    out.append(Label(at: CGPoint(x: at.x, y: at.y - Self.pointSize / 2), text: item.name, shelf: false))
                }
            }
        }
        return out
    }

    private func drawLabels(_ ctx: CGContext, _ labels: [Label]) {
        var taken: [CGRect] = []
        let pad: CGFloat = 3
        let shelfFont = UIFont.boldSystemFont(ofSize: 14), itemFont = UIFont.systemFont(ofSize: 11)
        for l in labels {
            let font = l.shelf ? shelfFont : itemFont
            let text = NSAttributedString(string: l.text, attributes: [.font: font, .foregroundColor: UIColor.white])
            let size = text.size()
            var box = CGRect(x: l.at.x - size.width / 2 - pad, y: l.at.y - size.height - 2 * pad, width: size.width + 2 * pad, height: size.height + 2 * pad)   // 点や棚のすぐ上に出す
            if l.shelf {
                box.origin.x += box.minX < 0 ? -box.minX : min(0, bounds.width - box.maxX)
                box.origin.y += box.minY < topInset ? topInset - box.minY : min(0, bounds.height - bottomInset - box.maxY)
            }
            if taken.contains(where: { $0.intersects(box) }) { continue }   // 重なる文字は読めないので描かない
            taken.append(box)
            ctx.setFillColor(l.shelf ? uiColor(0x1b6fd0, alpha: 0.8).cgColor : uiColor(0, alpha: 0.69).cgColor)
            ctx.addPath(UIBezierPath(roundedRect: box, cornerRadius: pad).cgPath)
            ctx.fillPath()
            text.draw(at: CGPoint(x: box.minX + pad, y: box.minY + pad))
        }
    }

    private func drawCaption(_ ctx: CGContext, _ caption: String) {
        let style = NSMutableParagraphStyle()
        style.alignment = .center
        let text = NSAttributedString(string: caption, attributes: [.font: UIFont.systemFont(ofSize: 20), .foregroundColor: UIColor.white, .paragraphStyle: style])
        let bottom = bounds.height - 40
        let strip = CGRect(x: 0, y: bottom - 44, width: bounds.width, height: 44)
        ctx.setFillColor(uiColor(0, alpha: 0.69).cgColor)
        ctx.fill(strip)
        text.draw(in: strip.insetBy(dx: 0, dy: 9))
    }

    /// 立体音響のテスト: 正面（-z）を向いて立つ人の線画と、その人から見た音源の位置を描く。@return 画面の下に出す文字
    private func drawDemo(_ s: DemoState) -> String {
        _ = begin(70, topView: false)
        for r in 1...4 { ring(.zero, Float(r), StoreLayer.floor.color, plane: 0) }
        person(0, 0, 0, Self.meColor)
        let l = s.listener
        let az = Binaural.azimuth(l, s.source)
        let horizontal = Binaural.horizontalDistance(l, s.source)
        let rise = s.source.y - l.position.y
        let p = SIMD3(horizontal * sin(az), Self.headY + rise, -horizontal * cos(az))
        let color = s.sounding ? StoreLayer.point.color : Self.movingColor
        line(SIMD3(p.x, 0, p.z), p, color)   // 床からの線で高さを見せる
        dotsOf([p], color, 15)
        let deg = Int((az * 180 / .pi).rounded()), up = Int((atan2(rise, horizontal) * 180 / .pi).rounded())
        let distance = (horizontal * horizontal + rise * rise).squareRoot()
        return "\(deg >= 0 ? "右" : "左") \(abs(deg))°   \(up >= 0 ? "上" : "下") \(abs(up))°   \(String(format: "%.1f", distance))m"
    }
}

/// 項目ごとのスイッチを縦に並べた凡例。文字の色が描画の色になる。
final class StoreLegend: UIStackView {
    private(set) var shown = Set(StoreLayer.allCases)
    private let onChange: (Set<StoreLayer>) -> Void
    private var topViewRows: [UIButton] = []
    private var layerOf: [UIButton: StoreLayer] = [:]

    init(onChange: @escaping (Set<StoreLayer>) -> Void) {
        self.onChange = onChange
        super.init(frame: .zero)
        axis = .vertical
        alignment = .leading
        backgroundColor = uiColor(0, alpha: 0.63)
        for layer in StoreLayer.allCases {
            let row = UIButton(type: .system)
            row.contentHorizontalAlignment = .left
            row.titleLabel?.font = .systemFont(ofSize: 13)
            row.setTitleColor(uiColor(layer.color), for: .normal)
            row.addTarget(self, action: #selector(toggled(_:)), for: .touchUpInside)
            layerOf[row] = layer
            update(row, layer)
            if layer.topViewOnly {
                row.isHidden = true
                topViewRows.append(row)
            }
            addArrangedSubview(row)
        }
    }

    required init(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    private func update(_ row: UIButton, _ layer: StoreLayer) {
        let on = shown.contains(layer)
        row.setTitle("\(on ? "☑" : "☐") \(layer.label)", for: .normal)
        row.accessibilityLabel = layer.label
        row.accessibilityValue = on ? "オン" : "オフ"
    }

    @objc private func toggled(_ row: UIButton) {
        guard let layer = layerOf[row] else { return }
        if shown.contains(layer) { shown.remove(layer) } else { shown.insert(layer) }
        update(row, layer)
        onChange(shown)
    }

    /// RANSAC のように 2D でだけ描く項目の行を出し入れする。
    func showTopViewRows(_ show: Bool) {
        for row in topViewRows { row.isHidden = !show }
    }
}
