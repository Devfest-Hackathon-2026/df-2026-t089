import UIKit

/// 画面に出す内部状態。各キューが書き、DebugOverlay が読むだけ。
final class AssistDebug {
    struct PersonView {
        let box: NormalizedBox
        let ownHand: Bool
        let label: String
    }

    // カメラ・向き
    @Locked var tracking = "準備中"
    @Locked var headingDeg: Float? = nil
    /// センサー画像の正規化座標 (0,0), (1,0), (0,1) が画面のどこに来るか（画面の正規化座標）。
    @Locked var corners: [Float]? = nil
    @Locked var rotation = FrameCapture.portraitRotation

    // ① 写っているもの
    @Locked var scene = Scene.none
    @Locked var productBoxes: [[Float]] = []
    /// 最後の商品の解析で名前が付いた商品。
    @Locked var products: [RecognizedProduct] = []
    @Locked var sceneMs = ""
    @Locked var sceneAt: Int64 = 0

    /// カメラの動きの速さと、省いた処理の数。
    @Locked var motion = ""
    @Locked var skipInfo = ""

    // ② 棚の説明
    @Locked var kept: [RecognizedProduct] = []
    @Locked var removed: [RecognizedProduct] = []
    @Locked var unnamed = 0
    @Locked var mainGroup: String? = nil
    @Locked var description = ""
    @Locked var describeInfo = ""
    /// 説明した棚から別の棚に移ったかの判定（ShelfChange）と、説明した場所からの距離。
    @Locked var shelfChange = ""
    @Locked var productsAt: Int64 = 0

    // ③ 長押し
    @Locked var longPress = ""
    @Locked var focusBox: [Float]? = nil
    @Locked var focusAt: Int64 = 0
    @Locked var lookup = ""
    /// 商品の枠ごとのスマホからの距離（productBoxes と同じ順、分からなければ nil）と、1 つだけ特に近い商品（NearProduct）。
    @Locked var boxDistances: [Float?] = []
    @Locked var nearBox: [Float]? = nil
    @Locked var near = ""
    @Locked var nearDistance: Float? = nil

    // ④ 人
    @Locked var people: [PersonView] = []
    @Locked var peopleInfo = ""
    @Locked var peopleAt: Int64 = 0

    // 声と設定
    @Locked var voice = ""
    @Locked var settings = ""
}

/**
 * カメラの映像の上に、認識の状況をすべて描く（Android 版と同じ内容）。
 * 映像の上: 商品の枠（緑＝説明に使った、橙＝場違いとして外した、灰＝最後の解析の枠）、
 * 長押しで調べた範囲（水色）、人の枠（赤＝人、紫の点線＝自分の手として外した）。下: 文字の一覧。
 */
final class DebugOverlay: UIView {
    private static let staleMs: Int64 = 3000
    private static let productsMs: Int64 = 8000
    /// 近い商品の帯を出し続ける長さ（最後の解析から）。
    private static let nearBannerMs: Int64 = 1500
    private static let gray = UIColor(white: 0.62, alpha: 1)
    private static let green = UIColor(red: 0.30, green: 0.69, blue: 0.31, alpha: 1)
    private static let orange = UIColor(red: 1, green: 0.6, blue: 0, alpha: 1)
    private static let cyan = UIColor(red: 0, green: 0.9, blue: 1, alpha: 1)
    private static let red = UIColor(red: 0.96, green: 0.26, blue: 0.21, alpha: 1)
    private static let purple = UIColor(red: 0.88, green: 0.25, blue: 0.98, alpha: 1)

    private let d: AssistDebug
    private var timer: Timer?
    private let textFont = UIFont.systemFont(ofSize: 11)
    private let smallFont = UIFont.systemFont(ofSize: 9)
    private let bannerFont = UIFont.boldSystemFont(ofSize: 26)

    init(debug: AssistDebug) {
        d = debug
        super.init(frame: .zero)
        isOpaque = false
        backgroundColor = .clear
        isUserInteractionEnabled = false
        isAccessibilityElement = false
        accessibilityElementsHidden = true
        timer = Timer.scheduledTimer(timeInterval: 0.1, target: self, selector: #selector(refresh), userInfo: nil, repeats: true)
    }

    @objc private func refresh() { setNeedsDisplay() }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        let now = nowMs()
        let corners = d.corners
        drawImageLayer(ctx, now, corners)
        drawNearBanner(ctx, now, safeAreaInsets.top)
        drawPanel(ctx, safeAreaInsets.top, now)
    }

    /// 正立画像の正規化座標 → 画面のポイント。
    private func toView(_ u: Float, _ v: Float, _ c: [Float]?) -> CGPoint? {
        guard let c, c.count == 6 else { return nil }
        let (sx, sy) = Unprojector.uprightToSensor(u, v, d.rotation)
        let x = c[0] + (c[2] - c[0]) * sx + (c[4] - c[0]) * sy
        let y = c[1] + (c[3] - c[1]) * sx + (c[5] - c[1]) * sy
        return CGPoint(x: CGFloat(x) * bounds.width, y: CGFloat(y) * bounds.height)
    }

    private func text(_ s: String, at p: CGPoint, font: UIFont, color: UIColor = .white) {
        (s as NSString).draw(at: p, withAttributes: [.font: font, .foregroundColor: color])
    }

    private func width(_ s: String, _ font: UIFont) -> CGFloat {
        (s as NSString).size(withAttributes: [.font: font]).width
    }

    private func rect(_ ctx: CGContext, _ b: [Float], _ c: [Float]?, _ color: UIColor, label: String? = nil, dash: Bool = false, lineWidth: CGFloat = 2) {
        let pts = [toView(b[0], b[1], c), toView(b[2], b[1], c), toView(b[2], b[3], c), toView(b[0], b[3], c)].compactMap { $0 }
        guard pts.count == 4 else { return }
        let r = CGRect(
            x: pts.map(\.x).min()!, y: pts.map(\.y).min()!,
            width: pts.map(\.x).max()! - pts.map(\.x).min()!, height: pts.map(\.y).max()! - pts.map(\.y).min()!
        )
        ctx.setStrokeColor(color.cgColor)
        ctx.setLineWidth(lineWidth)
        ctx.setLineDash(phase: 0, lengths: dash ? [8, 6] : [])
        ctx.stroke(r)
        ctx.setLineDash(phase: 0, lengths: [])
        if let label {
            let h = smallFont.lineHeight
            ctx.setFillColor(color.withAlphaComponent(0.7).cgColor)
            ctx.fill(CGRect(x: r.minX, y: r.minY - h - 1, width: width(label, smallFont) + 4, height: h + 1))
            text(label, at: CGPoint(x: r.minX + 2, y: r.minY - h), font: smallFont)
        }
    }

    private func box(_ p: RecognizedProduct) -> [Float] { [p.left, p.top, p.right, p.bottom] }
    private func box(_ b: NormalizedBox) -> [Float] { [b.left, b.top, b.right, b.bottom] }

    private func drawImageLayer(_ ctx: CGContext, _ now: Int64, _ c: [Float]?) {
        if now - d.sceneAt < Self.staleMs {
            // 商品の枠（最後の解析）と、名前が付いた商品
            for b in d.productBoxes { rect(ctx, b, c, Self.gray) }
            for p in d.products { rect(ctx, box(p), c, .white, label: short(p.name)) }
        }
        // 説明に使った商品と、外した商品
        if now - d.productsAt < Self.productsMs {
            for p in d.kept { rect(ctx, box(p), c, Self.green, label: short(p.name)) }
            for p in d.removed { rect(ctx, box(p), c, Self.orange, label: "×" + short(p.name)) }
        }
        if now - d.sceneAt < Self.staleMs {
            // 枠ごとのスマホからの距離（枠の右下）と、1 つだけ特に近い商品（太い緑）
            let boxes = d.productBoxes
            let dist = d.boxDistances
            if dist.count == boxes.count {
                for (i, b) in boxes.enumerated() {
                    guard let m = dist[i], let p = toView(b[2], b[3], c) else { continue }
                    let label = String(format: "%.2fm", m)
                    text(label, at: CGPoint(x: p.x - width("0.00m", smallFont) - 2, y: p.y - smallFont.lineHeight - 1), font: smallFont)
                }
            }
            if let b = d.nearBox { rect(ctx, b, c, Self.green, label: "近い商品", lineWidth: 4) }
        }
        // 長押しで調べた範囲
        if now - d.focusAt < Self.productsMs, let b = d.focusBox {
            rect(ctx, b, c, Self.cyan, label: "調べた範囲", lineWidth: 4)
        }
        // 人
        if now - d.peopleAt < Self.staleMs {
            for p in d.people { rect(ctx, box(p.box), c, p.ownHand ? Self.purple : Self.red, label: p.label, dash: p.ownHand) }
        }
    }

    /// 1 つだけ特に近い商品が見えている間、画面の上に大きな緑の帯を出す（読み取れたことがひと目で分かるように）。
    private func drawNearBanner(_ ctx: CGContext, _ now: Int64, _ top: CGFloat) {
        if now - d.sceneAt >= Self.nearBannerMs || d.nearBox == nil { return }
        let label = "● 近い商品あり  " + (d.nearDistance.map { String(format: "%.2fm", $0) } ?? "")
        let h = bannerFont.pointSize * 1.6
        let y = top + 8
        ctx.setFillColor(UIColor(red: 0.30, green: 0.69, blue: 0.31, alpha: 0.88).cgColor)
        ctx.fill(CGRect(x: 0, y: y, width: bounds.width, height: h))
        let w = width(label, bannerFont)
        text(label, at: CGPoint(x: (bounds.width - w) / 2, y: y + (h - bannerFont.lineHeight) / 2), font: bannerFont)
    }

    private func drawPanel(_ ctx: CGContext, _ top: CGFloat, _ now: Int64) {
        func age(_ t: Int64) -> String { t == 0 ? "-" : "\((now - t) / 1000)秒前" }
        let removed = d.removed
        var lines = [
            "追跡: \(d.tracking)  向き: \(d.headingDeg.map { String(format: "%.0f°", $0) } ?? "-")  \(d.settings)",
            "場面: \(d.scene.rawValue)  商品の枠: \(d.productBoxes.count)  \(d.sceneMs)（\(age(d.sceneAt))）",
            "動き: \(d.motion.isEmpty ? "-" : d.motion)  省いた: \(d.skipInfo.isEmpty ? "なし" : d.skipInfo)",
            "② 主な分類: \(d.mainGroup ?? "-")  使った \(d.kept.count) / 外した \(removed.count) / 名前なし \(d.unnamed)  \(d.describeInfo)（\(age(d.productsAt))）",
        ]
        if !removed.isEmpty {
            lines.append("   外した: " + removed.orderedGroups { $0.genre ?? "?" }.map { "\($0.key)×\($0.values.count)" }.joined(separator: "、"))
        }
        if !d.description.isEmpty { lines.append("   説明: \(d.description)") }
        if !d.shelfChange.isEmpty { lines.append("   棚の替わり: \(d.shelfChange)") }
        lines.append("③ 近い商品: \(d.near.isEmpty ? "-" : d.near)")
        lines.append("③ 長押し: \(d.longPress.isEmpty ? "-" : d.longPress)（\(age(d.focusAt))）")
        if !d.lookup.isEmpty { lines.append("   商品: \(d.lookup)") }
        lines.append("④ 人: \(d.peopleInfo.isEmpty ? "-" : d.peopleInfo)")
        lines.append("声: \(d.voice)")

        let body = lines.joined(separator: "\n") as NSString
        let attrs: [NSAttributedString.Key: Any] = [.font: textFont, .foregroundColor: UIColor.white]
        let maxWidth = bounds.width - 16
        let size = body.boundingRect(with: CGSize(width: maxWidth, height: .greatestFiniteMagnitude), options: .usesLineFragmentOrigin, attributes: attrs, context: nil)
        let bottomInset = safeAreaInsets.bottom
        let panelTop = max(top, bounds.height - bottomInset - size.height - 12)
        ctx.setFillColor(UIColor(white: 0, alpha: 0.69).cgColor)
        ctx.fill(CGRect(x: 0, y: panelTop, width: bounds.width, height: bounds.height - panelTop))
        body.draw(with: CGRect(x: 8, y: panelTop + 6, width: maxWidth, height: bounds.height - panelTop), options: .usesLineFragmentOrigin, attributes: attrs, context: nil)
    }

    private func short(_ name: String) -> String {
        let after = name.firstIndex(of: " ").map { String(name[name.index(after: $0)...]) } ?? name
        return String(after.prefix(10))
    }

    private func pct(_ p: Float) -> String { "\(Int(p * 100))%" }
}
