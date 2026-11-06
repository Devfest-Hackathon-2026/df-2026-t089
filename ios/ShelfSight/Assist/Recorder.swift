import CoreGraphics
import Foundation

/**
 * 1 秒ごとの撮影を記録する。起動ごとにフォルダを作り、画像（frame_<番号>.jpg）と、1 行 1 枚の records.jsonl
 * （時刻、ARKit のカメラの姿勢、向き、カメラの内部パラメータ、認識した商品）を書く。
 * 「ファイル」アプリの「このiPhone内 > ShelfSight > records」から取り出せる（Info.plist の UIFileSharingEnabled）。
 * 書式は Android 版と同じ。
 */
final class Recorder {
    private static let jpegQuality: CGFloat = 0.8

    let dir: URL
    private let log: URL
    private var n = 0

    /// 商品 1 つの記録。position は地図に置いた位置（置けなければ nil）。
    struct Product {
        let p: RecognizedProduct
        let position: Vec3?
    }

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let f = DateFormatter()
        f.locale = Locale(identifier: "ja_JP")
        f.dateFormat = "yyyyMMdd_HHmmss"
        dir = docs.appendingPathComponent("records").appendingPathComponent(f.string(from: Date()))
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        log = dir.appendingPathComponent("records.jsonl")
    }

    func record(_ time: Int64, _ image: CGImage, _ g: CaptureGeometry, headingDeg: Float?, walls: Int, products: [Product]) {
        let name = String(format: "frame_%05d.jpg", n)
        n += 1
        do {
            try ImageUtil.jpeg(image, quality: Self.jpegQuality).write(to: dir.appendingPathComponent(name))
            let intrinsics: [String: Any] = [
                "fx": Double(g.focalX), "fy": Double(g.focalY), "cx": Double(g.principalX), "cy": Double(g.principalY),
                "sensor_width": g.imageWidth, "sensor_height": g.imageHeight, "upright_rotation": g.uprightRotationDegrees,
            ]
            let items: [[String: Any]] = products.map { pr in
                let p = pr.p
                let position = orNull(pr.position.map { [Double($0.x), Double($0.y), Double($0.z)] })
                return [
                    "name": p.name, "genre": orNull(p.genre), "fine": orNull(p.fine), "sim": Double(p.sim),
                    "box": [p.left, p.top, p.right, p.bottom].map { Double($0) }, "position": position,
                ]
            }
            let o: [String: Any] = [
                "t": time,
                "image": name,
                "width": image.width, "height": image.height,
                // 列優先の 4x4（カメラ → 世界）。カメラ座標は +X が右、+Y が上、-Z が前（センサー画像の向き）
                "camera_to_world": g.cameraToWorld.map { Double($0) },
                "heading_deg": orNull(headingDeg.map { Double($0) }),
                "intrinsics": intrinsics,
                "walls": walls,
                // 棚の種類の判別は入れていないので、いつも null（Android 版の書式に合わせる）
                "shelf_columns": NSNull(),
                "products": items,
                "platform": "ios",
            ]
            var line = JSON.serialize(o)
            line.append(0x0A)
            if let h = try? FileHandle(forWritingTo: log) {
                defer { try? h.close() }
                h.seekToEndOfFile()
                h.write(line)
            } else {
                try line.write(to: log)
            }
        } catch {
            logInfo("Recorder", "could not record \(name): \(error)")   // 記録できなくても案内は続ける
        }
    }

    /// JSON の null にする。
    private func orNull(_ v: Any?) -> Any { v ?? NSNull() }
}
