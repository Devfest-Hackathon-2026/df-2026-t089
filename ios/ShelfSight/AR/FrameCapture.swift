import ARKit
import CoreImage

/// AR のキューから取り出した撮影データ。ARFrame への参照は持たない（持ち続けると ARKit のカメラ画像が足りなくなる）。
final class CapturedFrame {
    /// 正立させた画像（縦持ち）。
    let image: CGImage
    let geometry: CaptureGeometry

    init(image: CGImage, geometry: CaptureGeometry) {
        self.image = image
        self.geometry = geometry
    }
}

enum FrameCapture {
    /// 人の検出には十分で、AR のキューを止めない大きさ。
    static let defaultLongSide = 640
    /// 縦持ちの画面では、センサー画像（横向き）を時計回りに 90° 回すと正立する。
    static let portraitRotation = 90

    private static let context = CIContext(options: [.cacheIntermediates: false])

    /**
     * AR のキューで呼ぶ。正立させた画像と、撮影時のカメラの状態を写し取る。
     * - Parameter fullResolution: 商品の文字を読むときなど、カメラ画像をそのままの解像度で使うなら true
     */
    static func capture(
        _ frame: ARFrame,
        depthEnabled: Bool,
        fullResolution: Bool = false,
        usualLongSide: Int = FrameCapture.defaultLongSide
    ) -> CapturedFrame? {
        let buffer = frame.capturedImage
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        guard w > 0, h > 0 else { return nil }
        let k: Float = fullResolution ? 1 : min(1, Float(usualLongSide) / Float(max(w, h)))
        let sw = max(2, Int(Float(w) * k))
        let sh = max(2, Int(Float(h) * k))
        var ci = CIImage(cvPixelBuffer: buffer)
        if sw != w || sh != h {
            ci = ci.transformed(by: CGAffineTransform(scaleX: CGFloat(sw) / CGFloat(w), y: CGFloat(sh) / CGFloat(h)))
        }
        ci = ci.oriented(.right)
        let rect = CGRect(x: ci.extent.minX.rounded(), y: ci.extent.minY.rounded(), width: CGFloat(sh), height: CGFloat(sw))
        guard let upright = context.createCGImage(ci, from: rect) else { return nil }

        let kx = Float(sw) / Float(w)
        let ky = Float(sh) / Float(h)
        let K = frame.camera.intrinsics
        let t = frame.camera.transform
        let m: [Float] = [
            t.columns.0.x, t.columns.0.y, t.columns.0.z, t.columns.0.w,
            t.columns.1.x, t.columns.1.y, t.columns.1.z, t.columns.1.w,
            t.columns.2.x, t.columns.2.y, t.columns.2.z, t.columns.2.w,
            t.columns.3.x, t.columns.3.y, t.columns.3.z, t.columns.3.w,
        ]
        let geometry = CaptureGeometry(
            imageWidth: sw,
            imageHeight: sh,
            focalX: K.columns.0.x * kx,
            focalY: K.columns.1.y * ky,
            principalX: K.columns.2.x * kx,
            principalY: K.columns.2.y * ky,
            cameraToWorld: m,
            uprightRotationDegrees: portraitRotation,
            depth: depthEnabled ? captureDepth(frame) : nil,
            // ARKit の深度はカメラ画像と同じ向き・同じ範囲なので、正規化座標がそのまま使える
            depthU0: 0,
            depthUPerPixel: 1 / Float(sw),
            depthV0: 0,
            depthVPerPixel: 1 / Float(sh)
        )
        return CapturedFrame(image: upright, geometry: geometry)
    }

    /// LiDAR のない機種では深度がないので nil（人の距離は写った高さから、商品の位置は壁から決める）。
    private static func captureDepth(_ frame: ARFrame) -> DepthSnapshot? {
        guard let d = frame.sceneDepth ?? frame.smoothedSceneDepth else { return nil }
        let map = d.depthMap
        guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map)
        let h = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
            for x in 0..<w { out[y * w + x] = row[x] }
        }
        return DepthSnapshot(width: w, height: h, meters: out)
    }
}
