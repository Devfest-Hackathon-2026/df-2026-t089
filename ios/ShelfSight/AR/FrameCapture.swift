import ARKit
import CoreGraphics

/// AR のキューから取り出した撮影データ。ARFrame への参照は持たない（持ち続けると ARKit のカメラ画像が足りなくなる）。
final class CapturedFrame {
    /// 正立させた画像（縦持ち）。RGBA のバイト列のまま（Core Graphics を通さない）。
    let image: RGBAImage
    let geometry: CaptureGeometry

    init(image: RGBAImage, geometry: CaptureGeometry) {
        self.image = image
        self.geometry = geometry
    }
}

enum FrameCapture {
    /// 人の検出には十分で、AR のキューを止めない大きさ。
    static let defaultLongSide = 640
    /// 縦持ちの画面では、センサー画像（横向き）を時計回りに 90° 回すと正立する。
    static let portraitRotation = 90

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
        // 正立画像は縦長（幅 = センサーの高さ）。Core Image は使わない（解析のたびにメモリが 20MB ずつ増え、20 秒ほどで落ちた）
        guard let upright = uprightImage(buffer, width: sh, height: sw) else { return nil }

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

    /**
     * カメラの画像（YCbCr 4:2:0、横向き）を、時計回りに 90° 回して縮めた RGB の画像にする（Android 版の NV21 の変換にあたる）。
     * 縮めるときは近い画素を取る。width × height は正立画像の大きさ。
     */
    private static func uprightImage(_ buffer: CVPixelBuffer, width outW: Int, height outH: Int) -> RGBAImage? {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        let videoRange = format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        guard videoRange || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
              CVPixelBufferGetPlaneCount(buffer) >= 2 else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let cBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return nil }
        let w = CVPixelBufferGetWidthOfPlane(buffer, 0)
        let h = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let yRow = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let cRow = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let yp = yBase.assumingMemoryBound(to: UInt8.self)
        let cp = cBase.assumingMemoryBound(to: UInt8.self)

        // 正立画像の (ox, oy) は、センサー画像の x = oy の位置、y = 下から ox の位置（uprightToSensor の 90°）
        let sensorY: [Int] = (0..<outW).map { ox in
            Int((1 - (Double(ox) + 0.5) / Double(outW)) * Double(h)).clamped(0, h - 1)
        }
        var rgba = [UInt8](repeating: 255, count: outW * outH * 4)
        rgba.withUnsafeMutableBufferPointer { o in
            sensorY.withUnsafeBufferPointer { sy in
                for oy in 0..<outH {
                    let sx = Int((Double(oy) + 0.5) / Double(outH) * Double(w)).clamped(0, w - 1)
                    let cx = sx & ~1
                    var p = oy * outW * 4
                    for ox in 0..<outW {
                        let y0 = sy[ox]
                        var luma = Int(yp[y0 * yRow + sx])
                        let ci = (y0 >> 1) * cRow + cx
                        var cb = Int(cp[ci]) - 128
                        var cr = Int(cp[ci + 1]) - 128
                        if videoRange {
                            luma = ((luma - 16) * 298) >> 8
                            cb = (cb * 292) >> 8
                            cr = (cr * 292) >> 8
                        }
                        // BT.601（ARKit の見本と同じ）。Swift ではシフトがかけ算より先に計算されるので、かっこで囲む
                        o[p] = UInt8(clamping: luma + ((359 * cr) >> 8))
                        o[p + 1] = UInt8(clamping: luma - ((88 * cb + 183 * cr) >> 8))
                        o[p + 2] = UInt8(clamping: luma + ((454 * cb) >> 8))
                        p += 4
                    }
                }
            }
        }
        return RGBAImage(width: outW, height: outH, pixels: rgba)
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
