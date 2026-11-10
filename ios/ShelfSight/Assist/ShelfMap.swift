import Foundation

/// 壁（ARKit の縦の平面）。normal は単位ベクトル。
struct Wall {
    let center: Vec3
    let normal: Vec3
    let radius: Float
}

/// 地図に置いた 1 つの商品。同じ商品を何度も見るたびに位置をならし、seen を増やす。
struct MapItem {
    let name: String
    let genre: String?
    let fine: String?
    var position: Vec3
    var widthM: Float
    var heightM: Float
    var seen: Int
    var lastSeen: Int64
    /// この商品を最後に見たときのカメラの場所。視線の表示とガラスの補正に使う。
    var cameraAt: Vec3?
}

/// 認識した商品を AR 空間に貯める（アプリを閉じると消える）。同じ場所で違う名前になったものは、見えた回数の多い方を使う。
final class ShelfMap {
    static let mergeM: Float = 0.25
    static let sameSpotM: Float = 0.12
    static let minDepthM: Float = 0.2
    static let maxDepthM = Float(StoreModel.maxDepthM)
    static let maxItems = 3000
    /// 深度と壁がこれ以上離れていたら、壁に寄せない（棚の手前に出ている商品など）。
    static let snapToWallM: Float = 0.3

    private let lock = NSLock()
    private var items: [MapItem] = []
    /// 見つけたが、まだ分類できていない点。案内には使わず、地図に暗く描くだけ。
    private var pending: [Vec3] = []

    var size: Int { lock.lock(); defer { lock.unlock() }; return items.count }

    func addPending(_ position: Vec3) {
        lock.lock(); defer { lock.unlock() }
        if pending.contains(where: { $0.distanceTo(position) < Self.sameSpotM }) { return }
        pending.append(position)
        if pending.count > Self.maxItems { pending.removeFirst() }
    }

    func pendingPoints() -> [Vec3] { lock.lock(); defer { lock.unlock() }; return pending }

    func snapshot() -> [MapItem] { lock.lock(); defer { lock.unlock() }; return items }

    private func nearest(_ position: Vec3, within: Float, name: String? = nil) -> Int? {
        var best: Int?
        var bestD = within
        for (i, item) in items.enumerated() where name == nil || item.name == name {
            let d = item.position.distanceTo(position)
            if d < bestD { best = i; bestD = d }
        }
        return best
    }

    /// 新しく見えた商品を足す。近くに同じ名前の商品があれば、それに重ねる。
    func add(_ name: String, _ genre: String?, _ fine: String?, _ position: Vec3, _ widthM: Float, _ heightM: Float, _ now: Int64, _ cameraAt: Vec3? = nil) {
        lock.lock(); defer { lock.unlock() }
        pending.removeAll { $0.distanceTo(position) < Self.sameSpotM }
        guard let same = nearest(position, within: Self.mergeM, name: name) else {
            items.append(MapItem(name: name, genre: genre, fine: fine, position: position, widthM: widthM, heightM: heightM, seen: 1, lastSeen: now, cameraAt: cameraAt))
            if items.count > Self.maxItems, let oldest = items.indices.min(by: { items[$0].lastSeen < items[$1].lastSeen }) { items.remove(at: oldest) }
            return
        }
        merge(same, position, widthM, heightM, now, cameraAt)
    }

    /// 判定済みの商品がこの場所にあれば、見えたことだけ記録する。なければ false（新しい点なので判定がいる）。
    func touch(_ position: Vec3, _ widthM: Float, _ heightM: Float, _ now: Int64, _ cameraAt: Vec3?) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard let same = nearest(position, within: Self.sameSpotM) else { return false }
        merge(same, position, widthM, heightM, now, cameraAt)
        return true
    }

    private func merge(_ i: Int, _ position: Vec3, _ widthM: Float, _ heightM: Float, _ now: Int64, _ cameraAt: Vec3?) {
        let k = 1 / Float(items[i].seen + 1)
        items[i].position = items[i].position + (position - items[i].position) * k
        items[i].widthM += (widthM - items[i].widthM) * k
        items[i].heightM += (heightM - items[i].heightM) * k
        items[i].seen += 1
        items[i].lastSeen = now
        if let cameraAt { items[i].cameraAt = cameraAt }
    }

    /// 今の画面に入る商品を、画像の上の枠つきで返す。同じ場所に違う名前があれば、見えた回数の多い方だけ。
    func visible(_ g: CaptureGeometry) -> [RecognizedProduct] {
        let inView = snapshot().compactMap { item -> (MapItem, (Float, Float, Float))? in
            guard let p = Self.project(item.position, g), p.2 >= Self.minDepthM, p.2 <= Self.maxDepthM else { return nil }
            return (item, p)
        }
        var chosen: [(MapItem, (Float, Float, Float))] = []
        for c in inView.stableSorted(by: { $0.0.seen > $1.0.seen }) where !chosen.contains(where: { $0.0.position.distanceTo(c.0.position) < Self.sameSpotM }) {
            chosen.append(c)
        }
        return chosen.map { item, p in
            let (w, h) = Self.sizeInImage(item.widthM, item.heightM, p.2, g)
            return RecognizedProduct(left: p.0 - w / 2, top: p.1 - h / 2, right: p.0 + w / 2, bottom: p.1 + h / 2, name: item.name, genre: item.genre, fine: item.fine, sim: Float(item.seen))
        }
    }

    /// 世界座標の点を、写した画像の正立座標 (u, v) と、カメラからの奥行き (m) にする。カメラの後ろや画面の外なら nil。
    static func project(_ p: Vec3, _ g: CaptureGeometry) -> (Float, Float, Float)? {
        let m = g.cameraToWorld
        let dx = p.x - m[12], dy = p.y - m[13], dz = p.z - m[14]
        // 回転の逆は転置（列優先なので、列 i がカメラの軸 i）
        let x = m[0] * dx + m[1] * dy + m[2] * dz
        let y = m[4] * dx + m[5] * dy + m[6] * dz
        let z = m[8] * dx + m[9] * dy + m[10] * dz
        let depth = -z
        if depth <= 0 { return nil }
        let sx = (g.focalX * x / depth + g.principalX) / Float(g.imageWidth)
        let sy = (-g.focalY * y / depth + g.principalY) / Float(g.imageHeight)
        if sx < 0 || sx > 1 || sy < 0 || sy > 1 { return nil }
        let (u, v) = sensorToUpright(sx, sy, g.uprightRotationDegrees)
        return (u, v, depth)
    }

    /// センサー画像の正規化座標を、正立画像の正規化座標へ（Unprojector.uprightToSensor の逆）。
    static func sensorToUpright(_ sx: Float, _ sy: Float, _ rotation: Int) -> (Float, Float) {
        switch rotation {
        case 90: return (1 - sy, sx)
        case 180: return (1 - sx, 1 - sy)
        case 270: return (sy, 1 - sx)
        default: return (sx, sy)
        }
    }

    /// 正立画像の焦点距離（ピクセル）と大きさ。
    private static func upright(_ g: CaptureGeometry) -> (Float, Float, Float, Float) {
        g.uprightRotationDegrees % 180 != 0
            ? (g.focalY, g.focalX, Float(g.imageHeight), Float(g.imageWidth))
            : (g.focalX, g.focalY, Float(g.imageWidth), Float(g.imageHeight))
    }

    /// 画像の上の大きさ（正規化）を、奥行き depth での実際の大きさ (m) にする。
    static func sizeInMeters(_ w: Float, _ h: Float, _ depth: Float, _ g: CaptureGeometry) -> (Float, Float) {
        let (fu, fv, iw, ih) = upright(g)
        return (w * iw * depth / fu, h * ih * depth / fv)
    }

    static func sizeInImage(_ wM: Float, _ hM: Float, _ depth: Float, _ g: CaptureGeometry) -> (Float, Float) {
        let (fu, fv, iw, ih) = upright(g)
        return (wM * fu / depth / iw, hM * fv / depth / ih)
    }

    /// 商品の位置を決める。LiDAR の深度があればその点を、近くに壁があれば壁の上へ寄せる。深度がなければ、カメラから商品への向きを壁と交わらせる。
    static func place(camera: Vec3, along: Vec3, depthPoint: Vec3?, walls: [Wall]) -> Vec3? {
        func distance(_ w: Wall, _ p: Vec3) -> Float { Vec3.dot(w.normal, p - w.center) }
        func near(_ w: Wall, _ p: Vec3) -> Bool { (p - w.center).length < w.radius + 0.5 }
        if let depthPoint {
            let snap = walls.filter { abs(distance($0, depthPoint)) < snapToWallM && near($0, depthPoint) }
                .min { abs(distance($0, depthPoint)) < abs(distance($1, depthPoint)) }
            guard let w = snap else { return depthPoint }
            return depthPoint - w.normal * distance(w, depthPoint)
        }
        let d = (along - camera).normalized
        return walls.compactMap { w -> (Float, Vec3)? in
            let denom = Vec3.dot(w.normal, d)
            if abs(denom) < 0.2 { return nil }   // 壁とほぼ平行
            let t = Vec3.dot(w.normal, w.center - camera) / denom
            if t < minDepthM || t > maxDepthM { return nil }
            let p = camera + d * t
            return near(w, p) ? (t, p) : nil
        }.min { $0.0 < $1.0 }?.1
    }
}
