import Foundation

/**
 * MobileCLIP の判定を待つ、点ごとの切り抜き。判定には時間がかかり全部は追いつかないので、
 * 今いる場所に近く、新しく、大きく写り、検出の確からしさが高いものから取り出す。
 */
final class CropQueue<T> {
    struct Entry {
        let crop: T
        let box: [Float]
        let position: Vec3
        let camera: Vec3
        let widthM: Float
        let heightM: Float
        let area: Float
        let time: Int64
        let confidence: Float
    }

    private static var ageSeconds: Float { 0.2 }   // 5 秒古いと 1 m 遠いのと同じ
    private static var areaWeight: Float { 2 }
    private static var confidenceWeight: Float { 1 }  // 確からしさ 0.5 の差は 0.5 m 近いのと同じ
    private static var retryArea: Float { 1.5 }
    private static var maxRejected: Int { 1000 }

    private let capacity: Int
    private let lock = NSLock()
    private var items: [Entry] = []
    private var rejected: [(position: Vec3, area: Float)] = []

    init(capacity: Int) { self.capacity = capacity }

    var size: Int { lock.lock(); defer { lock.unlock() }; return items.count }

    /// 判定しても分からなかった場所を覚える。より大きく写るまで、同じ場所は判定し直さない。
    func reject(_ entry: Entry) {
        lock.lock(); defer { lock.unlock() }
        rejected.removeAll { $0.position.distanceTo(entry.position) < ShelfMap.sameSpotM }
        rejected.append((entry.position, entry.area))
        if rejected.count > Self.maxRejected { rejected.removeFirst() }
    }

    /// 同じ場所の切り抜きは大きく写った方だけ残す。
    func add(_ entry: Entry) {
        lock.lock(); defer { lock.unlock() }
        if let failed = rejected.first(where: { $0.position.distanceTo(entry.position) < ShelfMap.sameSpotM }), entry.area < failed.area * Self.retryArea { return }
        if let same = items.firstIndex(where: { $0.position.distanceTo(entry.position) < ShelfMap.sameSpotM }) {
            if items[same].area >= entry.area { return }
            items.remove(at: same)
            items.append(entry)
            return
        }
        items.append(entry)
        if items.count > capacity, let oldest = items.indices.min(by: { items[$0].time < items[$1].time }) { items.remove(at: oldest) }
    }

    /// いちばん優先度の高い切り抜きを取り出す。空なら nil。
    func take(user: Vec3?, now: Int64) -> Entry? {
        lock.lock(); defer { lock.unlock() }
        func score(_ e: Entry) -> Float {
            (user?.distanceTo(e.position) ?? 0) + Float(now - e.time) / 1000 * Self.ageSeconds - e.area * Self.areaWeight - e.confidence * Self.confidenceWeight
        }
        guard let best = items.indices.min(by: { score(items[$0]) < score(items[$1]) }) else { return nil }
        return items.remove(at: best)
    }
}
