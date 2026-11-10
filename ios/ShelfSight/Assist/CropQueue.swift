import Foundation

/**
 * MobileCLIP の判定を待つ、点ごとの切り抜き（iOS 版）。Core ML で判定が速くなり計算に余裕があるので、
 * 検出（YOLO）の確からしさが高い順に、片っ端から取り出す（同じなら大きく写った方から）。
 * Android 版は、今いる場所に近く新しいものを先にする（CropQueue.kt）。
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

    /**
     * 同じ場所の切り抜きは大きく写った方だけ残す。判定待ちに入った（または同じ場所の切り抜きがすでに待っている）なら added。
     * 分からなかった場所で、前より十分大きく写っていなければ入れない（added は false）。
     * 上限を超えたら、検出の確からしさがいちばん低いものを捨て、その場所を dropped で返す。
     */
    @discardableResult
    func add(_ entry: Entry) -> (added: Bool, dropped: Vec3?) {
        lock.lock(); defer { lock.unlock() }
        if let failed = rejected.first(where: { $0.position.distanceTo(entry.position) < ShelfMap.sameSpotM }), entry.area < failed.area * Self.retryArea {
            return (false, nil)
        }
        if let same = items.firstIndex(where: { $0.position.distanceTo(entry.position) < ShelfMap.sameSpotM }) {
            if items[same].area < entry.area { items[same] = entry }
            return (true, nil)
        }
        items.append(entry)
        guard items.count > capacity, let worst = items.indices.min(by: { Self.before(items[$1], items[$0]) }) else { return (true, nil) }
        let dropped = items.remove(at: worst)
        return (dropped.position != entry.position, dropped.position)
    }

    /// いちばん確からしい切り抜きを取り出す。空なら nil。
    func take() -> Entry? {
        lock.lock(); defer { lock.unlock() }
        guard let best = items.indices.min(by: { Self.before(items[$0], items[$1]) }) else { return nil }
        return items.remove(at: best)
    }

    /// a を b より先に判定するか。
    private static func before(_ a: Entry, _ b: Entry) -> Bool {
        a.confidence != b.confidence ? a.confidence > b.confidence : a.area > b.area
    }
}
