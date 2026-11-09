import Foundation

/**
 * 商品認識（ProductRecognizer）を持ち、CPU 側のメモリ（malloc）が増えたら、解析を止めずに裏で作り直して入れ替える（iOS だけ）。
 *
 * ONNX Runtime は、動かすうちに作業用のメモリを中に取っておき、OS に返さない（写る場所が変わるたびに増え、
 * 3GB ほどで iOS のアプリごとの上限を超えて止められた）。セッションを捨てると、中に取っておいたメモリはまとめて返る。
 * 1. 新しい認識器を別のキューで作る（その間も、今の認識器で解析を続ける）
 * 2. できたら入れ替える。古い認識器は、使っている解析が終わったときに捨てられ、メモリが返る
 * 3. 返った分を malloc から OS へ返すよう頼む（malloc_zone_pressure_relief）
 */
final class RecognizerPool {
    /// 作り直すきっかけ: malloc で使っている量がこれを超えたか、残りのメモリがこれを切ったとき。
    static let reloadMallocMB: Double = 1000
    static let reloadAvailableMB: Double = 1000
    /// 続けて作り直さない間隔。
    static let minIntervalMs: Int64 = 5000

    @Locked private var current: ProductRecognizer? = nil
    private let loadLock = NSLock()
    @Locked private var reloading = false
    @Locked private(set) var reloads = 0
    @Locked private(set) var lastReloadAt: Int64 = 0
    /// 最後に作り直したときに返った量（MB、作り直す前と、入れ替えて少し待った後の malloc の差）。
    @Locked private(set) var lastFreedMB: Double = 0
    private let queue = DispatchQueue(label: "recognizer-reload", qos: .utility, autoreleaseFrequency: .workItem)

    /// 今の認識器。まだなければ、ここで作る（最初の 1 回）。
    var get: ProductRecognizer? {
        if let c = current { return c }
        loadLock.lock()
        defer { loadLock.unlock() }
        if let c = current { return c }
        let made = Self.make()
        current = made
        return made
    }

    /// メモリが増えていれば、裏で作り直す。解析のたびに呼んでよい（必要なときだけ動く）。
    func reloadIfNeeded() {
        guard current != nil, !reloading, nowMs() - lastReloadAt >= Self.minIntervalMs else { return }
        let heap = Diagnostics.mallocMB()
        guard heap > Self.reloadMallocMB || Diagnostics.availableMB() < Self.reloadAvailableMB else { return }
        reloading = true
        queue.async { [self] in
            logInfo("Recognizer", String(format: "reload: malloc %.0fMB, available %.0fMB", heap, Diagnostics.availableMB()))
            if let fresh = Self.make() {
                current = fresh   // 古い認識器は、使っている解析が終わったときに捨てられる
            }
            lastReloadAt = nowMs()
            reloads += 1
            // 古い認識器を使っていた解析が終わるのを少し待ってから、空いた分を OS に返し、返った量を測る
            queue.asyncAfter(deadline: .now() + 1.5) { [self] in
                malloc_zone_pressure_relief(nil, 0)
                lastFreedMB = heap - Diagnostics.mallocMB()
                logInfo("Recognizer", String(format: "reload done: freed %.0fMB", lastFreedMB))
                reloading = false
            }
        }
    }

    /// 画面に出す 1 行。
    var summary: String {
        let where_ = current?.backend ?? "未読み込み"
        if reloads == 0 { return "計算 \(where_)・作り直し 0 回" }
        let ago = (nowMs() - lastReloadAt) / 1000
        return "計算 \(where_)・" + String(format: "作り直し %d 回（最後は %d 秒前、返った量 %.0fMB）", reloads, Int(ago), lastFreedMB)
    }

    private static func make() -> ProductRecognizer? {
        do {
            return try ProductRecognizer()
        } catch {
            logInfo("Assist", "product recognizer: \(error)")
            return nil
        }
    }
}
