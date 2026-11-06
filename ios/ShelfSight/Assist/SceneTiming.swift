import Foundation

/**
 * 商品の解析 1 回ごとの時間を貯め、直近 window 回の平均・最小・最大を出す（商品だけを計るモードの表示用）。
 * 1 回 = 撮る（AR のキュー）→ 正立画像にする → 枠を出す（YOLO）→ 名前を付ける（埋め込みと照合）。
 */
final class SceneTiming {
    private struct Run {
        let capture: Int64
        let convert: Int64
        let detect: Int64
        let name: Int64
        let boxes: Int
        let named: Int
        let at: Int64
        var total: Int64 { capture + convert + detect + name }
    }

    private let window: Int
    private let clock: () -> Int64
    private var runs: [Run] = []
    private var count = 0
    private let lock = NSLock()

    init(window: Int = 30, clock: @escaping () -> Int64 = nowMs) {
        self.window = window
        self.clock = clock
    }

    func add(captureMs: Int64, convertMs: Int64, detectMs: Int64, nameMs: Int64, boxes: Int, named: Int) {
        lock.lock(); defer { lock.unlock() }
        runs.append(Run(capture: captureMs, convert: convertMs, detect: detectMs, name: nameMs, boxes: boxes, named: named, at: clock()))
        while runs.count > window { runs.removeFirst() }
        count += 1
    }

    func lastLine() -> String {
        lock.lock(); defer { lock.unlock() }
        guard let r = runs.last else { return "-" }
        return "#\(count) total \(r.total)ms (capture \(r.capture), convert \(r.convert), detect \(r.detect), name \(r.name)), " +
            "boxes \(r.boxes), named \(r.named)"
    }

    func summary() -> String {
        lock.lock(); defer { lock.unlock() }
        if runs.isEmpty { return "計測中…" }
        func stat(_ label: String, _ f: (Run) -> Int64) -> String {
            let v = runs.map(f)
            return "\(label) 平均 \(v.reduce(0, +) / Int64(v.count)) / 最小 \(v.min()!) / 最大 \(v.max()!)ms"
        }
        // 1 回あたりの間隔（解析が終わってから次が終わるまで）。前の解析が終わるとすぐ次を回すので、これが実際の周期
        let period: Int64? = runs.count >= 2 ? (runs[runs.count - 1].at - runs[0].at) / Int64(runs.count - 1) : nil
        let periodText = period.map { "\($0)ms（\(String(format: "%.1f", 1000 / Float(max($0, 1)))) 回/秒）" } ?? "-"
        let boxes = Float(runs.map(\.boxes).reduce(0, +)) / Float(runs.count)
        let named = Float(runs.map(\.named).reduce(0, +)) / Float(runs.count)
        return [
            "直近 \(runs.count) 回（通算 \(count) 回）  周期 \(periodText)",
            stat("合計") { $0.total },
            stat("枠") { $0.detect } + "  " + stat("名前") { $0.name },
            stat("撮る") { $0.capture } + "  " + stat("変換") { $0.convert },
            String(format: "枠 平均 %.1f 個 / 名前 平均 %.1f 個", boxes, named),
        ].joined(separator: "\n")
    }
}
