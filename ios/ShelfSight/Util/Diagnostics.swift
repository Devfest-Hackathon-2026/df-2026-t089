import Foundation
import MetricKit
import os

/**
 * アプリが落ちる原因を調べるための記録（iOS だけの調査用。案内の機能には関わらない）。
 *
 * - 1 秒ごとに、経過時間・メモリ・あとどれだけ使えるか・温度の状態・AR のフレーム数・今の処理を
 *   Documents/diagnostics/<起動時刻>.log に 1 行ずつ書く。落ちても直前の行まで残る（「ファイル」アプリで見られる）
 * - 次に起動したとき、前回の最後の行と、MetricKit が届けた落ちた理由（クラッシュ・強制終了の回数）を画面に出す
 */
final class Diagnostics: NSObject, MXMetricManagerSubscriber {
    static let shared = Diagnostics()

    /// 今どの処理をしているか（落ちたときに最後に何をしていたかを残す）。
    @Locked var stage = "起動"
    /// 画面に出す 1 行（今の状態）と、前回の終わり方。
    @Locked private(set) var line = ""
    @Locked private(set) var previous = ""

    /// 処理ごとの、1 回あたりに増えたメモリ（MB の合計と回数）。
    @Locked private var growth: [String: (total: Double, count: Int)] = [:]
    private let growthLock = NSLock()

    @Locked private var lastFootprint: Double = -1
    @Locked private var lastMalloc: Double = -1

    @Locked private var frames = 0
    @Locked private var scenes = 0
    private let started = Date()
    private let queue = DispatchQueue(label: "diagnostics", qos: .utility, autoreleaseFrequency: .workItem)
    private var timer: DispatchSourceTimer?
    private var handle: FileHandle?
    private let dir: URL

    private override init() {
        dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("diagnostics")
        super.init()
    }

    func start() {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        previous = Self.lastLine(of: latestLog()) ?? ""
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        let url = dir.appendingPathComponent(f.string(from: started) + ".log")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try? FileHandle(forWritingTo: url)
        write("start build \(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?")")

        MXMetricManager.shared.add(self)
        didReceive(MXMetricManager.shared.pastDiagnosticPayloads)

        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 1, repeating: 1)
        t.setEventHandler { [weak self] in self?.tick() }
        t.resume()
        timer = t
    }

    /// body の前後でメモリを測り、増えた量を name に足す（今の処理として name も残す）。
    func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        stage = name
        let before = Self.footprintMB()
        defer {
            let d = Self.footprintMB() - before
            growthLock.lock()
            var g = growth
            let old = g[name] ?? (0, 0)
            g[name] = (old.total + d, old.count + 1)
            growth = g
            growthLock.unlock()
        }
        return try body()
    }

    func frame() { frames += 1 }
    func scene() { scenes += 1 }

    private func tick() {
        let fps = frames
        frames = 0
        // この 1 秒に増えたメモリ
        let footprint = Self.footprintMB()
        // 全体のうち、CPU 側のメモリ（malloc）が使っている量。全体だけが増えるなら、カメラの画像や GPU など malloc 以外が原因
        let heap = Self.mallocMB()
        var phaseText = String(format: "うち CPU 側（malloc）%.0fMB・それ以外 %.0fMB", heap, footprint - heap)
        if lastFootprint >= 0 && lastMalloc >= 0 {
            phaseText += String(format: "  この 1 秒の増え方: 全体 %+.0fMB（CPU 側 %+.0fMB・それ以外 %+.0fMB）",
                                footprint - lastFootprint, heap - lastMalloc, (footprint - heap) - (lastFootprint - lastMalloc))
        }
        lastFootprint = footprint
        lastMalloc = heap
        let elapsed = Int(Date().timeIntervalSince(started))
        let text = String(format: "%3ds メモリ %.0fMB（残り %.0fMB）熱 %@ AR %d/秒 解析 %d 回 処理「%@」",
                          elapsed, Self.footprintMB(), Self.availableMB(), Self.thermal(), fps, scenes, stage)
        let g = growth
        let perRun = g.sorted { $0.key < $1.key }
            .map { String(format: "%@ %+.1f", $0.key, $0.value.total / Double(max($0.value.count, 1))) }
            .joined(separator: " ")
        let full = (perRun.isEmpty ? text : text + "\n  1 回あたり増えた MB: " + perRun) + "\n  " + phaseText
        line = full
        write(full.replacingOccurrences(of: "\n", with: " |"))
    }

    private func write(_ text: String) {
        guard let h = handle else { return }
        let stamp = ISO8601DateFormatter().string(from: Date())
        // FileHandle.write はプロセスの中で溜めずに書くので、落ちても直前の行まで残る
        h.write(Data("\(stamp) \(text)\n".utf8))
    }

    // ------------------------------------------------------------ MetricKit（落ちた理由）

    func didReceive(_ payloads: [MXDiagnosticPayload]) {
        for p in payloads {
            save(p.jsonRepresentation(), "diagnostic")
            for c in p.crashDiagnostics ?? [] {
                let reason = [
                    c.exceptionType.map { "例外 \($0)" }, c.signal.map { "シグナル \($0)" },
                    c.terminationReason, c.virtualMemoryRegionInfo,
                ].compactMap { $0 }.joined(separator: " ")
                previous = "前回のクラッシュ: \(reason)（詳しくは diagnostics フォルダ）\n" + previous
            }
            if let hangs = p.hangDiagnostics, !hangs.isEmpty {
                previous = "前回の固まり: \(hangs.count) 回（\(hangs.first?.hangDuration.description ?? "")）\n" + previous
            }
        }
    }

    func didReceive(_ payloads: [MXMetricPayload]) {
        for p in payloads {
            save(p.jsonRepresentation(), "metrics")
            guard let e = p.applicationExitMetrics?.foregroundExitData else { continue }
            previous = "画面に出ている間の終了の回数: メモリ \(e.cumulativeMemoryResourceLimitExitCount)・" +
                "固まり \(e.cumulativeAppWatchdogExitCount)・クラッシュ \(e.cumulativeBadAccessExitCount + e.cumulativeIllegalInstructionExitCount + e.cumulativeAbnormalExitCount)\n" + previous
        }
    }

    private func save(_ json: Data, _ kind: String) {
        let name = "\(kind)-\(Int(Date().timeIntervalSince1970)).json"
        try? json.write(to: dir.appendingPathComponent(name))
    }

    // ------------------------------------------------------------ 補助

    private func latestLog() -> URL? {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "log" }.sorted { $0.lastPathComponent < $1.lastPathComponent }.last
    }

    private static func lastLine(of url: URL?) -> String? {
        guard let url, let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let lines = text.split(separator: "\n")
        guard let last = lines.last else { return nil }
        return "前回の最後の記録（\(lines.count) 行目）: " + String(last)
    }

    /// このアプリが使っているメモリ（OS がメモリ不足で止めるかどうかを決めるのに使う値）。
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    /// malloc（CPU 側のメモリ）で今使っている量。ONNX Runtime・画像の処理・Swift の配列などはここに入る。
    static func mallocMB() -> Double {
        var stats = malloc_statistics_t()
        malloc_zone_statistics(nil, &stats)
        return Double(stats.size_in_use) / 1_048_576
    }

    /// OS に止められるまでに、あとどれだけメモリを使えるか。
    static func availableMB() -> Double { Double(os_proc_available_memory()) / 1_048_576 }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "正常"
        case .fair: return "やや高い"
        case .serious: return "高い"
        case .critical: return "危険"
        @unknown default: return "?"
        }
    }
}
