import Foundation

/// 複数のスレッド（キュー）から読み書きする値。Android 版の @Volatile にあたる。
@propertyWrapper
final class Locked<Value> {
    private var value: Value
    private let lock = NSLock()

    init(wrappedValue: Value) { value = wrappedValue }

    var wrappedValue: Value {
        get { lock.lock(); defer { lock.unlock() }; return value }
        set { lock.lock(); value = newValue; lock.unlock() }
    }
}

/// 処理中かどうか（Android 版の AtomicBoolean の compareAndSet(false, true)）。
final class AtomicFlag {
    private var busy = false
    private let lock = NSLock()

    func tryAcquire() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if busy { return false }
        busy = true
        return true
    }

    func release() {
        lock.lock(); busy = false; lock.unlock()
    }
}

/// 初めて使うときに 1 回だけ作る（どのスレッドから呼んでもよい）。
final class Once<T> {
    private let make: () -> T
    private var value: T?
    private let lock = NSLock()

    init(_ make: @escaping () -> T) { self.make = make }

    var get: T {
        lock.lock(); defer { lock.unlock() }
        if let v = value { return v }
        let v = make()
        value = v
        return v
    }
}

/// 起動してからのミリ秒（Android 版の SystemClock.elapsedRealtime）。
func nowMs() -> Int64 { Int64(ProcessInfo.processInfo.systemUptime * 1000) }

func logInfo(_ tag: String, _ message: String) { NSLog("[%@] %@", tag, message) }

extension Comparable {
    func clamped(_ lo: Self, _ hi: Self) -> Self { min(max(self, lo), hi) }
}

extension Array {
    /// 同じ順位のものは元の順に並べる（Kotlin の sortedBy と同じ）。
    func stableSorted(by less: (Element, Element) -> Bool) -> [Element] {
        enumerated().sorted { a, b in
            if less(a.element, b.element) { return true }
            if less(b.element, a.element) { return false }
            return a.offset < b.offset
        }.map(\.element)
    }

    /// 最初に出てきた順のまとまり（Kotlin の groupBy と同じ順）。
    func orderedGroups<K: Hashable>(by key: (Element) -> K) -> [(key: K, values: [Element])] {
        var index: [K: Int] = [:]
        var out: [(key: K, values: [Element])] = []
        for e in self {
            let k = key(e)
            if let i = index[k] {
                out[i].values.append(e)
            } else {
                index[k] = out.count
                out.append((key: k, values: [e]))
            }
        }
        return out
    }
}

/// JSONSerialization の値を読む補助。
enum JSON {
    static func object(_ data: Data) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    static func object(_ text: String) -> [String: Any]? { object(Data(text.utf8)) }

    static func string(_ o: [String: Any]?, _ key: String) -> String {
        (o?[key] as? String) ?? ""
    }

    /// null や空なら nil。
    static func text(_ o: [String: Any]?, _ key: String) -> String? {
        guard let s = o?[key] as? String, !s.isEmpty else { return nil }
        return s
    }

    static func double(_ o: [String: Any]?, _ key: String) -> Double? { (o?[key] as? NSNumber)?.doubleValue }
    static func int(_ o: [String: Any]?, _ key: String) -> Int? { (o?[key] as? NSNumber)?.intValue }
    static func objects(_ o: [String: Any]?, _ key: String) -> [[String: Any]] { (o?[key] as? [Any])?.compactMap { $0 as? [String: Any] } ?? [] }

    static func serialize(_ value: Any) -> Data {
        (try? JSONSerialization.data(withJSONObject: value, options: [])) ?? Data()
    }
}
