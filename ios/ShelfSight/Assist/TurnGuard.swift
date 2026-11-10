import Foundation

/**
 * 棚や商品の説明を始めたときの向きから、maxTurnDeg より大きく向きを変えたら説明をやめる（Android 版の TurnGuard）。
 * 向いている棚が変わったので、続きを聞いても今の前の棚の話ではなくなるため。
 *
 * start で説明を始めたときの向きと、そのときの声の世代（Voice.turn）を覚える。
 * turned を毎フレーム呼び、やめるべきときに true を返す（1 回だけ）。ほかの声で上書きされたり（世代が変わる）、
 * 話し終わったりしたら、見るのをやめる。複数のスレッドから呼んでよい。
 */
final class TurnGuard {
    static let maxTurnDeg: Float = 90
    static let startGraceMs: Int64 = 3000

    private let lock = NSLock()
    private var forwardX: Float = 0
    private var forwardZ: Float = 0
    private var turn = 0
    private var startedAt: Int64 = 0
    private var active = false

    func start(_ l: Listener, turn: Int, now: Int64) {
        lock.lock(); defer { lock.unlock() }
        forwardX = l.forwardX
        forwardZ = l.forwardZ
        self.turn = turn
        startedAt = now
        active = true
    }

    func clear() {
        lock.lock(); defer { lock.unlock() }
        active = false
    }

    /// speaking は声が鳴っているか。合成を待つ間は鳴っていないので、始めてから startGraceMs は待つ。
    func turned(_ l: Listener, turn: Int, speaking: Bool, now: Int64) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard active else { return false }
        if turn != self.turn || (!speaking && now - startedAt > Self.startGraceMs) {
            active = false
            return false
        }
        if abs(Self.angleDeg(forwardX, forwardZ, l.forwardX, l.forwardZ)) <= Self.maxTurnDeg { return false }
        active = false
        return true
    }

    /// 水平の向き (ax, az) から (bx, bz) への角度（度、-180〜180）。
    static func angleDeg(_ ax: Float, _ az: Float, _ bx: Float, _ bz: Float) -> Float {
        atan2(ax * bz - az * bx, ax * bx + az * bz) * 180 / .pi
    }
}
