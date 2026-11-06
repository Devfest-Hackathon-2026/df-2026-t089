import Foundation

enum PersonMotion {
    case unknown, stationary, approaching, leaving, crossing

    var labelJa: String {
        switch self {
        case .unknown: return "動きを確認中"
        case .stationary: return "止まっている"
        case .approaching: return "向かってくる"
        case .leaving: return "離れていく"
        case .crossing: return "横切っている"
        }
    }
}

/// 1 回の検出で見つかった人。距離が分からない場合、位置は nil。
struct PersonObservation {
    let box: NormalizedBox
    let distanceM: Float?
    let position: Vec3?
}

struct TrackedPerson {
    let box: NormalizedBox
    let distanceM: Float?
    let motion: PersonMotion
}

/// 追跡中の人の、最新の推定位置。
struct PersonMarker {
    let id: Int
    let position: Vec3
    let motion: PersonMotion
    let lastSeenSeconds: Double
}

/// 検出した人までの距離を求める。
enum PersonDistance {
    private static let assumedPersonHeightM: Float = 1.65
    private static let fullBodyEdgeMargin: Float = 0.02

    /**
     * 全身が写っていれば、写った高さから距離を求める（深度は止まった景色を前提にしているので、動く人には頼れない）。
     * 身体が画面からはみ出している近距離では、深度を使う。
     */
    static func estimate(_ box: NormalizedBox, _ g: CaptureGeometry) -> Float? {
        let fullBody = box.top > fullBodyEdgeMargin && box.bottom < 1 - fullBodyEdgeMargin
        if fullBody {
            let rotated = g.uprightRotationDegrees % 180 != 0
            let uprightHeight = Float(rotated ? g.imageWidth : g.imageHeight)
            let focal = rotated ? g.focalX : g.focalY
            return focal * assumedPersonHeightM / ((box.bottom - box.top) * uprightHeight)
        }
        return Unprojector.medianDepthMeters(box, g)
    }
}

/**
 * 検出結果を時間方向につなぎ、人の動きを判定する。
 * 位置は ARKit のワールド座標で比べる。胸のカメラは自分と一緒に動くので、画面上の動きでは自分と相手の動きを区別できない。
 */
final class PersonTracker {
    private static let windowSeconds = 1.5
    private static let trackTimeoutSeconds = 2.0
    private static let smoothing: Float = 0.5
    private static let matchRadiusM: Float = 1.2
    private static let minSamples = 4
    private static let minSpanSeconds = 0.7
    private static let stationarySpeed: Float = 0.35
    private static let closingSpeed: Float = 0.35

    private final class Track {
        let id: Int
        var lastSeenSeconds: Double
        var times: [Double] = []
        var xs: [Float] = []
        var zs: [Float] = []
        var position = Vec3.zero
        var motion = PersonMotion.unknown

        init(id: Int, lastSeenSeconds: Double) {
            self.id = id
            self.lastSeenSeconds = lastSeenSeconds
        }

        func add(_ t: Double, _ p: Vec3) {
            // 距離の推定は検出ごとにばらつくので滑らかにする
            position = times.isEmpty ? p : position + (p - position) * PersonTracker.smoothing
            lastSeenSeconds = t
            times.append(t)
            xs.append(p.x)
            zs.append(p.z)
            while let first = times.first, first < t - PersonTracker.windowSeconds {
                times.removeFirst()
                xs.removeFirst()
                zs.removeFirst()
            }
        }
    }

    private var tracks: [Track] = []
    private var nextId = 0

    /// 見失ってから時間が経っていない人も含めた、位置の分かっている人。
    func markers() -> [PersonMarker] {
        tracks.map { PersonMarker(id: $0.id, position: $0.position, motion: $0.motion, lastSeenSeconds: $0.lastSeenSeconds) }
    }

    func update(_ t: Double, _ observations: [PersonObservation], camera: Vec3) -> [TrackedPerson] {
        tracks.removeAll { t - $0.lastSeenSeconds > Self.trackTimeoutSeconds }
        var available = tracks
        return observations.map { o in
            guard let position = o.position else {
                return TrackedPerson(box: o.box, distanceM: o.distanceM, motion: .unknown)
            }
            let track: Track
            if let near = available.filter({ $0.position.distanceTo(position) < Self.matchRadiusM })
                .min(by: { $0.position.distanceTo(position) < $1.position.distanceTo(position) }) {
                available.removeAll { $0 === near }
                track = near
            } else {
                track = Track(id: nextId, lastSeenSeconds: t)
                nextId += 1
                tracks.append(track)
            }
            track.add(t, position)
            track.motion = classify(track, camera)
            return TrackedPerson(box: o.box, distanceM: o.distanceM, motion: track.motion)
        }
    }

    private func classify(_ track: Track, _ camera: Vec3) -> PersonMotion {
        guard track.times.count >= Self.minSamples,
              let first = track.times.first, let last = track.times.last,
              last - first >= Self.minSpanSeconds else { return .unknown }
        let vx = slope(track.times, track.xs)
        let vz = slope(track.times, track.zs)
        if (vx * vx + vz * vz).squareRoot() < Self.stationarySpeed { return .stationary }
        let tx = camera.x - track.position.x
        let tz = camera.z - track.position.z
        let length = (tx * tx + tz * tz).squareRoot()
        if length < 1e-3 { return .approaching }
        let closing = (vx * tx + vz * tz) / length
        if closing > Self.closingSpeed { return .approaching }
        if closing < -Self.closingSpeed { return .leaving }
        return .crossing
    }

    /// 最小二乗法で求めた、1 秒あたりの変化量。
    private func slope(_ times: [Double], _ values: [Float]) -> Float {
        let meanT = times.reduce(0, +) / Double(times.count)
        let meanV = values.reduce(0) { $0 + Double($1) } / Double(values.count)
        var num = 0.0
        var den = 0.0
        for i in times.indices {
            let dt = times[i] - meanT
            num += dt * (Double(values[i]) - meanV)
            den += dt * dt
        }
        return den > 0 ? Float(num / den) : 0
    }
}
