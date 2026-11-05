package jp.devfest.shelfsight.perception

import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.sqrt

enum class PersonMotion(val labelJa: String) {
    UNKNOWN("動きを確認中"),
    STATIONARY("止まっている"),
    APPROACHING("向かってくる"),
    LEAVING("離れていく"),
    CROSSING("横切っている"),
}

/** 1 回の検出で見つかった人。距離が分からない場合、位置は null。 */
internal data class PersonObservation(
    val box: NormalizedBox,
    val distanceM: Float?,
    val position: Vec3?,
)

internal data class TrackedPerson(
    val box: NormalizedBox,
    val distanceM: Float?,
    val motion: PersonMotion,
)

/** 地図に置く人。追跡中の人の、最新の推定位置。 */
data class PersonMarker(
    val id: Int,
    val position: Vec3,
    val motion: PersonMotion,
    val lastSeenSeconds: Double,
)

/**
 * 検出結果を時間方向につなぎ、人の動きを判定する。
 *
 * 位置は ARCore のワールド座標で比べる。胸につけたカメラは自分と一緒に動くので、
 * 画面上の動きでは、自分が歩いているのか相手が動いているのかを区別できないため。
 */
internal class PersonTracker {
    private class Track(val id: Int, var lastSeenSeconds: Double) {
        val times = ArrayList<Double>()
        val xs = ArrayList<Float>()
        val zs = ArrayList<Float>()
        var position = Vec3(0f, 0f, 0f)
        var motion = PersonMotion.UNKNOWN

        fun add(timeSeconds: Double, p: Vec3) {
            // 距離の推定は検出ごとにばらつくので、地図の表示がちらつかないよう滑らかにする。
            position = if (times.isEmpty()) {
                p
            } else {
                Vec3(
                    position.x + (p.x - position.x) * SMOOTHING,
                    position.y + (p.y - position.y) * SMOOTHING,
                    position.z + (p.z - position.z) * SMOOTHING,
                )
            }
            lastSeenSeconds = timeSeconds
            times += timeSeconds
            xs += p.x
            zs += p.z
            while (times.first() < timeSeconds - WINDOW_SECONDS) {
                times.removeAt(0)
                xs.removeAt(0)
                zs.removeAt(0)
            }
        }
    }

    private val tracks = ArrayList<Track>()
    private var nextId = 0

    /** 見失ってから時間が経っていない人も含めた、位置の分かっている人。 */
    fun markers(): List<PersonMarker> =
        tracks.map { PersonMarker(it.id, it.position, it.motion, it.lastSeenSeconds) }

    fun update(timeSeconds: Double, observations: List<PersonObservation>, camera: Vec3): List<TrackedPerson> {
        tracks.removeAll { timeSeconds - it.lastSeenSeconds > TRACK_TIMEOUT_SECONDS }

        val available = tracks.toMutableList()
        val result = observations.map { observation ->
            val position = observation.position
                ?: return@map TrackedPerson(observation.box, observation.distanceM, PersonMotion.UNKNOWN)

            val track = available
                .filter { it.position.distanceTo(position) < MATCH_RADIUS_M }
                .minByOrNull { it.position.distanceTo(position) }
                ?.also { available.remove(it) }
                ?: Track(nextId++, timeSeconds).also { tracks += it }
            track.add(timeSeconds, position)
            track.motion = classify(track, camera)
            TrackedPerson(observation.box, observation.distanceM, track.motion)
        }
        return result
    }

    private fun classify(track: Track, camera: Vec3): PersonMotion {
        val n = track.times.size
        if (n < MIN_SAMPLES || track.times.last() - track.times.first() < MIN_SPAN_SECONDS) {
            return PersonMotion.UNKNOWN
        }
        val vx = slope(track.times, track.xs)
        val vz = slope(track.times, track.zs)
        if (sqrt(vx * vx + vz * vz) < STATIONARY_SPEED) return PersonMotion.STATIONARY

        val toCameraX = camera.x - track.position.x
        val toCameraZ = camera.z - track.position.z
        val length = sqrt(toCameraX * toCameraX + toCameraZ * toCameraZ)
        if (length < 1e-3f) return PersonMotion.APPROACHING
        val closing = (vx * toCameraX + vz * toCameraZ) / length
        return when {
            closing > CLOSING_SPEED -> PersonMotion.APPROACHING
            closing < -CLOSING_SPEED -> PersonMotion.LEAVING
            else -> PersonMotion.CROSSING
        }
    }

    /** 最小二乗法で求めた、1 秒あたりの変化量。 */
    private fun slope(times: List<Double>, values: List<Float>): Float {
        val meanT = times.average()
        val meanV = values.average()
        var numerator = 0.0
        var denominator = 0.0
        for (i in times.indices) {
            val dt = times[i] - meanT
            numerator += dt * (values[i] - meanV)
            denominator += dt * dt
        }
        return if (denominator > 0.0) (numerator / denominator).toFloat() else 0f
    }

    private companion object {
        const val WINDOW_SECONDS = 1.5
        const val TRACK_TIMEOUT_SECONDS = 2.0
        const val SMOOTHING = 0.5f
        const val MATCH_RADIUS_M = 1.2f
        const val MIN_SAMPLES = 4
        const val MIN_SPAN_SECONDS = 0.7
        const val STATIONARY_SPEED = 0.35f
        const val CLOSING_SPEED = 0.35f
    }
}
