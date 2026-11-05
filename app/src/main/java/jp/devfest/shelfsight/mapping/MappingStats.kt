package jp.devfest.shelfsight.mapping

/**
 * 1 回の記録全体の集計。
 *
 * ARCore のログが多く、logcat には直近の分しか残らないので、累計を定期的に出力する。
 * GL スレッドのみが触る。
 */
internal class MappingStats {
    private val lagBuckets = IntArray(BUCKET_COUNT)
    private var accepted = 0
    private var maxAcceptedLagMs = 0L
    private var stale = 0
    private var staleEpisodes = 0
    private var inStaleEpisode = false
    private var maxStaleLagMs = 0L
    private var noPose = 0
    private var trackingLosses = 0
    private var wasTracking = false
    private var trailJumps = 0
    private var maxTrailJumpM = 0f

    fun onAccepted(lagMs: Long) {
        inStaleEpisode = false
        accepted++
        if (lagMs > maxAcceptedLagMs) maxAcceptedLagMs = lagMs
        lagBuckets[(lagMs / BUCKET_MS).toInt().coerceIn(0, BUCKET_COUNT - 1)]++
    }

    fun onStale(lagMs: Long) {
        if (!inStaleEpisode) staleEpisodes++
        inStaleEpisode = true
        stale++
        if (lagMs > maxStaleLagMs) maxStaleLagMs = lagMs
    }

    fun onNoPose() {
        noPose++
    }

    fun onTracking(tracking: Boolean) {
        if (wasTracking && !tracking) trackingLosses++
        wasTracking = tracking
    }

    fun onTrailJump(meters: Float) {
        trailJumps++
        if (meters > maxTrailJumpM) maxTrailJumpM = meters
    }

    /** 採用した深度の時間差の中央値（バケットの下限）。 */
    fun medianLagMs(): Long {
        var remaining = accepted / 2
        for (bucket in 0 until BUCKET_COUNT) {
            remaining -= lagBuckets[bucket]
            if (remaining < 0) return bucket * BUCKET_MS
        }
        return 0
    }

    fun summary(): String =
        "accepted=$accepted lagMedianMs=${medianLagMs()} lagMaxMs=$maxAcceptedLagMs " +
            "stale=$stale staleEpisodes=$staleEpisodes staleMaxMs=$maxStaleLagMs noPose=$noPose " +
            "trackingLosses=$trackingLosses trailJumps=$trailJumps trailJumpMaxM=$maxTrailJumpM"

    private companion object {
        const val BUCKET_MS = 10L
        const val BUCKET_COUNT = 64
    }
}
