package jp.devfest.shelfsight.mapping

import kotlin.math.abs

/**
 * 直近のカメラ姿勢を時刻付きで保持する。
 *
 * 深度画像は最新のカメラフレームより少し古いので、深度と同じ時刻の姿勢を使わないと点がずれる。
 */
internal class PoseHistory<T : Any>(capacity: Int = DEFAULT_CAPACITY) {
    private val timestamps = LongArray(capacity)
    private val values = arrayOfNulls<Any>(capacity)
    private var next = 0
    private var size = 0
    private var hasLatest = false
    private var latestTimestamp = 0L

    fun record(timestampNanos: Long, value: T) {
        // カメラの更新より描画が速いと、同じフレームが続けて渡される。
        if (hasLatest && timestampNanos == latestTimestamp) return
        hasLatest = true
        latestTimestamp = timestampNanos
        timestamps[next] = timestampNanos
        values[next] = value
        next = (next + 1) % timestamps.size
        if (size < timestamps.size) size++
    }

    /** [timestampNanos] に最も近い姿勢を返す。差が [maxGapNanos] を超えるなら null を返す。 */
    @Suppress("UNCHECKED_CAST")
    fun nearest(timestampNanos: Long, maxGapNanos: Long): T? {
        var best = -1
        var bestGap = Long.MAX_VALUE
        for (i in 0 until size) {
            val gap = abs(timestamps[i] - timestampNanos)
            if (gap < bestGap) {
                bestGap = gap
                best = i
            }
        }
        if (best < 0 || bestGap > maxGapNanos) return null
        return values[best] as T
    }

    private companion object {
        const val DEFAULT_CAPACITY = 64 // 30fps で約 2 秒分
    }
}
