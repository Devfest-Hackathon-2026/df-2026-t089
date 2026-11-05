package jp.devfest.shelfsight.scan

import jp.devfest.shelfsight.model.Vec3

/** 推論は電力と発熱の負担が大きいので、カメラが十分に動いたときだけ次のスキャンを許可する。 */
internal object ScanGate {
    private const val MIN_TRANSLATION_M = 0.15f
    private const val MIN_ROTATION_COS = 0.97f // およそ 14 度

    /** [lastForward] と [forward] はカメラ前方向の単位ベクトル。 */
    fun hasMovedEnough(lastPosition: Vec3, lastForward: Vec3, position: Vec3, forward: Vec3): Boolean {
        if (position.distanceTo(lastPosition) >= MIN_TRANSLATION_M) return true
        val cos = lastForward.x * forward.x + lastForward.y * forward.y + lastForward.z * forward.z
        return cos < MIN_ROTATION_COS
    }
}
