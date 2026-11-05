package jp.devfest.shelfsight.mapping

import jp.devfest.shelfsight.model.Vec3
import kotlin.math.cos
import kotlin.math.sin

/** 地図を見るための視点。タッチ操作は UI スレッド、描画は GL スレッドから使う。 */
class OrbitCamera {
    private var yaw = 0f
    private var pitch = DEFAULT_PITCH
    private var distance = 10f
    private var target = Vec3(0f, 0f, 0f)

    @Synchronized
    fun fit(center: Vec3, radius: Float) {
        target = center
        distance = (radius * 2.2f).coerceIn(MIN_FIT_DISTANCE, MAX_DISTANCE)
        pitch = DEFAULT_PITCH
    }

    @Synchronized
    fun rotate(dxPixels: Float, dyPixels: Float) {
        yaw -= dxPixels * RADIANS_PER_PIXEL
        pitch = (pitch + dyPixels * RADIANS_PER_PIXEL).coerceIn(MIN_PITCH, MAX_PITCH)
    }

    @Synchronized
    fun zoom(factor: Float) {
        if (factor > 0f) distance = (distance / factor).coerceIn(MIN_DISTANCE, MAX_DISTANCE)
    }

    /** 指の動きに地図が付いてくるよう、注視点を床と平行に動かす。 */
    @Synchronized
    fun pan(dxPixels: Float, dyPixels: Float) {
        val scale = distance * METERS_PER_PIXEL_AT_UNIT_DISTANCE
        val rightX = cos(yaw)
        val rightZ = -sin(yaw)
        val forwardX = -sin(yaw)
        val forwardZ = -cos(yaw)
        target = Vec3(
            target.x - rightX * dxPixels * scale + forwardX * dyPixels * scale,
            target.y,
            target.z - rightZ * dxPixels * scale + forwardZ * dyPixels * scale,
        )
    }

    @Synchronized
    fun eye(): Vec3 = Vec3(
        target.x + distance * cos(pitch) * sin(yaw),
        target.y + distance * sin(pitch),
        target.z + distance * cos(pitch) * cos(yaw),
    )

    @Synchronized
    fun center(): Vec3 = target

    private companion object {
        const val DEFAULT_PITCH = 1.1f // 約 63 度。店内を斜め上から見下ろす
        const val MIN_PITCH = 0.05f
        const val MAX_PITCH = 1.55f
        const val MIN_DISTANCE = 0.5f
        const val MIN_FIT_DISTANCE = 4f
        const val MAX_DISTANCE = 200f
        const val RADIANS_PER_PIXEL = 0.006f
        const val METERS_PER_PIXEL_AT_UNIT_DISTANCE = 0.0015f
    }
}
