package jp.devfest.shelfsight.perception

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.Vec3

/** 指の先にある表面。[u], [v] は正立画像の正規化座標、[depthM] はカメラからの距離。 */
internal data class PointingHit(val u: Float, val v: Float, val depthM: Float)

/** 指先から指の向きへ線を伸ばし、最初に当たる表面を深度から探す。 */
internal object PointingTarget {
    private const val STEP_M = 0.05f
    private const val MAX_RANGE_M = 5f
    private const val START_BEYOND_TIP_M = 0.15f

    /** 手そのものの深度に当たったと誤らないよう、指先より確実に奥の表面だけを採用する。 */
    private const val MIN_BEYOND_HAND_M = 0.1f

    fun find(fingertip: Vec3, direction: Vec3, geometry: CaptureGeometry): PointingHit? {
        if (geometry.depth == null) return null
        val tipDepth = project(fingertip, geometry)?.depthM ?: return null

        var distance = START_BEYOND_TIP_M
        while (distance <= MAX_RANGE_M) {
            val point = Vec3(
                fingertip.x + direction.x * distance,
                fingertip.y + direction.y * distance,
                fingertip.z + direction.z * distance,
            )
            val projected = project(point, geometry) ?: return null
            if (projected.u !in 0f..1f || projected.v !in 0f..1f) return null
            val surface = Unprojector.depthAt(projected.u, projected.v, geometry)
            if (surface != null && surface > tipDepth + MIN_BEYOND_HAND_M && projected.depthM >= surface) {
                return PointingHit(projected.u, projected.v, surface)
            }
            distance += STEP_M
        }
        return null
    }

    /** ワールド座標の点を正立画像へ投影する。カメラの後ろなら null を返す。 */
    fun project(point: Vec3, geometry: CaptureGeometry): PointingHit? {
        val m = geometry.cameraToWorld
        val dx = point.x - m[12]
        val dy = point.y - m[13]
        val dz = point.z - m[14]
        // 回転行列の逆行列は転置行列。
        val cameraX = m[0] * dx + m[1] * dy + m[2] * dz
        val cameraY = m[4] * dx + m[5] * dy + m[6] * dz
        val cameraZ = m[8] * dx + m[9] * dy + m[10] * dz
        val depth = -cameraZ
        if (depth <= 0f) return null

        val sensorX = (geometry.focalX * cameraX / depth + geometry.principalX) / geometry.imageWidth
        val sensorY = (-geometry.focalY * cameraY / depth + geometry.principalY) / geometry.imageHeight
        val (u, v) = when (geometry.uprightRotationDegrees) {
            90 -> 1f - sensorY to sensorX
            180 -> 1f - sensorX to 1f - sensorY
            270 -> sensorY to 1f - sensorX
            else -> sensorX to sensorY
        }
        return PointingHit(u, v, depth)
    }
}
