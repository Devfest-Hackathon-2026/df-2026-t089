package jp.devfest.shelfsight.perception

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.NormalizedBox

/** 検出した人までの距離を求める。 */
internal object PersonDistance {
    private const val ASSUMED_PERSON_HEIGHT_M = 1.65f
    private const val FULL_BODY_EDGE_MARGIN = 0.02f

    /**
     * 全身が写っていれば、写った高さから距離を求める。深度は静止した景色を前提に計算されるため、
     * 動いている人には信頼できない可能性があるので、こちらを優先する。
     * 身体が画面からはみ出している近距離では、深度を使う。
     */
    fun estimate(box: NormalizedBox, geometry: CaptureGeometry): Float? {
        val fullBody = box.top > FULL_BODY_EDGE_MARGIN && box.bottom < 1f - FULL_BODY_EDGE_MARGIN
        if (fullBody) {
            val rotated = geometry.uprightRotationDegrees % 180 != 0
            val uprightHeight = if (rotated) geometry.imageWidth else geometry.imageHeight
            val focal = if (rotated) geometry.focalX else geometry.focalY
            return focal * ASSUMED_PERSON_HEIGHT_M / ((box.bottom - box.top) * uprightHeight)
        }
        return Unprojector.medianDepthMeters(box, geometry)
    }
}
