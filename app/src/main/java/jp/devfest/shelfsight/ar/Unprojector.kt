package jp.devfest.shelfsight.ar

import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3

/** 16 ビット深度画像のコピー。値はミリメートルで、0 は無効。 */
class DepthSnapshot(val width: Int, val height: Int, val millimeters: ShortArray) {
    fun at(x: Int, y: Int): Int = millimeters[y * width + x].toInt() and 0xFFFF
}

/**
 * 撮影した瞬間のカメラの状態。
 *
 * Gemma の推論には時間がかかり、その間にカメラが動く。結果が返った時点の Frame を使うと
 * ラベルがずれるため、撮影時の値を保存しておき、それを使って 3D 位置へ戻す。
 */
class CaptureGeometry(
    val imageWidth: Int,
    val imageHeight: Int,
    val focalX: Float,
    val focalY: Float,
    val principalX: Float,
    val principalY: Float,
    /** 列優先の 4x4 行列。カメラ座標は +X が画像の右、+Y が画像の上、-Z が前方。 */
    val cameraToWorld: FloatArray,
    /** センサー画像を正立させるために時計回りに回す角度（0, 90, 180, 270）。 */
    val uprightRotationDegrees: Int,
    val depth: DepthSnapshot?,
    /** CPU 画像のピクセル座標から深度画像の正規化座標への変換。 */
    val depthU0: Float,
    val depthUPerPixel: Float,
    val depthV0: Float,
    val depthVPerPixel: Float,
)

internal object Unprojector {
    private const val FALLBACK_DISTANCE_M = 1.0f
    private const val MIN_DEPTH_M = 0.2f
    private const val MAX_DEPTH_M = 8.0f
    private const val SAMPLE_GRID = 9
    private const val SAMPLED_BOX_FRACTION = 0.4f

    data class Result(val position: Vec3, val hasMeasuredDepth: Boolean)

    fun unproject(box: NormalizedBox, geometry: CaptureGeometry): Result {
        val u = (box.left + box.right) / 2f
        val v = (box.top + box.bottom) / 2f
        val measured = medianDepthMeters(box, geometry)
        val (sx, sy) = uprightToSensor(u, v, geometry.uprightRotationDegrees)
        val position = pixelToWorld(
            pixelX = sx * geometry.imageWidth,
            pixelY = sy * geometry.imageHeight,
            depth = measured ?: FALLBACK_DISTANCE_M,
            geometry = geometry,
        )
        return Result(position, measured != null)
    }

    /** 正立画像の正規化座標を、回転前のセンサー画像の正規化座標へ戻す。 */
    fun uprightToSensor(u: Float, v: Float, rotationDegrees: Int): Pair<Float, Float> =
        when (rotationDegrees) {
            90 -> v to 1f - u
            180 -> 1f - u to 1f - v
            270 -> 1f - v to u
            else -> u to v
        }

    /** 正立画像の座標 ([u], [v]) にある、カメラから [depth] メートルの点のワールド座標。 */
    fun worldAt(u: Float, v: Float, depth: Float, geometry: CaptureGeometry): Vec3 {
        val (sx, sy) = uprightToSensor(u, v, geometry.uprightRotationDegrees)
        return pixelToWorld(sx * geometry.imageWidth, sy * geometry.imageHeight, depth, geometry)
    }

    /** 正立画像の座標 ([u], [v]) の深度。無効または範囲外なら null を返す。 */
    fun depthAt(u: Float, v: Float, geometry: CaptureGeometry): Float? {
        val depth = geometry.depth ?: return null
        val (sx, sy) = uprightToSensor(u, v, geometry.uprightRotationDegrees)
        val du = geometry.depthU0 + sx * geometry.imageWidth * geometry.depthUPerPixel
        val dv = geometry.depthV0 + sy * geometry.imageHeight * geometry.depthVPerPixel
        val dx = (du * depth.width).toInt()
        val dy = (dv * depth.height).toInt()
        if (dx !in 0 until depth.width || dy !in 0 until depth.height) return null
        val meters = depth.at(dx, dy) / 1000f
        return meters.takeIf { it in MIN_DEPTH_M..MAX_DEPTH_M }
    }

    /** 矩形の端は棚の背景を拾いやすいので、中央部分だけを見て中央値を取る。 */
    fun medianDepthMeters(box: NormalizedBox, geometry: CaptureGeometry): Float? {
        if (geometry.depth == null) return null
        val centerU = (box.left + box.right) / 2f
        val centerV = (box.top + box.bottom) / 2f
        val halfW = (box.right - box.left) * SAMPLED_BOX_FRACTION / 2f
        val halfH = (box.bottom - box.top) * SAMPLED_BOX_FRACTION / 2f

        val samples = ArrayList<Float>(SAMPLE_GRID * SAMPLE_GRID)
        for (row in 0 until SAMPLE_GRID) {
            for (col in 0 until SAMPLE_GRID) {
                val u = centerU - halfW + 2f * halfW * col / (SAMPLE_GRID - 1)
                val v = centerV - halfH + 2f * halfH * row / (SAMPLE_GRID - 1)
                depthAt(u, v, geometry)?.let { samples += it }
            }
        }
        if (samples.isEmpty()) return null
        samples.sort()
        return samples[samples.size / 2]
    }

    private fun pixelToWorld(pixelX: Float, pixelY: Float, depth: Float, geometry: CaptureGeometry): Vec3 {
        val x = (pixelX - geometry.principalX) / geometry.focalX * depth
        val y = -(pixelY - geometry.principalY) / geometry.focalY * depth
        val z = -depth
        val m = geometry.cameraToWorld
        return Vec3(
            m[0] * x + m[4] * y + m[8] * z + m[12],
            m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14],
        )
    }
}
