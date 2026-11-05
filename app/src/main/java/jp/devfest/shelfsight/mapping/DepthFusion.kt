package jp.devfest.shelfsight.mapping

import jp.devfest.shelfsight.ar.CaptureGeometry

/** 1 回分の生深度と、色を取るためのカメラ画像。ARCore のオブジェクトへの参照は持たない。 */
internal class DepthSample(
    val depthWidth: Int,
    val depthHeight: Int,
    /** ミリメートル。0 は無効。 */
    val depthMm: ShortArray,
    /** 0〜255。大きいほど信頼できる。 */
    val confidence: ByteArray,
    val nv21: ByteArray,
    /** intrinsics と、カメラ画像と深度画像の対応。cameraToWorld は使わない。 */
    val geometry: CaptureGeometry,
    /** カメラ座標から区画のローカル座標への変換（列優先の 4x4 行列）。 */
    val chunkFromCamera: FloatArray,
)

/** 深度画像の各ピクセルを 3D の点に戻し、[VoxelGrid] に書き込む。 */
internal object DepthFusion {
    private const val MIN_DEPTH_M = 0.3f
    private const val MAX_DEPTH_M = 4.0f // 遠いほど誤差が大きく、地図の精度を下げる
    private const val MIN_CONFIDENCE = 128
    private const val NO_COLOR = 0xFF808080.toInt()

    fun integrate(sample: DepthSample, grid: VoxelGrid) {
        val g = sample.geometry
        val m = sample.chunkFromCamera
        for (dy in 0 until sample.depthHeight) {
            val row = dy * sample.depthWidth
            for (dx in 0 until sample.depthWidth) {
                val index = row + dx
                if ((sample.confidence[index].toInt() and 0xFF) < MIN_CONFIDENCE) continue
                val depth = (sample.depthMm[index].toInt() and 0xFFFF) / 1000f
                if (depth < MIN_DEPTH_M || depth > MAX_DEPTH_M) continue

                // 深度ピクセル → テクスチャ座標 → カメラ画像のピクセル
                val u = (dx + 0.5f) / sample.depthWidth
                val v = (dy + 0.5f) / sample.depthHeight
                val px = (u - g.depthU0) / g.depthUPerPixel
                val py = (v - g.depthV0) / g.depthVPerPixel

                // カメラ座標は +X が画像の右、+Y が画像の上、-Z が前方
                val cx = (px - g.principalX) / g.focalX * depth
                val cy = -(py - g.principalY) / g.focalY * depth
                val cz = -depth

                grid.add(
                    m[0] * cx + m[4] * cy + m[8] * cz + m[12],
                    m[1] * cx + m[5] * cy + m[9] * cz + m[13],
                    m[2] * cx + m[6] * cy + m[10] * cz + m[14],
                    colorAt(sample.nv21, g.imageWidth, g.imageHeight, px.toInt(), py.toInt()),
                )
            }
        }
    }

    /** NV21 の 1 ピクセルを 0xAABBGGRR に変換する（BT.601）。 */
    private fun colorAt(nv21: ByteArray, width: Int, height: Int, x: Int, y: Int): Int {
        if (x !in 0 until width || y !in 0 until height) return NO_COLOR
        val luma = (nv21[y * width + x].toInt() and 0xFF)
        val chroma = width * height + (y / 2) * width + (x / 2) * 2
        val cr = (nv21[chroma].toInt() and 0xFF) - 128
        val cb = (nv21[chroma + 1].toInt() and 0xFF) - 128
        val r = (luma + 1.402f * cr).toInt().coerceIn(0, 255)
        val g = (luma - 0.344f * cb - 0.714f * cr).toInt().coerceIn(0, 255)
        val b = (luma + 1.772f * cb).toInt().coerceIn(0, 255)
        return (0xFF shl 24) or (b shl 16) or (g shl 8) or r
    }
}
