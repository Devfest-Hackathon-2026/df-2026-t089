package jp.devfest.shelfsight.assist

import java.nio.FloatBuffer
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.sqrt

/** 商品を見た場所から、その商品の点までの視線（矢じりは点の側）。 */
internal object ProductViewArrow {
    const val FLOATS = 18 // three GL_LINES segments
    private const val HEAD_M = 0.10f
    private const val HALF_WIDTH_M = 0.07f

    fun append(point: Vec3, camera: Vec3, out: FloatBuffer) {
        val dx = point.x - camera.x
        val dz = point.z - camera.z
        val length = sqrt(dx * dx + dz * dz).coerceAtLeast(1e-4f)
        val fx = dx / length
        val fz = dz / length
        val backX = point.x - fx * HEAD_M
        val backZ = point.z - fz * HEAD_M
        val sideX = -fz * HALF_WIDTH_M
        val sideZ = fx * HALF_WIDTH_M
        fun vertex(x: Float, z: Float) { out.put(x).put(point.y).put(z) }
        out.put(camera.x).put(camera.y).put(camera.z)
        vertex(point.x, point.z)
        vertex(point.x, point.z)
        vertex(backX + sideX, backZ + sideZ)
        vertex(point.x, point.z)
        vertex(backX - sideX, backZ - sideZ)
    }
}
