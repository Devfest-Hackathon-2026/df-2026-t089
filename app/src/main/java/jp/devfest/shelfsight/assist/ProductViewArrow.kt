package jp.devfest.shelfsight.assist

import java.nio.FloatBuffer
import jp.devfest.shelfsight.model.Vec3

/** A horizontal arrow ending at the product, pointing the way the camera faced when it was seen. */
internal object ProductViewArrow {
    const val FLOATS = 18 // three GL_LINES segments
    private const val LENGTH_M = 0.38f
    private const val HEAD_M = 0.10f
    private const val HALF_WIDTH_M = 0.07f

    fun append(point: Vec3, forward: Vec3, out: FloatBuffer) {
        val fx = forward.x
        val fz = forward.z
        val tipX = point.x
        val tipZ = point.z
        val backX = tipX - fx * HEAD_M
        val backZ = tipZ - fz * HEAD_M
        val sideX = -fz * HALF_WIDTH_M
        val sideZ = fx * HALF_WIDTH_M
        fun vertex(x: Float, z: Float) { out.put(x).put(point.y).put(z) }
        vertex(tipX - fx * LENGTH_M, tipZ - fz * LENGTH_M)
        vertex(tipX, tipZ)
        vertex(tipX, tipZ)
        vertex(backX + sideX, backZ + sideZ)
        vertex(tipX, tipZ)
        vertex(backX - sideX, backZ - sideZ)
    }
}
