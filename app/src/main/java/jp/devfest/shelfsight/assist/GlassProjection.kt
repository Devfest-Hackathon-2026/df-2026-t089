package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import kotlin.math.cos
import kotlin.math.sin

/** Project an unreliable depth point onto an evidenced glass front along its observed camera ray. */
internal object GlassProjection {
    fun corrected(item: MapItem, view: StoreView?): Vec3? {
        val camera = item.cameraAt ?: return null
        val store = view ?: return null
        val cs = cos(store.yaw); val sn = sin(store.yaw)
        fun local(p: Vec3) = doubleArrayOf(p.x * cs + p.z * sn, -p.x * sn + p.z * cs)
        val origin = local(camera)
        val raw = local(item.position)
        return store.fixtures.asSequence().filter { it.kind == "glass" }.mapNotNull { f ->
            val side = when {
                f.faces.containsKey("+") && !f.faces.containsKey("-") -> 1
                f.faces.containsKey("-") && !f.faces.containsKey("+") -> -1
                else -> return@mapNotNull null
            }
            val ax = f.axis; val along = 1 - ax
            val face = f.c + side * f.d / 2
            val delta = raw[ax] - origin[ax]
            val at = StoreModel.restoredAlong(origin[ax], origin[along], raw[ax], raw[along], face, side) ?: return@mapNotNull null
            if (at < f.a || at > f.b) return@mapNotNull null
            val t = (face - origin[ax]) / delta
            val u = origin[0] + t * (raw[0] - origin[0])
            val v = origin[1] + t * (raw[1] - origin[1])
            t to Vec3((u * cs - v * sn).toFloat(),
                (camera.y + t * (item.position.y - camera.y)).toFloat(), (u * sn + v * cs).toFloat())
        }.minByOrNull { it.first }?.second
    }
}
