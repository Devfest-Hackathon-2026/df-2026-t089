package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.abs
import kotlin.math.sqrt

/** 壁（ARCore の縦の平面）。[normal] は単位ベクトル。 */
data class Wall(val center: Vec3, val normal: Vec3, val radius: Float)

/** 地図に置いた 1 つの商品。同じ商品を何度も見るたびに位置をならし、[seen] を増やす。 */
class MapItem(
    val name: String, val genre: String?, val fine: String?,
    var position: Vec3, var widthM: Float, var heightM: Float, var seen: Int, var lastSeen: Long,
)

/**
 * 1 秒ごとの商品認識の結果を、AR 空間の壁の上に貯める（アプリを閉じると消える）。
 * 長押しで棚を説明するときは、今の画面に入る商品をここから取り出す。同じ場所で違う名前になったものは、
 * 見えた回数の多い方を使うので、1 枚ごとの読み違いに強い。
 */
class ShelfMap {
    private val items = ArrayList<MapItem>()

    @get:Synchronized val size: Int get() = items.size

    @Synchronized
    fun snapshot(): List<MapItem> = items.toList()

    /** 新しく見えた商品を足す。近くに同じ名前の商品があれば、それに重ねる。 */
    @Synchronized
    fun add(name: String, genre: String?, fine: String?, position: Vec3, widthM: Float, heightM: Float, now: Long) {
        val same = items.filter { it.name == name && it.position.distanceTo(position) < MERGE_M }.minByOrNull { it.position.distanceTo(position) }
        if (same == null) {
            items += MapItem(name, genre, fine, position, widthM, heightM, 1, now)
            if (items.size > MAX_ITEMS) items.remove(items.minBy { it.lastSeen })
            return
        }
        val k = 1f / (same.seen + 1)
        same.position = Vec3(
            same.position.x + (position.x - same.position.x) * k,
            same.position.y + (position.y - same.position.y) * k,
            same.position.z + (position.z - same.position.z) * k,
        )
        same.widthM += (widthM - same.widthM) * k
        same.heightM += (heightM - same.heightM) * k
        same.seen++
        same.lastSeen = now
    }

    /**
     * 今の画面に入る商品を、画像の上の枠つきで返す。同じ場所（[SAME_SPOT_M] 以内）に違う名前があれば、見えた回数の多い方だけ。
     */
    fun visible(g: CaptureGeometry): List<RecognizedProduct> {
        val inView = snapshot().mapNotNull { item -> project(item.position, g)?.let { item to it } }
            .filter { (_, p) -> p[2] in MIN_DEPTH_M..MAX_DEPTH_M }
        val chosen = mutableListOf<Pair<MapItem, FloatArray>>()
        for (c in inView.sortedByDescending { it.first.seen }) {
            if (chosen.none { it.first.position.distanceTo(c.first.position) < SAME_SPOT_M }) chosen += c
        }
        return chosen.map { (item, p) ->
            val (w, h) = sizeInImage(item.widthM, item.heightM, p[2], g)
            RecognizedProduct(p[0] - w / 2, p[1] - h / 2, p[0] + w / 2, p[1] + h / 2, item.name, item.genre, item.fine, item.seen.toFloat())
        }
    }

    companion object {
        const val MERGE_M = 0.25f
        const val SAME_SPOT_M = 0.12f
        const val MIN_DEPTH_M = 0.2f
        const val MAX_DEPTH_M = 3.5f
        const val MAX_ITEMS = 3000
        /** 深度と壁がこれ以上離れていたら、壁に寄せない（棚の手前に出ている商品など）。 */
        const val SNAP_TO_WALL_M = 0.3f

        /** 世界座標の点を、写した画像の正立座標 (u, v) と、カメラからの奥行き (m) にする。カメラの後ろなら null。 */
        fun project(p: Vec3, g: CaptureGeometry): FloatArray? {
            val m = g.cameraToWorld
            val dx = p.x - m[12]; val dy = p.y - m[13]; val dz = p.z - m[14]
            // 回転の逆は転置（列優先なので、列 i がカメラの軸 i）
            val x = m[0] * dx + m[1] * dy + m[2] * dz
            val y = m[4] * dx + m[5] * dy + m[6] * dz
            val z = m[8] * dx + m[9] * dy + m[10] * dz
            val depth = -z
            if (depth <= 0f) return null
            val sx = (g.focalX * x / depth + g.principalX) / g.imageWidth
            val sy = (-g.focalY * y / depth + g.principalY) / g.imageHeight
            if (sx !in 0f..1f || sy !in 0f..1f) return null
            val (u, v) = sensorToUpright(sx, sy, g.uprightRotationDegrees)
            return floatArrayOf(u, v, depth)
        }

        /** センサー画像の正規化座標を、正立画像の正規化座標へ（Unprojector.uprightToSensor の逆）。 */
        fun sensorToUpright(sx: Float, sy: Float, rotation: Int): Pair<Float, Float> = when (rotation) {
            90 -> 1f - sy to sx
            180 -> 1f - sx to 1f - sy
            270 -> sy to 1f - sx
            else -> sx to sy
        }

        /** 正立画像の焦点距離（ピクセル）と大きさ。 */
        private fun upright(g: CaptureGeometry): FloatArray {
            val r = g.uprightRotationDegrees % 180 != 0
            return if (r) floatArrayOf(g.focalY, g.focalX, g.imageHeight.toFloat(), g.imageWidth.toFloat())
            else floatArrayOf(g.focalX, g.focalY, g.imageWidth.toFloat(), g.imageHeight.toFloat())
        }

        /** 画像の上の大きさ（正規化）を、奥行き [depth] での実際の大きさ (m) にする。 */
        fun sizeInMeters(w: Float, h: Float, depth: Float, g: CaptureGeometry): Pair<Float, Float> {
            val (fu, fv, iw, ih) = upright(g).let { listOf(it[0], it[1], it[2], it[3]) }
            return w * iw * depth / fu to h * ih * depth / fv
        }

        fun sizeInImage(wM: Float, hM: Float, depth: Float, g: CaptureGeometry): Pair<Float, Float> {
            val (fu, fv, iw, ih) = upright(g).let { listOf(it[0], it[1], it[2], it[3]) }
            return wM * fu / depth / iw to hM * fv / depth / ih
        }

        /**
         * 商品の位置を決める。深度があればその点を、近くに壁があれば壁の上へ寄せる。深度がなければ、
         * カメラから商品への向きを壁と交わらせる。どちらもなければ置けない（null）。
         */
        fun place(camera: Vec3, along: Vec3, depthPoint: Vec3?, walls: List<Wall>): Vec3? {
            if (depthPoint != null) {
                val w = walls.filter { abs(distance(it, depthPoint)) < SNAP_TO_WALL_M && near(it, depthPoint) }.minByOrNull { abs(distance(it, depthPoint)) }
                    ?: return depthPoint
                return depthPoint.minus(w.normal.times(distance(w, depthPoint)))
            }
            val d = along.minus(camera).normalized()
            return walls.mapNotNull { w ->
                val denom = dot(w.normal, d)
                if (abs(denom) < 0.2f) return@mapNotNull null   // 壁とほぼ平行
                val t = dot(w.normal, w.center.minus(camera)) / denom
                if (t !in MIN_DEPTH_M..MAX_DEPTH_M) return@mapNotNull null
                camera.plus(d.times(t)).takeIf { near(w, it) }?.let { t to it }
            }.minByOrNull { it.first }?.second
        }

        private fun distance(w: Wall, p: Vec3) = dot(w.normal, p.minus(w.center))
        private fun near(w: Wall, p: Vec3) = p.minus(w.center).length() < w.radius + 0.5f
        private fun dot(a: Vec3, b: Vec3) = a.x * b.x + a.y * b.y + a.z * b.z
        private fun Vec3.minus(o: Vec3) = Vec3(x - o.x, y - o.y, z - o.z)
        private fun Vec3.plus(o: Vec3) = Vec3(x + o.x, y + o.y, z + o.z)
        private fun Vec3.times(k: Float) = Vec3(x * k, y * k, z * k)
        private fun Vec3.length() = sqrt(x * x + y * y + z * z)
        private fun Vec3.normalized() = times(1f / length().coerceAtLeast(1e-6f))
    }
}
