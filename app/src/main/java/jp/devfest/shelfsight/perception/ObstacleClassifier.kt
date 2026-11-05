package jp.devfest.shelfsight.perception

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.max
import kotlin.math.sqrt

enum class AheadKind(val labelJa: String) {
    PERSON("人"),
    WALL("壁や平らな面"),
    OBJECT("物"),
    UNKNOWN("確認できません"),
}

internal data class AheadResult(
    val kind: AheadKind,
    val distanceM: Float?,
    val motion: PersonMotion? = null,
)

/** 前方の正面にある表面の点。 */
internal data class SurfaceSample(val position: Vec3, val depthM: Float)

/**
 * 正面の一番近いものが、人か、壁か、それ以外の物かを判定する。
 *
 * 人は検出器の結果で判断する。人以外の深度は 3D の点に戻し、縦の平面に並ぶなら壁とする。
 * データが足りないときは「何もない」とは言わず、[AheadKind.UNKNOWN] を返す。
 */
internal object ObstacleClassifier {
    private const val CORRIDOR_LEFT = 0.35f
    private const val CORRIDOR_RIGHT = 0.65f
    private const val CORRIDOR_TOP = 0.25f
    private const val CORRIDOR_BOTTOM = 0.75f
    private const val SAMPLE_GRID = 15
    private const val BOX_MARGIN = 0.05f

    private const val FLOOR_BELOW_CAMERA_M = 1.0f // 胸の高さから床までより少し短い
    private const val CEILING_ABOVE_CAMERA_M = 1.0f
    private const val MIN_SAMPLES = 12
    private const val NEAREST_PERCENTILE = 0.1f
    private const val SURFACE_THICKNESS_M = 0.6f
    private const val MIN_WALL_POINTS = 10
    private const val MIN_WALL_WIDTH_M = 0.2f
    private const val MIN_WALL_HEIGHT_M = 0.25f
    private const val MIN_PLANE_TOLERANCE_M = 0.05f
    private const val PLANE_TOLERANCE_PER_METER = 0.03f

    fun isInCorridor(box: NormalizedBox): Boolean =
        box.left < CORRIDOR_RIGHT && box.right > CORRIDOR_LEFT

    /** 正面の範囲の深度を、人の範囲を除いて取り出す。 */
    fun collectSamples(geometry: CaptureGeometry, personBoxes: List<NormalizedBox>): List<SurfaceSample> {
        val samples = ArrayList<SurfaceSample>(SAMPLE_GRID * SAMPLE_GRID)
        for (row in 0 until SAMPLE_GRID) {
            val v = CORRIDOR_TOP + (CORRIDOR_BOTTOM - CORRIDOR_TOP) * row / (SAMPLE_GRID - 1)
            for (col in 0 until SAMPLE_GRID) {
                val u = CORRIDOR_LEFT + (CORRIDOR_RIGHT - CORRIDOR_LEFT) * col / (SAMPLE_GRID - 1)
                if (personBoxes.any { it.containsWithMargin(u, v) }) continue
                val depth = Unprojector.depthAt(u, v, geometry) ?: continue
                samples += SurfaceSample(Unprojector.worldAt(u, v, depth, geometry), depth)
            }
        }
        return samples
    }

    private fun NormalizedBox.containsWithMargin(u: Float, v: Float) =
        u >= left - BOX_MARGIN && u <= right + BOX_MARGIN && v >= top - BOX_MARGIN && v <= bottom + BOX_MARGIN

    fun classify(samples: List<SurfaceSample>, cameraY: Float, people: List<TrackedPerson>): AheadResult {
        val surface = classifySurface(samples, cameraY)
        val person = people
            .filter { isInCorridor(it.box) }
            .minByOrNull { it.distanceM ?: Float.MAX_VALUE }
            ?: return surface
        val personDistance = person.distanceM
        val surfaceDistance = surface.distanceM
        // 距離が分からない人は、近い可能性があるものとして優先する。
        if (personDistance == null || surfaceDistance == null || personDistance <= surfaceDistance) {
            return AheadResult(AheadKind.PERSON, personDistance, person.motion)
        }
        return surface
    }

    private fun classifySurface(samples: List<SurfaceSample>, cameraY: Float): AheadResult {
        // 床と天井は、ぶつかるものではないので除く。
        val body = samples.filter {
            it.position.y > cameraY - FLOOR_BELOW_CAMERA_M && it.position.y < cameraY + CEILING_ABOVE_CAMERA_M
        }
        if (body.size < MIN_SAMPLES) return AheadResult(AheadKind.UNKNOWN, null)

        val sortedDepths = body.map { it.depthM }.sorted()
        val nearest = sortedDepths[(sortedDepths.size * NEAREST_PERCENTILE).toInt()]
        val front = body.filter { it.depthM <= nearest + SURFACE_THICKNESS_M }
        val kind = if (isVerticalPlane(front, nearest)) AheadKind.WALL else AheadKind.OBJECT
        return AheadResult(kind, nearest)
    }

    /** 真上から見て点が 1 本の線に並び、十分な幅と高さがあれば縦の平面とみなす。 */
    private fun isVerticalPlane(points: List<SurfaceSample>, distance: Float): Boolean {
        if (points.size < MIN_WALL_POINTS) return false
        val meanX = points.map { it.position.x }.average().toFloat()
        val meanZ = points.map { it.position.z }.average().toFloat()
        var xx = 0f
        var xz = 0f
        var zz = 0f
        for (p in points) {
            val dx = p.position.x - meanX
            val dz = p.position.z - meanZ
            xx += dx * dx
            xz += dx * dz
            zz += dz * dz
        }
        xx /= points.size
        xz /= points.size
        zz /= points.size

        // 2x2 共分散行列の固有値。小さい方が線からのばらつき、大きい方が線に沿った広がり。
        val halfTrace = (xx + zz) / 2f
        val root = sqrt(max(0f, (xx - zz) * (xx - zz) / 4f + xz * xz))
        val thickness = sqrt(max(0f, halfTrace - root))
        val spread = sqrt(max(0f, halfTrace + root))
        // 一様に並んだ点の範囲は、標準偏差のおよそ sqrt(12) 倍になる。
        val width = spread * 3.46f
        val ys = points.map { it.position.y }
        val height = ys.max() - ys.min()

        val tolerance = max(MIN_PLANE_TOLERANCE_M, distance * PLANE_TOLERANCE_PER_METER)
        return thickness < tolerance && width >= MIN_WALL_WIDTH_M && height >= MIN_WALL_HEIGHT_M
    }
}
