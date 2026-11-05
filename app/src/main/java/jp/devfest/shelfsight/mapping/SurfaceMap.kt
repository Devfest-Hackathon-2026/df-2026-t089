package jp.devfest.shelfsight.mapping

import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import java.nio.FloatBuffer
import jp.devfest.shelfsight.ar.GlUtil

/** ARCore の平面 1 枚。追跡が止まっても地図から消えないよう、最後の形と位置を保つ。 */
internal class SurfacePatch(val plane: Plane) {
    val worldFromPlane = FloatArray(16)
    var kind = SurfaceKind.FLOOR
    var polygon = FloatArray(0)
        private set
    var buffer: FloatBuffer = GlUtil.floatBuffer(polygon)
        private set
    var area = 0f
        private set

    val vertexCount: Int
        get() = polygon.size / 2

    val y: Float
        get() = worldFromPlane[13]

    fun readPose() = plane.centerPose.toMatrix(worldFromPlane, 0)

    fun readPolygon() {
        val source = plane.polygon
        source.rewind()
        val latest = FloatArray(source.remaining()).also { source.get(it) }
        if (latest.contentEquals(polygon)) return
        polygon = latest
        buffer = GlUtil.floatBuffer(
            FloatArray(latest.size / 2 * 3).also { xyz ->
                for (i in 0 until latest.size / 2) {
                    xyz[i * 3] = latest[i * 2]
                    xyz[i * 3 + 2] = latest[i * 2 + 1]
                }
            },
        )
        area = SurfaceLogic.polygonArea(latest)
    }
}

/**
 * ARCore の平面検出の結果を、地図の面として持つ。GL スレッドのみが触る。
 *
 * ARCore は重なった平面を 1 枚に統合し、位置の補正も平面ごとに反映するので、ここでは結果をなぞるだけでよい。
 */
internal class SurfaceMap {
    private val patches = LinkedHashMap<Plane, SurfacePatch>()

    val surfaces: Collection<SurfacePatch>
        get() = patches.values

    /** @param grow false なら面を増やさず形も変えない。位置の補正だけ反映する */
    fun update(session: Session, grow: Boolean) {
        if (grow) {
            addNewPlanes(session)
            patches.values.removeAll { it.plane.trackingState == TrackingState.STOPPED || it.plane.subsumedBy != null }
        }
        for (patch in patches.values) {
            if (patch.plane.trackingState != TrackingState.TRACKING) continue
            patch.readPose()
            if (grow) patch.readPolygon()
        }
        classify()
    }

    private fun addNewPlanes(session: Session) {
        for (plane in session.getAllTrackables(Plane::class.java)) {
            if (plane in patches) continue
            if (plane.trackingState != TrackingState.TRACKING || plane.subsumedBy != null) continue
            if (plane.type == Plane.Type.HORIZONTAL_DOWNWARD_FACING) continue // 天井
            val patch = SurfacePatch(plane)
            patch.readPolygon()
            if (patch.area < SurfaceLogic.MIN_AREA_M2) continue // 育ったら次の更新で追加される
            patch.readPose()
            patches[plane] = patch
        }
    }

    private fun classify() {
        val floorY = SurfaceLogic.floorHeight(
            patches.values.filter { it.plane.type != Plane.Type.VERTICAL }.map { it.y to it.area },
        )
        for (patch in patches.values) {
            patch.kind = if (patch.plane.type == Plane.Type.VERTICAL) {
                SurfaceKind.WALL
            } else {
                SurfaceLogic.horizontalKind(patch.y, floorY)
            }
        }
    }
}
