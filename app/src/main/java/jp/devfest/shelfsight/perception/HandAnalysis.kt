package jp.devfest.shelfsight.perception

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * MediaPipe が返す手の 21 点。
 *
 * @param imagePoints 正立画像の正規化座標（x, y のみ使用）
 * @param worldPoints 手の中心を原点とするメートル単位の座標。+X が画像の右、+Y が画像の下、+Z がカメラの奥
 * @param isRightHand 実際の右手なら true
 */
internal class HandLandmarks(
    val imagePoints: List<Vec3>,
    val worldPoints: List<Vec3>,
    val isRightHand: Boolean,
)

enum class HandGesture(val labelJa: String) {
    POINTING("指さし"),
    OPEN("開いた手"),
    FIST("握った手"),
    OTHER("その他の形"),
}

internal data class HandInfo(
    val landmarks: HandLandmarks,
    val box: NormalizedBox,
    val gesture: HandGesture,
    /** 人差し指の付け根から指先への単位ベクトル。軸は [HandLandmarks.worldPoints] と同じ。 */
    val indexDirection: Vec3,
    /** 手のひらがカメラ側を向いていれば true。 */
    val palmFacesCamera: Boolean,
    /** カメラから手までの距離の推定値。 */
    val distanceM: Float?,
    /** 人差し指の指先のワールド座標。 */
    val fingertipWorld: Vec3?,
    /** 指の向きを ARCore のワールド座標で表した単位ベクトル。 */
    val pointingWorld: Vec3,
)

/** 手の 21 点から、形、向き、3D の位置を求める。 */
internal object HandAnalysis {
    const val WRIST = 0
    const val INDEX_MCP = 5
    const val INDEX_TIP = 8
    private const val MIDDLE_MCP = 9
    private const val PINKY_MCP = 17
    private val FINGERS = listOf(5, 9, 13, 17) // 人差し指、中指、薬指、小指の付け根

    private const val STRAIGHT_COS = 0.7f
    private const val MIN_DISTANCE_M = 0.1f
    private const val MAX_DISTANCE_M = 1.5f
    private const val BOX_PADDING = 0.03f

    fun analyze(hand: HandLandmarks, geometry: CaptureGeometry): HandInfo {
        val world = hand.worldPoints
        val direction = normalized(world[INDEX_TIP] - world[INDEX_MCP])
        val distance = estimateDistance(hand, geometry)
        val tip = hand.imagePoints[INDEX_TIP]
        return HandInfo(
            landmarks = hand,
            box = boxOf(hand),
            gesture = gestureOf(world),
            indexDirection = direction,
            palmFacesCamera = palmFacesCamera(hand),
            distanceM = distance,
            fingertipWorld = distance?.let { Unprojector.worldAt(tip.x, tip.y, it, geometry) },
            pointingWorld = toWorldDirection(direction, geometry),
        )
    }

    fun gestureOf(world: List<Vec3>): HandGesture {
        val extended = FINGERS.map { isExtended(world, it) }
        val count = extended.count { it }
        return when {
            extended[0] && count == 1 -> HandGesture.POINTING
            count == FINGERS.size -> HandGesture.OPEN
            count == 0 -> HandGesture.FIST
            else -> HandGesture.OTHER
        }
    }

    /** 付け根から第二関節、第二関節から指先までが同じ向きで、指先が手首から遠ければ伸びている。 */
    private fun isExtended(world: List<Vec3>, mcpIndex: Int): Boolean {
        val mcp = world[mcpIndex]
        val pip = world[mcpIndex + 1]
        val tip = world[mcpIndex + 3]
        val wrist = world[WRIST]
        val straight = dot(normalized(pip - mcp), normalized(tip - pip)) > STRAIGHT_COS
        return straight && tip.distanceTo(wrist) > pip.distanceTo(wrist)
    }

    /**
     * 手のひらの法線の向きで判定する。カメラから見て手のひらが見える右手では、
     * （手首→人差し指の付け根）×（手首→小指の付け根）がカメラ側（-Z）を向く。左手は逆になる。
     */
    private fun palmFacesCamera(hand: HandLandmarks): Boolean {
        val world = hand.worldPoints
        val normal = cross(world[INDEX_MCP] - world[WRIST], world[PINKY_MCP] - world[WRIST])
        return if (hand.isRightHand) normal.z < 0f else normal.z > 0f
    }

    /**
     * 手の実寸と画像上の大きさの比から距離を求める。
     *
     * 近い手は ARCore の深度が取れないことが多いので、深度を使わない。
     * 手が傾いていても崩れないよう、実寸も画像面内の成分だけで比べる。
     */
    private fun estimateDistance(hand: HandLandmarks, geometry: CaptureGeometry): Float? {
        val rotated = geometry.uprightRotationDegrees % 180 != 0
        val uprightWidth = if (rotated) geometry.imageHeight else geometry.imageWidth
        val uprightHeight = if (rotated) geometry.imageWidth else geometry.imageHeight
        val focal = (geometry.focalX + geometry.focalY) / 2f

        val estimates = PALM_EDGES.mapNotNull { (a, b) ->
            val dx3 = hand.worldPoints[a].x - hand.worldPoints[b].x
            val dy3 = hand.worldPoints[a].y - hand.worldPoints[b].y
            val dx = (hand.imagePoints[a].x - hand.imagePoints[b].x) * uprightWidth
            val dy = (hand.imagePoints[a].y - hand.imagePoints[b].y) * uprightHeight
            val pixels = sqrt(dx * dx + dy * dy)
            if (pixels < 1f) null else focal * sqrt(dx3 * dx3 + dy3 * dy3) / pixels
        }.sorted()
        if (estimates.isEmpty()) return null
        return estimates[estimates.size / 2].takeIf { it in MIN_DISTANCE_M..MAX_DISTANCE_M }
    }

    private val PALM_EDGES = listOf(
        WRIST to INDEX_MCP,
        WRIST to MIDDLE_MCP,
        WRIST to PINKY_MCP,
        INDEX_MCP to PINKY_MCP,
    )

    /** 正立画像基準の向きを、ARCore のカメラ座標を経由してワールド座標へ変換する。 */
    fun toWorldDirection(direction: Vec3, geometry: CaptureGeometry): Vec3 {
        val (sensorX, sensorY) = when (geometry.uprightRotationDegrees) {
            90 -> direction.y to -direction.x
            180 -> -direction.x to -direction.y
            270 -> -direction.y to direction.x
            else -> direction.x to direction.y
        }
        // ARCore のカメラ座標は +X が画像の右、+Y が画像の上、-Z が奥。
        val cx = sensorX
        val cy = -sensorY
        val cz = -direction.z
        val m = geometry.cameraToWorld
        return Vec3(
            m[0] * cx + m[4] * cy + m[8] * cz,
            m[1] * cx + m[5] * cy + m[9] * cz,
            m[2] * cx + m[6] * cy + m[10] * cz,
        )
    }

    private fun boxOf(hand: HandLandmarks): NormalizedBox {
        val xs = hand.imagePoints.map { it.x }
        val ys = hand.imagePoints.map { it.y }
        return NormalizedBox(
            left = (xs.min() - BOX_PADDING).coerceIn(0f, 1f),
            top = (ys.min() - BOX_PADDING).coerceIn(0f, 1f),
            right = (xs.max() + BOX_PADDING).coerceIn(0f, 1f),
            bottom = (ys.max() + BOX_PADDING).coerceIn(0f, 1f),
        )
    }

    /** 指す向きを言葉にする。軸は [HandLandmarks.worldPoints] と同じ。 */
    fun directionLabel(direction: Vec3): String {
        if (direction.z > 0.7f) return "前方"
        if (direction.z < -0.7f) return "手前"
        val horizontal = when {
            direction.x > 0.38f -> "右"
            direction.x < -0.38f -> "左"
            else -> ""
        }
        val vertical = when {
            direction.y < -0.38f -> "上"
            direction.y > 0.38f -> "下"
            else -> ""
        }
        return (horizontal + vertical).ifEmpty { "前方" }
    }

    private operator fun Vec3.minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)

    private fun dot(a: Vec3, b: Vec3) = a.x * b.x + a.y * b.y + a.z * b.z

    private fun cross(a: Vec3, b: Vec3) = Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )

    private fun normalized(v: Vec3): Vec3 {
        val length = sqrt(dot(v, v))
        return if (length < 1e-6f) Vec3(0f, 0f, 0f) else Vec3(v.x / length, v.y / length, v.z / length)
    }
}

/** 手を一瞬見失っても誤認を防げるよう、直前に手があった場所を覚えておく。 */
internal class HandBoxMemory(private val holdSeconds: Double = DEFAULT_HOLD_SECONDS) {
    private class Entry(val timeSeconds: Double, val box: NormalizedBox)

    private val entries = ArrayList<Entry>()

    /** [boxes] を記録し、保持時間内に見えた手の範囲をすべて返す。手と人のスレッドから呼ばれる。 */
    @Synchronized
    fun update(timeSeconds: Double, boxes: List<NormalizedBox>): List<NormalizedBox> {
        entries.removeAll { timeSeconds - it.timeSeconds > holdSeconds }
        boxes.forEach { entries += Entry(timeSeconds, it) }
        return entries.map { it.box }
    }

    private companion object {
        const val DEFAULT_HOLD_SECONDS = 0.7
    }
}

/** 人の検出器が、装着者の手や腕を人と誤認した結果を取り除く。 */
internal object OwnHandFilter {
    /**
     * 人の枠のうち手が占める割合の下限。全身が写った他人なら、手はごく一部にしかならない。
     * 手や腕だけを人と誤認した枠では、手が大きな割合を占める。
     */
    private const val MIN_HAND_SHARE_OF_PERSON = 0.12f
    private const val MIN_HAND_INSIDE = 0.7f

    /** 画面のこれより下だけに写る枠は、胸の高さのカメラで見た立っている人の頭を含まない。 */
    private const val MIN_TOP_OF_LOWER_ONLY_BOX = 0.55f
    private const val BOTTOM_EDGE = 0.98f

    /**
     * 画面の下端から立ち上がり、上半分に届かない枠。手や腕の可能性が高い。
     *
     * しゃがんだ人や子ども、カメラが下を向いた場合も当てはまるので、
     * 呼び出し側は人としては扱わないが、深度による障害物の判定には残す。
     */
    fun isLowerOnly(person: NormalizedBox): Boolean =
        person.bottom >= BOTTOM_EDGE && person.top >= MIN_TOP_OF_LOWER_ONLY_BOX

    fun withoutOwnHands(people: List<NormalizedBox>, hands: List<NormalizedBox>): List<NormalizedBox> =
        people.filterNot { person -> hands.any { isHandOnly(person, it) } }

    private fun isHandOnly(person: NormalizedBox, hand: NormalizedBox): Boolean {
        val handArea = areaOf(hand)
        val personArea = areaOf(person)
        if (handArea <= 0f || personArea <= 0f) return false
        val inside = areaOf(intersection(person, hand)) / handArea
        return inside >= MIN_HAND_INSIDE && handArea / personArea >= MIN_HAND_SHARE_OF_PERSON
    }

    private fun intersection(a: NormalizedBox, b: NormalizedBox) = NormalizedBox(
        left = maxOf(a.left, b.left),
        top = maxOf(a.top, b.top),
        right = minOf(a.right, b.right),
        bottom = minOf(a.bottom, b.bottom),
    )

    private fun areaOf(box: NormalizedBox): Float =
        abs(maxOf(0f, box.right - box.left) * maxOf(0f, box.bottom - box.top))
}
