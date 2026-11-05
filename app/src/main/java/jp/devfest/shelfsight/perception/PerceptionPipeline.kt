package jp.devfest.shelfsight.perception

import android.os.SystemClock
import android.util.Log
import com.google.ar.core.Camera
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import java.io.File
import java.io.IOException
import java.io.PrintWriter
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.CapturedFrame
import jp.devfest.shelfsight.ar.FrameCapture
import jp.devfest.shelfsight.ar.JpegEncoder
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

/** 画面の正規化座標で表した矩形。 */
data class ViewRect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width get() = right - left
    val height get() = bottom - top
}

/** 画面の正規化座標で表した点。 */
data class ViewPoint(val x: Float, val y: Float)

/** 画面に描く手。[points] は MediaPipe の 21 点の順に並ぶ。 */
data class HandView(
    val points: List<ViewPoint>,
    /** 指さしのとき、指先と、その指す先を結ぶ線の終点。 */
    val pointingFrom: ViewPoint?,
    val pointingTo: ViewPoint?,
    /** 指の先で深度と交わった点。見つからなければ null。 */
    val pointingTarget: ViewPoint? = null,
)

data class PerceptionState(
    val ahead: String = "前方を確認しています",
    val people: String = "",
    val personRects: List<ViewRect> = emptyList(),
    val hands: String = "",
    val handViews: List<HandView> = emptyList(),
    /** 今回の検出で地図に置けた人と、距離が分からず置けなかった人の数。 */
    val placedPeople: Int = 0,
    val unplacedPeople: Int = 0,
)

/**
 * カメラ画像から人を検出し、正面にあるものを判定する。
 *
 * 描画を止めないよう、撮影だけを GL スレッドで行い、検出はワーカースレッドで行う。
 * 手は速く動き、前のフレームからの追跡が途切れると見失うので、人の検出とは別のスレッドで頻繁に処理する。
 */
/** @param isCurrent 今も見えていれば true。見失った直後は false。 */
data class VisibleMarker(val marker: PersonMarker, val isCurrent: Boolean)

class PerceptionPipeline internal constructor(
    private val detector: PersonDetector,
    private val handDetector: HandDetector,
    /** logcat はすぐ流れるので、判定結果をここにも残す。null なら記録しない。 */
    logFile: File? = null,
) {
    private val _state = MutableStateFlow(PerceptionState())
    val state: StateFlow<PerceptionState> = _state.asStateFlow()

    @Volatile var uprightRotationDegrees = 90

    /** 手のスレッドから、検出した手を知らせる。商品を手に取ったかの判定に使う。 */
    @Volatile internal var onHandsDetected: ((Double, List<HandInfo>) -> Unit)? = null

    /** GL スレッドが読む。追跡中の人と、最後に見えた時刻（SystemClock.elapsedRealtime 基準）。 */
    @Volatile private var liveMarkers: List<Pair<PersonMarker, Long>> = emptyList()

    private val worker = Executors.newSingleThreadExecutor { Thread(it, "perception") }
    private val handWorker = Executors.newSingleThreadExecutor { Thread(it, "hand-tracking") }
    private val busy = AtomicBoolean(false)
    private val handBusy = AtomicBoolean(false)
    private val tracker = PersonTracker()
    private val handMemory = HandBoxMemory()
    private val eventLog: PrintWriter? = logFile?.let { file ->
        try {
            PrintWriter(file).apply {
                println("kind,timeSeconds,millis,rawPeople,afterHands,finalPeople,lowerOnly,rememberedHands,hands,boxes")
            }
        } catch (e: IOException) {
            Log.w(TAG, "Could not open perception log", e)
            null
        }
    }
    private var lastCaptureMillis = 0L
    private var lastHandCaptureMillis = 0L
    @Volatile private var detectorFailed = false
    private var handDetectorFailed = false

    // 手のスレッドのみが書き込む。
    @Volatile private var handRuns = 0
    @Volatile private var totalHandMillis = 0L
    @Volatile private var handFrames = 0
    @Volatile private var pointingFrames = 0
    private var lastHandSeenSeconds = Double.NEGATIVE_INFINITY
    private var lastGesture: HandGesture? = null

    // 人のスレッドのみが触る集計。
    private var detections = 0
    private var totalDetectMillis = 0L
    private var removedAsOwnHand = 0
    private var removedAsLowerOnly = 0
    private var framesWithoutDepth = 0
    private var lastStatsLogMillis = 0L
    private val aheadCounts = HashMap<AheadKind, Int>()

    /** 画面表示の対応。センサー画像の正規化座標を、画面の正規化座標へ変換する。 */
    private class ViewMapping(private val corners: FloatArray) {
        fun map(sensorX: Float, sensorY: Float): Pair<Float, Float> = Pair(
            corners[0] + (corners[2] - corners[0]) * sensorX + (corners[4] - corners[0]) * sensorY,
            corners[1] + (corners[3] - corners[1]) * sensorX + (corners[5] - corners[1]) * sensorY,
        )
    }

    /** GL スレッドから毎フレーム呼ぶ。 */
    fun onFrame(frame: Frame, camera: Camera) {
        if (camera.trackingState != TrackingState.TRACKING) return
        val now = SystemClock.elapsedRealtime()
        val personDue = !detectorFailed && !busy.get() && now - lastCaptureMillis >= PERSON_INTERVAL_MS
        val handDue = !handDetectorFailed && !handBusy.get() && now - lastHandCaptureMillis >= HAND_INTERVAL_MS
        if (!personDue && !handDue) return

        val captured = FrameCapture.capture(frame, camera, uprightRotationDegrees, depthEnabled = true) ?: return
        val corners = FloatArray(6)
        frame.transformCoordinates2d(
            Coordinates2d.IMAGE_NORMALIZED, floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f),
            Coordinates2d.VIEW_NORMALIZED, corners,
        )
        val mapping = ViewMapping(corners)
        val timeSeconds = frame.timestamp / 1e9

        if (handDue) {
            lastHandCaptureMillis = now
            handBusy.set(true)
            handWorker.execute {
                try {
                    processHands(captured, mapping, timeSeconds)
                } catch (e: RuntimeException) {
                    Log.e(TAG, "Hand tracking failed", e)
                    handDetectorFailed = true
                    _state.update { it.copy(hands = "手の検出を開始できませんでした", handViews = emptyList()) }
                } finally {
                    handBusy.set(false)
                }
            }
        }
        if (personDue) {
            val pose = camera.pose
            val cameraPosition = Vec3(pose.tx(), pose.ty(), pose.tz())
            lastCaptureMillis = now
            busy.set(true)
            worker.execute {
                try {
                    process(captured, mapping, cameraPosition, timeSeconds)
                } catch (e: RuntimeException) {
                    // モデルを読み込めない場合など。地図の記録は続けられるので、アプリは止めない。
                    Log.e(TAG, "Person detection failed", e)
                    detectorFailed = true
                    _state.update { it.copy(ahead = "人の検出を開始できませんでした") }
                } finally {
                    busy.set(false)
                }
            }
        }
    }

    fun close() {
        worker.execute { detector.close() }
        worker.shutdown()
        handWorker.execute { handDetector.close() }
        handWorker.shutdown()
    }

    /** 地図に描く人。検出が止まっても古い位置を残さないよう、見失ってから一定時間で消す。 */
    fun visibleMarkers(nowMillis: Long = SystemClock.elapsedRealtime()): List<VisibleMarker> =
        liveMarkers.mapNotNull { (marker, seenMillis) ->
            val ageMillis = nowMillis - seenMillis
            if (ageMillis > MARKER_HOLD_MS) return@mapNotNull null
            VisibleMarker(marker, isCurrent = ageMillis <= MARKER_CURRENT_MS)
        }

    /** 手のスレッドで実行する。 */
    private fun processHands(frame: CapturedFrame, mapping: ViewMapping, timeSeconds: Double) {
        val startMillis = SystemClock.elapsedRealtime()
        val bitmap = JpegEncoder.toUprightBitmap(frame)
        val detected = try {
            handDetector.detect(bitmap, (timeSeconds * 1000).toLong())
        } finally {
            bitmap.recycle()
        }
        val geometry = frame.geometry
        val hands = detected.map { HandAnalysis.analyze(it, geometry) }
        onHandsDetected?.invoke(timeSeconds, hands)
        val handBoxes = hands.map { it.box }
        handMemory.update(timeSeconds, handBoxes)
        val millis = SystemClock.elapsedRealtime() - startMillis

        handRuns++
        totalHandMillis += millis
        if (hands.isNotEmpty()) {
            handFrames++
            lastHandSeenSeconds = timeSeconds
            lastGesture = hands.first().gesture
        }
        hands.firstOrNull { it.gesture == HandGesture.POINTING }?.let { pointing ->
            pointingFrames++
            Log.d(
                TAG,
                "pointing tip=${pointing.fingertipWorld} direction=${pointing.pointingWorld} " +
                    "distanceM=${pointing.distanceM}",
            )
        }
        if (hands.isNotEmpty()) {
            val text = hands.joinToString("|") { "${if (it.landmarks.isRightHand) "R" else "L"}:${it.gesture}" }
            writeEvent("hand", timeSeconds, millis, "", text, handBoxes)
        }

        val text = if (hands.isEmpty() && timeSeconds - lastHandSeenSeconds < LOST_HAND_MESSAGE_SECONDS) {
            "手を見失っています（直前：${lastGesture?.labelJa ?: "不明"}）"
        } else {
            handsText(hands, geometry)
        }
        val views = hands.map { toHandView(it, targetOf(it, geometry), mapping, geometry) }
        _state.update { it.copy(hands = text, handViews = views) }
    }

    /** 人のスレッドで実行する。 */
    private fun process(frame: CapturedFrame, mapping: ViewMapping, camera: Vec3, timeSeconds: Double) {
        val startMillis = SystemClock.elapsedRealtime()
        val bitmap = JpegEncoder.toUprightBitmap(frame)
        val detectedPeople = try {
            detector.detect(bitmap)
        } finally {
            bitmap.recycle()
        }
        val detectMillis = SystemClock.elapsedRealtime() - startMillis

        val geometry = frame.geometry
        val handBoxes = handMemory.update(timeSeconds, emptyList())
        val afterHands = OwnHandFilter.withoutOwnHands(detectedPeople, handBoxes)
        val boxes = afterHands.filterNot(OwnHandFilter::isLowerOnly)
        val lowerOnly = afterHands.filter(OwnHandFilter::isLowerOnly)
        removedAsOwnHand += detectedPeople.size - afterHands.size
        removedAsLowerOnly += lowerOnly.size
        if (detectedPeople.isNotEmpty()) {
            writeEvent(
                "person", timeSeconds, detectMillis,
                "${detectedPeople.size},${afterHands.size},${boxes.size},${lowerOnly.size},${handBoxes.size}",
                "", detectedPeople,
            )
        }
        val observations = boxes.map { box ->
            val distance = PersonDistance.estimate(box, geometry)
            val position = distance?.let {
                Unprojector.worldAt((box.left + box.right) / 2f, (box.top + box.bottom) / 2f, it, geometry)
            }
            PersonObservation(box, distance, position)
        }
        val people = tracker.update(timeSeconds, observations, camera)
        val nowMillis = SystemClock.elapsedRealtime()
        liveMarkers = tracker.markers().map { marker ->
            marker to nowMillis - ((timeSeconds - marker.lastSeenSeconds) * 1000).toLong()
        }
        val placed = observations.count { it.position != null }
        // 手の範囲は障害物に数えない。画面下部だけの枠は人と断定できないので、深度の判定には残す。
        val samples = ObstacleClassifier.collectSamples(geometry, detectedPeople - lowerOnly.toSet() + handBoxes)
        val ahead = ObstacleClassifier.classify(samples, camera.y, people)

        val rects = boxes.map { toViewRect(it, mapping, geometry.uprightRotationDegrees) }
        _state.update {
            it.copy(
                ahead = aheadText(ahead),
                people = peopleText(people),
                personRects = rects,
                placedPeople = placed,
                unplacedPeople = observations.size - placed,
            )
        }
        logStats(detectMillis, geometry.depth != null, ahead.kind, boxes.size)
    }

    /** 誤認の原因を後から確認できるよう、人や手の枠をファイルに残す。 */
    private fun writeEvent(
        kind: String,
        timeSeconds: Double,
        millis: Long,
        personCounts: String,
        hands: String,
        boxes: List<NormalizedBox>,
    ) {
        val log = eventLog ?: return
        val boxText = boxes.joinToString("|") {
            String.format(Locale.US, "%.2f %.2f %.2f %.2f", it.left, it.top, it.right, it.bottom)
        }
        val counts = personCounts.ifEmpty { ",,,," }
        val time = String.format(Locale.US, "%.2f", timeSeconds)
        synchronized(log) {
            log.println("$kind,$time,$millis,$counts,$hands,$boxText")
            log.flush()
        }
    }

    private fun targetOf(hand: HandInfo, geometry: CaptureGeometry): PointingHit? {
        if (hand.gesture != HandGesture.POINTING) return null
        val tip = hand.fingertipWorld ?: return null
        return PointingTarget.find(tip, hand.pointingWorld, geometry)
    }

    private fun toHandView(
        hand: HandInfo,
        target: PointingHit?,
        mapping: ViewMapping,
        geometry: CaptureGeometry,
    ): HandView {
        val rotation = geometry.uprightRotationDegrees
        fun view(u: Float, v: Float): ViewPoint {
            val (sensorX, sensorY) = Unprojector.uprightToSensor(u, v, rotation)
            val (x, y) = mapping.map(sensorX, sensorY)
            return ViewPoint(x, y)
        }
        val image = hand.landmarks.imagePoints
        if (hand.gesture != HandGesture.POINTING) {
            return HandView(image.map { view(it.x, it.y) }, null, null)
        }
        val tip = image[HandAnalysis.INDEX_TIP]
        val base = image[HandAnalysis.INDEX_MCP]
        val endU = tip.x + (tip.x - base.x) * POINTING_LINE_SCALE
        val endV = tip.y + (tip.y - base.y) * POINTING_LINE_SCALE
        val targetPoint = target?.let { view(it.u, it.v) }
        return HandView(image.map { view(it.x, it.y) }, view(tip.x, tip.y), targetPoint ?: view(endU, endV), targetPoint)
    }

    private fun handsText(hands: List<HandInfo>, geometry: CaptureGeometry): String =
        hands.joinToString("\n") { hand ->
            val side = if (hand.landmarks.isRightHand) "右手" else "左手"
            val palm = if (hand.palmFacesCamera) "手のひら" else "手の甲"
            val distance = hand.distanceM?.let { formatMeters(it) } ?: "距離不明"
            val direction = HandAnalysis.directionLabel(hand.indexDirection)
            val target = when {
                hand.gesture != HandGesture.POINTING -> ""
                else -> targetOf(hand, geometry)?.let { "\n指す先：カメラから ${formatMeters(it.depthM)}" }
                    ?: "\n指す先：確認できません"
            }
            "$side ${hand.gesture.labelJa}：$direction・$distance・$palm$target"
        }

    private fun toViewRect(box: NormalizedBox, mapping: ViewMapping, rotation: Int): ViewRect {
        val a = Unprojector.uprightToSensor(box.left, box.top, rotation)
        val b = Unprojector.uprightToSensor(box.right, box.bottom, rotation)
        val (ax, ay) = mapping.map(a.first, a.second)
        val (bx, by) = mapping.map(b.first, b.second)
        return ViewRect(minOf(ax, bx), minOf(ay, by), maxOf(ax, bx), maxOf(ay, by))
    }

    private fun aheadText(ahead: AheadResult): String {
        if (ahead.kind == AheadKind.UNKNOWN) return "前方を確認できません"
        val distance = ahead.distanceM?.let { formatMeters(it) } ?: "距離不明"
        val motion = ahead.motion?.let { "（${it.labelJa}）" }.orEmpty()
        return "前方 $distance：${ahead.kind.labelJa}$motion"
    }

    private fun peopleText(people: List<TrackedPerson>): String {
        if (people.isEmpty()) return ""
        val details = people
            .sortedBy { it.distanceM ?: Float.MAX_VALUE }
            .joinToString("、") { person ->
                val distance = person.distanceM?.let { formatMeters(it) } ?: "距離不明"
                "$distance ${person.motion.labelJa}"
            }
        return "人 ${people.size}人：$details"
    }

    private fun formatMeters(meters: Float) = String.format(Locale.JAPAN, "%.1fm", meters)

    private fun logStats(detectMillis: Long, hasDepth: Boolean, kind: AheadKind, personCount: Int) {
        detections++
        totalDetectMillis += detectMillis
        if (!hasDepth) framesWithoutDepth++
        aheadCounts[kind] = (aheadCounts[kind] ?: 0) + 1

        val now = SystemClock.elapsedRealtime()
        if (now - lastStatsLogMillis < STATS_LOG_INTERVAL_MS) return
        lastStatsLogMillis = now
        Log.i(
            TAG,
            "stats detections=$detections avgDetectMs=${totalDetectMillis / detections} " +
                "handRuns=$handRuns avgHandMs=${if (handRuns > 0) totalHandMillis / handRuns else 0} " +
                "handFrames=$handFrames " +
                "pointingFrames=$pointingFrames removedAsOwnHand=$removedAsOwnHand " +
                "removedAsLowerOnly=$removedAsLowerOnly " +
                "noDepth=$framesWithoutDepth ahead=$aheadCounts lastPersons=$personCount",
        )
    }

    private companion object {
        const val TAG = "PerceptionPipeline"
        const val PERSON_INTERVAL_MS = 150L
        const val MARKER_HOLD_MS = 2_000L
        const val MARKER_CURRENT_MS = 500L
        const val HAND_INTERVAL_MS = 50L
        const val LOST_HAND_MESSAGE_SECONDS = 2.0
        const val STATS_LOG_INTERVAL_MS = 10_000L
        const val POINTING_LINE_SCALE = 2f
    }
}
