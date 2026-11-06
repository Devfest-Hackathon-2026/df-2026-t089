package jp.devfest.shelfsight.assist

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLES20.GL_COLOR_BUFFER_BIT
import android.opengl.GLES20.GL_DEPTH_BUFFER_BIT
import android.opengl.GLES20.glClear
import android.opengl.GLES20.glClearColor
import android.opengl.GLES20.glViewport
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.GestureDetector
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import com.example.store3d.CameraBackground
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.UnavailableException
import jp.devfest.shelfsight.BuildConfig
import jp.devfest.shelfsight.ar.CapturedFrame
import jp.devfest.shelfsight.ar.FrameCapture
import jp.devfest.shelfsight.ar.JpegEncoder
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.cloud.GeminiClient
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.perception.MediaPipePersonDetector
import jp.devfest.shelfsight.perception.PersonDistance
import jp.devfest.shelfsight.perception.PersonMotion
import jp.devfest.shelfsight.perception.PersonObservation
import jp.devfest.shelfsight.perception.PersonTracker
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sqrt

/**
 * 視覚障害のある人のための、コンビニの棚と商品の案内。胸の前にスマホを立てて持ち、イヤホンで聞く。
 *
 * （① 棚の種類の読み上げ（歩行中の判別、8 列 × 7 分類）は、遅いので止めている）
 * ② 棚が 1 つ写っている: 商品を認識し、LLM で何があるかを短く自然な文にして話す（場違いな商品は言わない）
 * ③ 商品が 1 つ大きく写っている: Cloud Vision API に聞き、社名と商品名を言う（1 回の接近で 1 回だけ呼ぶ）
 * ④ 人が近づいてくる: その人の方向からピコピコ鳴らす
 *
 * 操作
 * - 長押し: 「調べています」と言い、手前の商品 1 つなら ③、棚の広い範囲なら ②、その間なら「一歩下がって」と頼む
 * - 2 回たたく: 質問する（決まった言い方はその操作、それ以外は LLM が画像を見て答える）
 * TalkBack が有効なときは、ダブルタップが質問、ダブルタップ長押しが長押しになる。
 */
class AssistActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var gl: GLSurfaceView
    /** 画面に出す内部状態。 */
    private val debug = AssistDebug()
    private var session: Session? = null
    private var installRequested = false
    private var sensorRotation = 90
    private var viewW = 0
    private var viewH = 0
    private var geometryChanged = false
    private val background = CameraBackground()

    @Volatile private var listener: Listener? = null
    private val mixer by lazy { SpatialMixer { listener } }
    private val voice by lazy { Voice(this, mixer) }
    private val listening by lazy { Listening(this) }

    private val sceneWorker = Executors.newSingleThreadExecutor()
    private val personWorker = Executors.newSingleThreadExecutor()
    private val lookupWorker = Executors.newSingleThreadExecutor()
    private val sceneBusy = AtomicBoolean(false)
    private val personBusy = AtomicBoolean(false)
    /** 次のフレームを高解像度で撮って渡す先（長押し・質問）。 */
    @Volatile private var pendingCapture: ((CapturedFrame) -> Unit)? = null

    private val recognizer by lazy { ProductRecognizer(this) }
    private val personDetector by lazy { MediaPipePersonDetector(this) }
    private val tracker = PersonTracker()
    private val lookup = ProductLookup(BuildConfig.CLOUD_VISION_API_KEY)
    private val narrator = Narrator(
        BuildConfig.GEMINI_API_KEY.takeIf { it.isNotEmpty() }
            // 話す文を作るだけなので、速い lite のモデルから試す（flash は 6〜9 秒、lite は 2 秒ほど）
            ?.let { GeminiClient(it, BuildConfig.GEMINI_MODELS.split(",").map(String::trim).filter(String::isNotEmpty).sortedBy { m -> "lite" !in m }) },
    )
    private val judge = SceneJudge()
    /** 1 秒ごとの商品認識の結果を、AR 空間の壁の上に貯めたもの（アプリを閉じると消える）。 */
    private val shelfMap = ShelfMap()
    private val recorder by lazy { Recorder(java.io.File(getExternalFilesDir(null), "records")) }
    /** 商品のジャンル（26 種類）→ 歩行中の判別と同じ 7 分類。場違いな商品を外すのに使う。 */
    private val groups by lazy { ShelfDescriber.Groups.parse(assets.open("coarse/categories.json").bufferedReader().readText()) }

    private var lastScene = 0L
    private var lastPerson = 0L
    private var greeted = false
    @Volatile private var auto = true
    /** ② を話した場所。ここから離れるか、棚 1 つの場面を離れるまでは繰り返さない。 */
    private var describedAt: Vec3? = null
    private var leftOneShelfAt = 0L
    private var lastDescribeTry = 0L
    /** 今の接近で、大きく写った商品を最後に見た時刻と、そのときの場所。 */
    private var approachSeenAt = 0L
    private var approachFrom: Vec3? = null
    /** 質問に答えるときに LLM に渡す、最後に話した説明と、最後に調べた商品。 */
    @Volatile private var lastDescription = ""
    @Volatile private var lastProduct = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        gl = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@AssistActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        // 画面全体を 1 つのボタンにする（どこを触ってもよい）
        val gestures = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent) = true
            override fun onDoubleTap(e: MotionEvent): Boolean { startListening(); return true }
            override fun onLongPress(e: MotionEvent) { onLongPressed() }
        })
        val touch = object : View(this) {
            @android.annotation.SuppressLint("ClickableViewAccessibility")
            override fun onTouchEvent(event: MotionEvent): Boolean = gestures.onTouchEvent(event)
        }.apply {
            contentDescription = "長押しで調べる。2回たたくと質問"
            isClickable = true
            isLongClickable = true
            // TalkBack から: ダブルタップ＝クリック（質問）、ダブルタップ長押し＝長押し
            setOnClickListener { startListening() }
            setOnLongClickListener { onLongPressed(); true }
        }
        setContentView(FrameLayout(this).apply {
            addView(gl)
            addView(DebugOverlay(this@AssistActivity, debug), FrameLayout.LayoutParams(-1, -1))
            addView(touch, FrameLayout.LayoutParams(-1, -1))
        })
        updateSettings()
    }

    override fun onResume() {
        super.onResume()
        val missing = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
            .filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), 0)
            return
        }
        if (session == null) try {
            if (ArCoreApk.getInstance().requestInstall(this, !installRequested) == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                installRequested = true
                return
            }
            session = Session(this).also { s ->
                useHighResolutionCameraImage(s)
                val config = Config(s).setFocusMode(Config.FocusMode.AUTO).setPlaneFindingMode(Config.PlaneFindingMode.VERTICAL)
                if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) config.depthMode = Config.DepthMode.AUTOMATIC
                s.configure(config)
                val cm = getSystemService(CameraManager::class.java)
                sensorRotation = cm.getCameraCharacteristics(s.cameraConfig.cameraId).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            }
        } catch (e: UnavailableException) {
            voice.say("このスマホでは、カメラの位置合わせが使えません。")
            debug.tracking = "ARCore が使えません: $e"
            return
        }
        session!!.resume()
        gl.onResume()
        if (!greeted) {
            greeted = true
            voice.say("準備ができました。長押しで、商品や棚を調べます。2回たたくと、質問できます。")
            voice.prepare(listOf("調べています"))
        }
    }

    override fun onPause() {
        super.onPause()
        listening.cancel()
        gl.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        for (w in listOf(sceneWorker, personWorker, lookupWorker)) w.shutdownNow()
        listening.close()
        voice.close()
        mixer.close()
        session?.close()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.any { it != PackageManager.PERMISSION_GRANTED }) {
            voice.say("カメラとマイクの許可が必要です。設定から許可してください。")
        }
    }

    override fun onSurfaceCreated(g: GL10?, c: EGLConfig?) {
        glClearColor(0f, 0f, 0f, 1f)
        background.init()
    }

    override fun onSurfaceChanged(g: GL10?, w: Int, h: Int) {
        glViewport(0, 0, w, h)
        viewW = w; viewH = h; geometryChanged = true
    }

    override fun onDrawFrame(g: GL10?) {
        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        s.setCameraTextureName(background.texId)
        if (geometryChanged) { s.setDisplayGeometry(display!!.rotation, viewW, viewH); geometryChanged = false }
        val frame = try { s.update() } catch (e: Exception) { return }
        background.draw(frame)
        val camera = frame.camera
        // 画面の表示用: センサー画像の正規化座標が、画面のどこに来るか
        val corners = FloatArray(6)
        frame.transformCoordinates2d(Coordinates2d.IMAGE_NORMALIZED, floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f), Coordinates2d.VIEW_NORMALIZED, corners)
        debug.corners = corners
        debug.rotation = sensorRotation
        debug.tracking = when (camera.trackingState) {
            TrackingState.TRACKING -> "追跡中"
            TrackingState.PAUSED -> "一時停止（${camera.trackingFailureReason}）"
            else -> "停止"
        }
        if (camera.trackingState != TrackingState.TRACKING) return

        // 聞く人の向き: スマホを胸の前に立てて持つので、カメラの前（-z）を水平にしたもの
        val pose = camera.pose
        val z = pose.zAxis
        val fx = -z[0]
        val fz = -z[2]
        val len = sqrt(fx * fx + fz * fz)
        if (len > 0.3f) {
            listener = Listener(Vec3(pose.tx(), pose.ty(), pose.tz()), fx / len, fz / len)
            debug.headingDeg = Math.toDegrees(kotlin.math.atan2(fx.toDouble(), -fz.toDouble())).toFloat()
        }

        val now = SystemClock.elapsedRealtime()
        val rotation = sensorRotation   // 画面は縦に固定している
        pendingCapture?.let { target ->
            FrameCapture.capture(frame, camera, rotation, depthEnabled = false, fullResolution = true)?.let { f ->
                pendingCapture = null
                lookupWorker.execute { target(f) }
            }
        }
        if (now - lastPerson >= PERSON_INTERVAL_MS && personBusy.compareAndSet(false, true)) {
            val f = FrameCapture.capture(frame, camera, rotation, depthEnabled = true)
            if (f == null) personBusy.set(false) else {
                lastPerson = now
                personWorker.execute { try { runPeople(f) } finally { personBusy.set(false) } }
            }
        }
        if (now - lastScene >= SCENE_INTERVAL_MS && sceneBusy.compareAndSet(false, true)) {
            val f = FrameCapture.capture(frame, camera, rotation, depthEnabled = true, usualLongSide = SCENE_LONG_SIDE)
            if (f == null) sceneBusy.set(false) else {
                lastScene = now
                val walls = s.getAllTrackables(Plane::class.java)
                    .filter { it.type == Plane.Type.VERTICAL && it.trackingState == TrackingState.TRACKING && it.subsumedBy == null }
                    .map { pl ->
                        val c = pl.centerPose
                        val n = c.yAxis
                        Wall(Vec3(c.tx(), c.ty(), c.tz()), Vec3(n[0], n[1], n[2]), maxOf(pl.extentX, pl.extentZ) / 2)
                    }
                val heading = debug.headingDeg
                sceneWorker.execute { try { runScene(f, walls, heading) } finally { sceneBusy.set(false) } }
            }
        }
    }

    // ---------------------------------------------------------------- ①② 写っているものの判定

    /**
     * 1 秒ごと: 商品の枠を出して写っているものを判定し、商品に名前を付けて地図（壁の上）に置き、画像と姿勢と結果を記録する。
     */
    private fun runScene(f: CapturedFrame, walls: List<Wall>, heading: Float?) {
        val image = JpegEncoder.toUprightBitmap(f, SCENE_LONG_SIDE)
        try {
            val now = SystemClock.elapsedRealtime()
            val boxes = recognizer.detect(image).map { it.first }
            val detectMs = SystemClock.elapsedRealtime() - now
            val switched = judge.update(SceneEvidence(0, boxes))
            val products = recognizer.recognize(image, boxes)
            val recognizeMs = SystemClock.elapsedRealtime() - now - detectMs
            val placed = products.map { p -> Recorder.Product(p, placeOnMap(p, f, walls, now)) }
            recorder.record(System.currentTimeMillis(), image, f.geometry, heading, walls.size, placed)
            debug.productBoxes = boxes
            debug.scene = judge.scene
            debug.sceneMs = "枠 ${detectMs}ms / 名前 ${recognizeMs}ms / 壁 ${walls.size}"
            debug.sceneAt = now
            debug.mapView = shelfMap.visible(f.geometry)
            debug.mapInfo = "${shelfMap.size} 個（今見えている ${debug.mapView.size}、この 1 枚で置いた ${placed.count { it.position != null }}/${products.size}）"
            val main = SceneJudge.mainProduct(boxes)
            val camera = cameraPosition(f)

            // 新しい商品に近づいたら、前に調べた結果を忘れる（次の長押しで API を 1 回呼ぶ）。
            // 商品 1 つの場面に入ったとき、大きく写った商品がしばらく見えなかったとき、歩いて離れたとき
            val gone = approachSeenAt != 0L && now - approachSeenAt > APPROACH_GAP_MS
            val moved = (approachFrom?.let { horizontal(it, camera) } ?: 0f) > APPROACH_MOVE_M
            if ((switched && judge.scene == Scene.ONE_PRODUCT) || gone || moved) {
                lookup.newApproach()
                approachSeenAt = 0L
                approachFrom = null
            }
            if (main != null) {
                approachSeenAt = now
                if (approachFrom == null) approachFrom = camera
            }
            if (judge.scene != Scene.ONE_SHELF && leftOneShelfAt == 0L) leftOneShelfAt = now
            if (judge.scene == Scene.ONE_SHELF) leftOneShelfAt = 0L
            if (leftOneShelfAt != 0L && now - leftOneShelfAt > FORGET_DESCRIPTION_MS) describedAt = null

            when {
                !auto || voice.speaking -> Unit
                judge.scene == Scene.ONE_SHELF && now - lastDescribeTry > DESCRIBE_RETRY_MS &&
                    (describedAt?.distanceTo(camera) ?: Float.MAX_VALUE) > MOVE_TO_REDESCRIBE_M -> {
                    lastDescribeTry = now
                    // LLM を待つ間も 1 秒ごとの記録を止めないよう、別のスレッドで話す
                    val jpg = jpeg(image, NARRATE_LONG_SIDE)
                    val inView = debug.mapView
                    lookupWorker.execute { describe(inView, jpg, boxes.size, camera, requested = false) }
                }
            }
        } catch (e: RuntimeException) {
            Log.e(TAG, "scene analysis failed", e)
        } finally {
            image.recycle()
        }
    }

    /**
     * ② 棚 1 つの商品を説明する。[products] は地図から取り出した今見えている商品（なければその場で認識したもの）。
     * LLM で短い自然な文にする（使えなければ決まった言い方の文）。LLM を待つので、呼んだスレッドはしばらく止まる。
     */
    private fun describe(products: List<RecognizedProduct>, jpg: ByteArray, boxCount: Int, camera: Vec3?, requested: Boolean) {
        val t1 = SystemClock.elapsedRealtime()
        val split = ShelfDescriber.split(products, groups)
        debug.kept = split.kept
        debug.removed = split.removed
        debug.unnamed = (boxCount - products.size).coerceAtLeast(0)
        debug.mainGroup = "${split.mainGroup}（${split.mainKinds.joinToString("・")}）"
        debug.productsAt = t1
        val llm = if (narrator.available) narrator.describeShelf(split.kept, jpg) else null
        val text = llm ?: ShelfDescriber.describe(split.kept)
        val t2 = SystemClock.elapsedRealtime()
        debug.description = text ?: "（商品が少なく説明なし）"
        debug.describeInfo = "文 ${if (llm != null) "LLM" else "定型"} ${t2 - t1}ms"
        Log.i(TAG, "describe: ${products.size} products, kept ${split.kept.size}, removed ${split.removed.size} (${split.mainGroup}) -> $text")
        if (text != null) {
            if (camera != null) describedAt = camera
            lastDescription = text
            voice.say(text, interrupt = requested)
        } else if (requested) {
            voice.say("商品がよく分かりませんでした。棚の正面から、棚全体を写してください。")
        }
    }

    // ---------------------------------------------------------------- 長押し: 商品か棚を調べる

    private fun onLongPressed() {
        listening.cancel()
        voice.say("調べています", interrupt = true, remember = false)
        pendingCapture = ::runLongPress
    }

    /** 撮った画像で、手前の商品 1 つ（③）か、棚の広い範囲（②）か、その間（一歩下がってもらう）かを決める。 */
    private fun runLongPress(f: CapturedFrame) {
        val image = JpegEncoder.toUprightBitmap(f, LOOKUP_LONG_SIDE)
        try {
            val boxes = recognizer.detect(image).map { it.first }
            val (focus, productBox) = SceneJudge.focus(boxes)
            val largest = boxes.maxOfOrNull { SceneJudge.area(it) } ?: 0f
            Log.i(TAG, "long press: ${boxes.size} boxes, largest $largest -> $focus")
            debug.longPress = "$focus（枠 ${boxes.size} 個、いちばん大きい枠 ${(largest * 100).toInt()}%）"
            debug.focusBox = productBox
            debug.focusAt = SystemClock.elapsedRealtime()
            debug.productBoxes = boxes
            debug.sceneAt = debug.focusAt
            when (focus) {
                Focus.PRODUCT -> lookupProduct(image, productBox!!)
                Focus.SHELF -> {
                    // 地図に貯めた、今見えている商品を使う（何枚も見た結果の多数決）。足りなければ、この 1 枚で認識する
                    val fromMap = shelfMap.visible(f.geometry)
                    debug.longPress += "、地図から ${fromMap.size} 個"
                    val products = if (fromMap.size >= ShelfDescriber.MIN_PRODUCTS) fromMap else recognizer.recognize(image, boxes)
                    describe(products, jpeg(image, NARRATE_LONG_SIDE), boxes.size, camera = null, requested = true)
                }
                Focus.STEP_BACK -> voice.say("一歩下がって、棚の全体か、商品を一つだけ写してください。")
                Focus.NOTHING -> voice.say("商品が見えません。棚か商品の方にスマホを向けてください。")
            }
        } catch (e: RuntimeException) {
            Log.e(TAG, "long press failed", e)
            voice.say("調べられませんでした。")
        } finally {
            image.recycle()
        }
    }

    /** ③ 手前の商品を Cloud Vision API で調べる。この接近ですでに調べていれば、API を呼ばずにその結果を言う。 */
    private fun lookupProduct(image: Bitmap, box: FloatArray) {
        lookup.cachedResult?.let { debug.lookup = "（この接近で調べた結果を使う）"; speakLookup(it); return }
        if (!lookup.configured) {
            voice.say("商品を調べる設定がされていません。")
            return
        }
        val crop = cropWithMargin(image, box, 0.08f)
        val t0 = SystemClock.elapsedRealtime()
        val result = try { lookup.lookup(crop) } finally { if (crop !== image) crop.recycle() }
        result.exceptionOrNull()?.let { Log.w(TAG, "lookup failed", it) }
        debug.lookup = result.fold({ it?.let { a -> "社名「${a.maker}」商品「${a.product}」" } ?: "分からない" }, { "失敗: ${it.message}" }) +
            "（Cloud Vision ${SystemClock.elapsedRealtime() - t0}ms）"
        speakLookup(result)
    }

    private fun speakLookup(result: Result<ProductAnswer?>) {
        result.getOrNull()?.let { lastProduct = it.spoken() }
        val text = result.fold(
            onSuccess = { it?.spoken() ?: "商品が分かりませんでした。商品の正面を、カメラに向けてください。" },
            onFailure = { "調べられませんでした。通信を確認してください。" },
        )
        voice.say(text)
    }

    // ---------------------------------------------------------------- ④ 人の接近

    private fun runPeople(f: CapturedFrame) {
        val image = JpegEncoder.toUprightBitmap(f, PERSON_LONG_SIDE)
        val w = image.width.toFloat()
        val h = image.height.toFloat()
        val detected = try { personDetector.detect(image) } catch (e: RuntimeException) {
            Log.e(TAG, "person detection failed", e); emptyList()
        } finally { image.recycle() }
        // 自分の手や、持っている商品を人と間違えたものを外す
        val judged = detected.map { box ->
            val aspect = (box.bottom - box.top) * h / ((box.right - box.left) * w).coerceAtLeast(1f)
            val depth = Unprojector.medianDepthMeters(box, f.geometry)
            Triple(box, PersonFilter.isOwnHand(box, aspect, depth), depth)
        }
        val boxes = judged.filterNot { it.second }.map { it.first }
        val t = SystemClock.elapsedRealtime() / 1000.0
        val camera = cameraPosition(f)
        val observations = boxes.map { box ->
            val d = PersonDistance.estimate(box, f.geometry)
            val pos = d?.let { Unprojector.worldAt((box.left + box.right) / 2f, (box.top + box.bottom) / 2f, it, f.geometry) }
            PersonObservation(box, d, pos)
        }
        val tracked = tracker.update(t, observations, camera)
        // 近づいてくる人と、すぐ近くにいる人から鳴らす。近いほど速く
        val sounding = tracker.markers()
            .filter { t - it.lastSeenSeconds < PERSON_HOLD_S }
            .mapNotNull { m ->
                val d = horizontal(m.position, camera)
                val approaching = m.motion == PersonMotion.APPROACHING && d < APPROACH_ALERT_M
                if (!approaching && d >= CLOSE_ALERT_M) return@mapNotNull null
                m.id to Triple(m.position, beepInterval(d), if (approaching) 1f else 0.6f)
            }.toMap()
        mixer.setEmitters(sounding)
        val m = { x: Float? -> x?.let { "%.1fm".format(it) } ?: "?" }
        debug.people = judged.filter { it.second }.map { (b, _, depth) -> AssistDebug.PersonView(b, true, "手 深度${m(depth)}") } +
            tracked.map { p -> AssistDebug.PersonView(p.box, false, "人 ${m(p.distanceM)} ${p.motion.labelJa}") }
        debug.peopleInfo = "検出 ${detected.size}（手として外した ${detected.size - boxes.size}）/ 追跡 ${tracker.markers().size} / 鳴らしている ${sounding.size}"
        debug.peopleAt = SystemClock.elapsedRealtime()
    }

    // ---------------------------------------------------------------- 声の操作

    private fun startListening() {
        if (!listening.available) {
            voice.say("このスマホでは、音声認識が使えません。", interrupt = true)
            return
        }
        voice.stop()
        debug.voice = "聞いています…"
        listening.listen(onReady = { mixer.play(SpatialMixer.Sound(SpatialMixer.makeTick(), gain = 0.6f)) }) { text ->
            Log.i(TAG, "heard: $text")
            debug.voice = "聞き取り「${text ?: "（聞き取れず）"}」→ ${text?.let(Commands::parse) ?: if (text == null) "-" else "LLM に質問"}"
            handle(text?.let(Commands::parse), text)
        }
    }

    private fun handle(command: Command?, heard: String?) {
        when (command) {
            Command.SHELVES -> ask(heard!!)   // 棚の種類の判定は止めているので、画像を見て LLM が答える
            Command.DESCRIBE, Command.LOOKUP -> onLongPressed()
            Command.PAUSE -> { auto = false; updateSettings(); voice.say("お知らせを止めます。人が近づいたときの音は鳴ります。", interrupt = true) }
            Command.RESUME -> { auto = true; updateSettings(); describedAt = null; voice.say("お知らせを再開します。", interrupt = true) }
            Command.REPEAT -> voice.say(voice.lastSpoken.ifEmpty { "まだ何も話していません。" }, interrupt = true)
            Command.HELP -> voice.say(Commands.HELP_TEXT, interrupt = true)
            null -> if (heard.isNullOrBlank()) voice.say("聞き取れませんでした。", interrupt = true) else ask(heard)
        }
    }

    /** 決まった言い方でない質問は、今のカメラの画像と分かっていることを LLM に渡して答えてもらう。 */
    private fun ask(question: String) {
        if (!narrator.available) {
            voice.say("質問に答える設定がされていません。", interrupt = true)
            return
        }
        voice.say("調べています", interrupt = true, remember = false)
        pendingCapture = { f ->
            val image = JpegEncoder.toUprightBitmap(f, QUESTION_LONG_SIDE)
            val context = buildString {
                if (lastDescription.isNotEmpty()) appendLine("- 最後に伝えた棚の説明: $lastDescription")
                if (lastProduct.isNotEmpty()) appendLine("- 最後に調べた商品: $lastProduct")
            }
            val t0 = SystemClock.elapsedRealtime()
            val answer = try { narrator.answer(question, context, jpeg(image, QUESTION_LONG_SIDE)) } finally { image.recycle() }
            debug.voice = "質問「$question」→「${answer ?: "失敗"}」（${SystemClock.elapsedRealtime() - t0}ms）"
            voice.say(answer ?: "答えられませんでした。通信を確認してください。")
        }
    }

    private fun jpeg(image: Bitmap, longSide: Int): ByteArray {
        val k = minOf(1f, longSide.toFloat() / maxOf(image.width, image.height))
        val scaled = if (k < 1f) Bitmap.createScaledBitmap(image, (image.width * k).toInt(), (image.height * k).toInt(), true) else image
        return ByteArrayOutputStream().use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, 85, out)
            if (scaled !== image) scaled.recycle()
            out.toByteArray()
        }
    }

    // ---------------------------------------------------------------- 補助

    /** 名前の付いた商品を地図の上に置く。深度か壁から位置が決まらなければ置かない。 */
    private fun placeOnMap(p: RecognizedProduct, f: CapturedFrame, walls: List<Wall>, now: Long): Vec3? {
        val g = f.geometry
        val box = NormalizedBox(p.left, p.top, p.right, p.bottom)
        val depth = Unprojector.medianDepthMeters(box, g)
        val along = Unprojector.worldAt(p.cx, p.cy, 1f, g)
        val pos = ShelfMap.place(cameraPosition(f), along, depth?.let { Unprojector.worldAt(p.cx, p.cy, it, g) }, walls) ?: return null
        val dist = pos.distanceTo(cameraPosition(f))
        val (w, h) = ShelfMap.sizeInMeters(p.right - p.left, p.bottom - p.top, dist, g)
        shelfMap.add(p.name, p.genre, p.fine, pos, w, h, now)
        return pos
    }

    private fun updateSettings() {
        val ok = { b: Boolean -> if (b) "○" else "×" }
        debug.settings = "Gemini${ok(narrator.available)} Vision${ok(lookup.configured)} 自動のお知らせ${ok(auto)}"
    }

    private fun useHighResolutionCameraImage(s: Session) {
        val best = s.getSupportedCameraConfigs(CameraConfigFilter(s))
            .filter { it.imageSize.width <= MAX_CAMERA_IMAGE_WIDTH && it.fpsRange.upper >= 30 }
            .maxByOrNull { it.imageSize.width * it.imageSize.height }
        if (best != null) s.cameraConfig = best
    }

    private fun cameraPosition(f: CapturedFrame): Vec3 {
        val m = f.geometry.cameraToWorld
        return Vec3(m[12], m[13], m[14])
    }

    private fun cropWithMargin(image: Bitmap, b: FloatArray, margin: Float): Bitmap {
        val w = b[2] - b[0]
        val h = b[3] - b[1]
        val l = ((b[0] - w * margin) * image.width).toInt().coerceIn(0, image.width - 1)
        val t = ((b[1] - h * margin) * image.height).toInt().coerceIn(0, image.height - 1)
        val r = ((b[2] + w * margin) * image.width).toInt().coerceIn(l + 1, image.width)
        val btm = ((b[3] + h * margin) * image.height).toInt().coerceIn(t + 1, image.height)
        return Bitmap.createBitmap(image, l, t, r - l, btm - t)
    }

    private companion object {
        const val TAG = "Assist"
        const val SCENE_INTERVAL_MS = 1000L
        const val PERSON_INTERVAL_MS = 200L
        const val SCENE_LONG_SIDE = 1280
        const val PERSON_LONG_SIDE = 640
        const val LOOKUP_LONG_SIDE = 1600
        const val MAX_CAMERA_IMAGE_WIDTH = 1920
        const val NARRATE_LONG_SIDE = 768
        const val QUESTION_LONG_SIDE = 1280
        const val FORGET_DESCRIPTION_MS = 3000L
        const val MOVE_TO_REDESCRIBE_M = 1.5f
        const val DESCRIBE_RETRY_MS = 4000L
        const val APPROACH_GAP_MS = 2000L
        const val APPROACH_MOVE_M = 0.6f
        const val PERSON_HOLD_S = 1.0
        const val APPROACH_ALERT_M = 6f
        const val CLOSE_ALERT_M = 1.5f

        fun horizontal(a: Vec3, b: Vec3): Float {
            val dx = a.x - b.x
            val dz = a.z - b.z
            return sqrt(dx * dx + dz * dz)
        }

        /** 鳴らす間隔（秒）: 近いほど短く。 */
        fun beepInterval(d: Float): Float = (0.1f + 0.2f * d).coerceIn(0.15f, 1.0f)
    }
}
