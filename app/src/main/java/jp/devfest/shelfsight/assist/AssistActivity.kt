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
import android.view.ScaleGestureDetector
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
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
import jp.devfest.shelfsight.mapping.OrbitCamera
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
 * 話すのはタップしたときだけ（人が近づいたときの音は別）。歩いている間は、認識した商品で店のモデルと地図を作り続ける。
 *
 * 操作
 * - 1 回たたく: 棚に向かっていれば、その棚の主な商品と、上・真ん中・下の商品。近くに棚がいくつかあれば、方向ごとに「ポン」と鳴らして主な商品を言う
 * - 2 回たたく: 手前の商品を Cloud Vision API に聞き、社名と商品名を言う（1 回の接近で 1 回だけ呼ぶ）
 * - 長押し: 話しかける。「〇〇はどこ？」などに、店の地図・今いる場所・調べた商品を知っている LLM が答える
 * - 人が近づいてくる: その人の方向からピコピコ鳴らす
 * TalkBack が有効なときは、ダブルタップが 1 回たたく、ダブルタップ長押しが長押し、「商品名を調べる」の操作が 2 回たたくになる。
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
    private val pon = SpatialMixer.makePon()
    private val lookup = ProductLookup(BuildConfig.CLOUD_VISION_API_KEY)
    private val narrator = Narrator(
        BuildConfig.GEMINI_API_KEY.takeIf { it.isNotEmpty() }
            // 話す文を作るだけなので、速い lite のモデルから試す（flash は 6〜9 秒、lite は 2 秒ほど）
            ?.let { GeminiClient(it, BuildConfig.GEMINI_MODELS.split(",").map(String::trim).filter(String::isNotEmpty).sortedBy { m -> "lite" !in m }) },
    )
    private val judge = SceneJudge()
    /** 1 秒ごとの商品認識の結果を、AR 空間の壁の上に貯めたもの（アプリを閉じると消える）。 */
    private val shelfMap = ShelfMap()
    /** 認識した商品の位置から推定する店のモデル。[sceneWorker] からだけ触る。 */
    private val storeModel by lazy { StoreModel(StoreModel.parseThemes(assets.open("store_themes.json").bufferedReader().readText())) }
    @Volatile private var storeView: StoreView? = null
    private var lastStoreEstimate = 0L
    private lateinit var storeRenderer: StoreModelRenderer
    private val orbit = OrbitCamera()
    @Volatile private var showModel = false
    @Volatile private var deviceAt: Vec3? = null
    private lateinit var modelText: ScrollView
    private lateinit var modelLabel: TextView
    private lateinit var debugOverlay: DebugOverlay
    private lateinit var tabs: List<Button>
    private lateinit var recordButton: Button
    /** 記録中ならその記録（GL スレッドで作って閉じる）。 */
    @Volatile private var sessionRecorder: SessionRecorder? = null
    @Volatile private var recordToggle = false
    private val recorder by lazy { Recorder(java.io.File(getExternalFilesDir(null), "records")) }
    /** 商品のジャンル（26 種類）→ 歩行中の判別と同じ 7 分類。場違いな商品を外すのに使う。 */
    private val groups by lazy { ShelfDescriber.Groups.parse(assets.open("coarse/categories.json").bufferedReader().readText()) }

    private var lastScene = 0L
    private var lastPerson = 0L
    private var greeted = false
    /** 今の接近で、大きく写った商品を最後に見た時刻と、そのときの場所。 */
    private var approachSeenAt = 0L
    private var approachFrom: Vec3? = null
    /** 調べた商品の記録（質問に答える LLM に渡す）。 */
    private val lookedUp = java.util.concurrent.CopyOnWriteArrayList<LookedUp>()

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
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean { onTap(); return true }
            override fun onDoubleTap(e: MotionEvent): Boolean { onProductTap(); return true }
            override fun onLongPress(e: MotionEvent) { startListening() }
        })
        val touch = object : View(this) {
            @android.annotation.SuppressLint("ClickableViewAccessibility")
            override fun onTouchEvent(event: MotionEvent): Boolean = if (showModel) orbitTouch(event) else gestures.onTouchEvent(event)
        }.apply {
            contentDescription = "1回たたくと棚の案内。2回たたくと商品の名前。長押しして質問"
            isClickable = true
            isLongClickable = true
            // TalkBack から: ダブルタップ＝クリック（棚の案内）、ダブルタップ長押し＝長押し（質問）。2 回たたくの代わりは操作メニューから
            setOnClickListener { onTap() }
            setOnLongClickListener { startListening(); true }
        }
        androidx.core.view.ViewCompat.addAccessibilityAction(touch, "商品名を調べる") { _, _ -> onProductTap(); true }
        debugOverlay = DebugOverlay(this, debug)
        modelLabel = TextView(this).apply {
            setTextColor(android.graphics.Color.WHITE)
            textSize = 11f
            typeface = android.graphics.Typeface.MONOSPACE
            setPadding(16, 8, 16, 8)
            text = "歩くと店のモデルができます"
        }
        modelText = ScrollView(this).apply {
            setBackgroundColor(0xAA000000.toInt())
            visibility = View.GONE
            addView(modelLabel)
        }
        val names = listOf("案内", "3D モデル")
        tabs = names.indices.map { i -> Button(this).apply { text = names[i]; setOnClickListener { selectTab(i == 1) } } }
        recordButton = Button(this).apply { text = RECORD_LABEL; setOnClickListener { recordToggle = true } }
        val tabBar = LinearLayout(this).apply {
            (tabs + recordButton).forEach { addView(it, LinearLayout.LayoutParams(0, -2, 1f)) }
        }
        setContentView(FrameLayout(this).apply {
            addView(gl)
            addView(debugOverlay, FrameLayout.LayoutParams(-1, -1))
            addView(touch, FrameLayout.LayoutParams(-1, -1))
            addView(modelText, FrameLayout.LayoutParams(-1, resources.displayMetrics.heightPixels / 3, Gravity.BOTTOM))
            addView(tabBar, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))
        })
        selectTab(false)
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
                val config = Config(s).setFocusMode(Config.FocusMode.AUTO).setPlaneFindingMode(Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL)
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
            voice.say("準備ができました。1回たたくと棚の案内、2回たたくと商品の名前、長押しで質問ができます。")
            voice.prepare(listOf("調べています"))
        }
    }

    override fun onPause() {
        super.onPause()
        listening.cancel()
        gl.onPause()
        session?.pause()
        stopRecorder()   // MP4 は一時停止で ARCore が閉じる
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
        storeRenderer = StoreModelRenderer()
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
        if (recordToggle) {
            recordToggle = false
            toggleRecording(s)
        }
        val frame = try { s.update() } catch (e: Exception) { return }
        val camera = frame.camera
        // 3D モデルを見ている間も、下の認識と地図づくりは止めない
        if (showModel) storeRenderer.draw(storeView, orbit, viewW, viewH, deviceAt, (deviceAt?.y ?: 0f) - DEVICE_HEIGHT_M)
        else background.draw(frame)
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
        // 記録中は認識を止めて、データを取りこぼさないようにする（認識は PC で再生して動かす）
        sessionRecorder?.let { rec ->
            rec.onFrame(frame, camera, sensorRotation)
            val now = System.currentTimeMillis()
            rec.statusIfDue(now)?.let { runOnUiThread { recordButton.text = it } }
            if (rec.outOfSpace()) toggleRecording(s)
            return
        }
        if (camera.trackingState != TrackingState.TRACKING) return

        // 聞く人の向き: スマホを胸の前に立てて持つので、カメラの前（-z）を水平にしたもの
        val pose = camera.pose
        deviceAt = Vec3(pose.tx(), pose.ty(), pose.tz())
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
                val counters = counterPoints(s, pose.ty())
                val heading = debug.headingDeg
                sceneWorker.execute { try { runScene(f, walls, counters, heading) } finally { sceneBusy.set(false) } }
            }
        }
    }

    // ---------------------------------------------------------------- ①② 写っているものの判定

    /**
     * 1 秒ごと: 商品の枠を出して写っているものを判定し、商品に名前を付けて地図（壁の上）に置き、画像と姿勢と結果を記録する。
     */
    private fun runScene(f: CapturedFrame, walls: List<Wall>, counters: List<Vec3>, heading: Float?) {
        val image = JpegEncoder.toUprightBitmap(f, SCENE_LONG_SIDE)
        try {
            val now = SystemClock.elapsedRealtime()
            val boxes = recognizer.detect(image).map { it.first }
            val detectMs = SystemClock.elapsedRealtime() - now
            val switched = judge.update(SceneEvidence(0, boxes))
            val products = recognizer.recognize(image, boxes)
            val recognizeMs = SystemClock.elapsedRealtime() - now - detectMs
            val placed = products.map { p -> Recorder.Product(p, placeOnMap(p, f, walls, now)) }
            updateStoreModel(boxes, placed, counters, f, now)
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
        } catch (e: RuntimeException) {
            Log.e(TAG, "scene analysis failed", e)
        } finally {
            image.recycle()
        }
    }

    // ---------------------------------------------------------------- 1 回たたく: 棚の案内

    private fun onTap() {
        sessionRecorder?.let { it.event("tap"); return }
        listening.cancel()
        val l = listener ?: return voice.say("位置を合わせています。少し周りを見回してから、もう一度たたいてください。", interrupt = true)
        when (val answer = StoreGuide.tapAnswer(storeView, shelfMap.snapshot(), l, groups, l.position.y - DEVICE_HEIGHT_M)) {
            is TapAnswer.Sentence -> { debug.description = answer.text; voice.say(answer.text, interrupt = true) }
            is TapAnswer.Around -> {
                debug.description = answer.clauses.joinToString("")
                voice.remember(debug.description)
                answer.clauses.forEachIndexed { i, c ->
                    val at = answer.shelves[i].at
                    voice.say(c, interrupt = i == 0, remember = false, before = SpatialMixer.Sound(pon, { at }))
                }
            }
            TapAnswer.Unknown -> voice.say("まだ棚が分かりません。棚の方を向いて、少し歩いてください。", interrupt = true)
        }
    }

    // ---------------------------------------------------------------- 2 回たたく: 商品の名前

    private fun onProductTap() {
        sessionRecorder?.let { it.event("double_tap"); return }
        listening.cancel()
        voice.say("調べています", interrupt = true, remember = false)
        pendingCapture = ::runProductTap
    }

    /** 撮った画像の手前の商品を調べる。棚が広く写っているだけなら、近づいてもらう。 */
    private fun runProductTap(f: CapturedFrame) {
        val image = JpegEncoder.toUprightBitmap(f, LOOKUP_LONG_SIDE)
        try {
            val boxes = recognizer.detect(image).map { it.first }
            val (focus, productBox) = SceneJudge.focus(boxes)
            val largest = boxes.maxOfOrNull { SceneJudge.area(it) } ?: 0f
            Log.i(TAG, "product tap: ${boxes.size} boxes, largest $largest -> $focus")
            debug.longPress = "$focus（枠 ${boxes.size} 個、いちばん大きい枠 ${(largest * 100).toInt()}%）"
            debug.focusBox = productBox
            debug.focusAt = SystemClock.elapsedRealtime()
            debug.productBoxes = boxes
            debug.sceneAt = debug.focusAt
            when (focus) {
                Focus.PRODUCT -> lookupProduct(image, productBox!!, cameraPosition(f))
                Focus.SHELF, Focus.STEP_BACK -> voice.say("商品を一つだけ、カメラに近づけて、もう一度2回たたいてください。")
                Focus.NOTHING -> voice.say("商品が見えません。商品の方にスマホを向けてください。")
            }
        } catch (e: RuntimeException) {
            Log.e(TAG, "product tap failed", e)
            voice.say("調べられませんでした。")
        } finally {
            image.recycle()
        }
    }

    /** ③ 手前の商品を Cloud Vision API で調べる。この接近ですでに調べていれば、API を呼ばずにその結果を言う。 */
    private fun lookupProduct(image: Bitmap, box: FloatArray, at: Vec3) {
        lookup.cachedResult?.let { debug.lookup = "（この接近で調べた結果を使う）"; speakLookup(it, at); return }
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
        speakLookup(result, at)
    }

    private fun speakLookup(result: Result<ProductAnswer?>, at: Vec3) {
        result.getOrNull()?.let { answer ->
            val name = answer.spoken()
            if (lookedUp.lastOrNull()?.name != name) lookedUp += LookedUp(name, at, SystemClock.elapsedRealtime())
            if (lookedUp.size > MAX_LOOKED_UP) lookedUp.removeAt(0)
        }
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
        sessionRecorder?.event("long_press")
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
        sessionRecorder?.let { it.event("speech", heard ?: ""); return }
        when (command) {
            Command.REPEAT -> voice.say(voice.lastSpoken.ifEmpty { "まだ何も話していません。" }, interrupt = true)
            Command.HELP -> voice.say(Commands.HELP_TEXT, interrupt = true)
            null -> if (heard.isNullOrBlank()) voice.say("聞き取れませんでした。", interrupt = true) else ask(heard)
        }
    }

    /** 決まった言い方でない質問は、今のカメラの画像と、店の地図や調べた商品を LLM に渡して答えてもらう。 */
    private fun ask(question: String) {
        if (!narrator.available) {
            voice.say("質問に答える設定がされていません。", interrupt = true)
            return
        }
        voice.say("調べています", interrupt = true, remember = false)
        pendingCapture = { f ->
            val image = JpegEncoder.toUprightBitmap(f, QUESTION_LONG_SIDE)
            val context = StoreGuide.context(storeView, listener, shelfMap.snapshot(), lookedUp.toList(), SystemClock.elapsedRealtime(), voice.lastSpoken)
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

    /** GL スレッドで記録を始める・止める。 */
    private fun toggleRecording(s: Session) {
        val current = sessionRecorder
        if (current == null) {
            val rec = SessionRecorder(this, java.io.File(getExternalFilesDir(null), "recordings"))
            rec.start(s, sensorRotation)
            sessionRecorder = rec
            runOnUiThread { recordButton.text = "■ 停止" }
            Log.i(TAG, "recording ${rec.name} mp4=${rec.hasMp4}")
        } else {
            try { s.stopRecording() } catch (e: com.google.ar.core.exceptions.RecordingFailedException) { Log.e(TAG, "stop recording failed", e) }
            stopRecorder()
        }
    }

    private fun stopRecorder() {
        val rec = sessionRecorder ?: return
        sessionRecorder = null
        rec.close()
        runOnUiThread { recordButton.text = RECORD_LABEL }
    }

    /** 店のモデルに今の 1 枚を足す。名前が付かなかった枠も、棚の形の手がかりとして位置だけ使う。 */
    private fun updateStoreModel(boxes: List<FloatArray>, placed: List<Recorder.Product>, counters: List<Vec3>, f: CapturedFrame, now: Long) {
        val cam = cameraPosition(f)
        val cx = cam.x.toDouble()
        val cz = cam.z.toDouble()
        storeModel.addCamera(cx, cz)
        for (pr in placed) pr.position?.let { storeModel.addProduct(cx, cz, it.x.toDouble(), it.z.toDouble(), pr.p.name, pr.p.genre ?: "") }
        for (b in boxes) {
            if (placed.any { it.p.left == b[0] && it.p.top == b[1] }) continue
            val depth = Unprojector.medianDepthMeters(NormalizedBox(b[0], b[1], b[2], b[3]), f.geometry) ?: continue
            if (depth > ShelfMap.MAX_DEPTH_M) continue
            val p = Unprojector.worldAt((b[0] + b[2]) / 2f, (b[1] + b[3]) / 2f, depth, f.geometry)
            storeModel.addProduct(cx, cz, p.x.toDouble(), p.z.toDouble(), "", "")
        }
        for (c in counters) storeModel.addCounter(c.x.toDouble(), c.z.toDouble())
        if (now - lastStoreEstimate < STORE_ESTIMATE_MS) return
        lastStoreEstimate = now
        val view = storeModel.estimate()
        storeView = view
        val description = StoreModel.text(view)
        runOnUiThread { modelLabel.text = description }
    }

    /** 腰の高さの水平面（レジの天板）の上の点。商品の棚の段は商品に隠れるので、ほぼレジだけが残る。 */
    private fun counterPoints(s: Session, deviceY: Float): List<Vec3> = s.getAllTrackables(Plane::class.java)
        .filter { it.type == Plane.Type.HORIZONTAL_UPWARD_FACING && it.trackingState == TrackingState.TRACKING && it.subsumedBy == null }
        .filter { it.centerPose.ty() - deviceY in COUNTER_BELOW_DEVICE_M && it.extentX * it.extentZ >= MIN_COUNTER_AREA_M2 }
        .flatMap { pl ->
            val x = pl.extentX / 2
            val z = pl.extentZ / 2
            listOf(0f to 0f, x to z, -x to z, x to -z, -x to -z).map { (dx, dz) ->
                val w = pl.centerPose.transformPoint(floatArrayOf(dx, 0f, dz))
                Vec3(w[0], w[1], w[2])
            }
        }

    private fun selectTab(model: Boolean) {
        showModel = model
        tabs.forEachIndexed { i, b -> b.alpha = if ((i == 1) == model) 1f else 0.55f }
        debugOverlay.visibility = if (model) View.GONE else View.VISIBLE
        modelText.visibility = if (model) View.VISIBLE else View.GONE
        if (model) storeView?.let { fitOrbit(it) }
    }

    private fun fitOrbit(v: StoreView) {
        val r = v.room ?: return
        val cu = (r[0] + r[2]) / 2
        val cv = (r[1] + r[3]) / 2
        val cs = kotlin.math.cos(v.yaw)
        val sn = kotlin.math.sin(v.yaw)
        orbit.fit(Vec3((cu * cs - cv * sn).toFloat(), (deviceAt?.y ?: 0f) - DEVICE_HEIGHT_M, (cu * sn + cv * cs).toFloat()), maxOf(r[2] - r[0], r[3] - r[1]).toFloat() / 2)
    }

    private val orbitScale by lazy {
        ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean { orbit.zoom(d.scaleFactor); return true }
        })
    }
    private var orbitX = 0f
    private var orbitY = 0f

    /** 3D タブの操作: 1 本指で回転、2 本指でつまんで拡大。 */
    private fun orbitTouch(e: MotionEvent): Boolean {
        orbitScale.onTouchEvent(e)
        if (e.actionMasked == MotionEvent.ACTION_MOVE && e.pointerCount == 1) orbit.rotate(e.x - orbitX, e.y - orbitY)
        orbitX = e.x
        orbitY = e.y
        return true
    }

    private fun updateSettings() {
        val ok = { b: Boolean -> if (b) "○" else "×" }
        debug.settings = "Gemini${ok(narrator.available)} Vision${ok(lookup.configured)}"
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
        const val MAX_LOOKED_UP = 20
        const val QUESTION_LONG_SIDE = 1280
        const val APPROACH_GAP_MS = 2000L
        const val APPROACH_MOVE_M = 0.6f
        const val PERSON_HOLD_S = 1.0
        const val APPROACH_ALERT_M = 6f
        const val CLOSE_ALERT_M = 1.5f
        const val STORE_ESTIMATE_MS = 3000L
        const val RECORD_LABEL = "● 記録"
        const val DEVICE_HEIGHT_M = 1.3f
        /** スマホ（胸の高さ）から見たレジの天板の高さ。 */
        val COUNTER_BELOW_DEVICE_M = -0.6f..-0.1f
        const val MIN_COUNTER_AREA_M2 = 0.1f

        fun horizontal(a: Vec3, b: Vec3): Float {
            val dx = a.x - b.x
            val dz = a.z - b.z
            return sqrt(dx * dx + dz * dz)
        }

        /** 鳴らす間隔（秒）: 近いほど短く。 */
        fun beepInterval(d: Float): Float = (0.1f + 0.2f * d).coerceIn(0.15f, 1.0f)
    }
}
