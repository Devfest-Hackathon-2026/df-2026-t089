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
import jp.devfest.shelfsight.mapping.MapRenderer
import jp.devfest.shelfsight.mapping.StoreMapper
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.perception.MediaPipePersonDetector
import jp.devfest.shelfsight.perception.PersonDistance
import jp.devfest.shelfsight.perception.PersonMotion
import jp.devfest.shelfsight.perception.PersonObservation
import jp.devfest.shelfsight.perception.PersonTracker
import jp.devfest.shelfsight.walking.CoarseClassifier
import jp.devfest.shelfsight.walking.CoarseResult
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sqrt

/**
 * 視覚障害のある人のための、コンビニの棚と商品の案内。胸の前にスマホを立てて持ち、イヤホンで聞く。
 *
 * ① 棚の種類（歩行中の判別、画面を 8 列に分けて列ごとに 7 分類）を画面に出し、AR 空間の棚の上に貯める（[ShelfLabelMap]）。
 *    同じ棚をもう一度向いたら、貯めたラベルで列の分類と商品の名前を直す（読み上げはしない）
 * ② 棚が 1 つ写っている: 商品を認識し、LLM で何があるかを短く自然な文にして話す（場違いな商品は言わない）
 * ③ ボタン: 正面の棚の大まかな種類（歩行中の判別の 7 分類）を言う
 * ④ スマホのすぐ前に商品がある（画面の中央の深度だけで決める、[NearProduct]）: 長押しで Cloud Vision API で文字を読み（1 回の接近で 1 回だけ呼ぶ）、
 *    その文字からカテゴリを決めて（端末の辞書で決めきれなければ Gemini、[CategoryJudge]）、商品名、カテゴリの順に言う（iOS 版と同じ）
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
    /** ARCore の深度が使える機種か。使えれば、長押しで調べる商品は深度だけで決める（[NearProduct]）。 */
    @Volatile private var depthSupported = false
    private var viewW = 0
    private var viewH = 0
    private var geometryChanged = false
    private val background = CameraBackground()

    @Volatile private var listener: Listener? = null
    private val mixer by lazy { SpatialMixer { listener } }
    private val voice by lazy { Voice(this, mixer) }
    private val listening by lazy { Listening(this) }
    private lateinit var debugOverlay: DebugOverlay

    /**
     * 商品の解析。棚の種類（[coarseWorker]）より後回しにするので、優先度を下げたスレッドで動かす。
     * ONNX Runtime の計算スレッドは作ったスレッドの優先度を引き継ぐので、モデルもこのスレッドで読み込む。
     */
    private val sceneWorker = Executors.newSingleThreadExecutor { r ->
        // 商品だけを計るときは譲る相手がいないので、優先度を下げない
        val priority = if (PRODUCT_ONLY) android.os.Process.THREAD_PRIORITY_DEFAULT else android.os.Process.THREAD_PRIORITY_BACKGROUND
        Thread { android.os.Process.setThreadPriority(priority); r.run() }
    }
    /** 商品の解析にかかった時間（[PRODUCT_ONLY] の計測用）。 */
    private val timing = SceneTiming()
    private val personWorker = Executors.newSingleThreadExecutor()
    private val lookupWorker = Executors.newSingleThreadExecutor()
    /**
     * 棚の種類の判別。商品の解析（1 秒ごと）とは別に [COARSE_INTERVAL_MS] ごとに動かす。
     * 同じ画像で同時に動かすと、どちらも遅くなって 1 回の解析が 3 秒ほどかかった。
     */
    private val coarseWorker = Executors.newSingleThreadExecutor { r ->
        Thread { android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_DEFAULT); r.run() }
    }
    private val coarseBusy = AtomicBoolean(false)
    private var lastCoarse = 0L
    /** 歩行中の判別のモデル。大きいのでリポジトリに含めておらず、入っていないビルドでは null（棚の種類だけが止まる）。 */
    private val coarse by lazy { CoarseClassifier.fromAssets(this, BuildConfig.COARSE_MODEL, COARSE_THREADS) }
    /** 最後の棚の種類の判別（地図で直した後）と、その時刻。 */
    @Volatile private var latestShelves: ShelfView? = null
    @Volatile private var latestShelvesAt = 0L
    /** カメラの動きの速さ（m/s と 度/s、ならしたもの）。GL スレッドで更新する。 */
    @Volatile private var speedMps = 0f
    @Volatile private var turnDps = 0f
    private var lastPose: FloatArray? = null
    private var lastPoseAt = 0L
    /** 棚の判別を、地図があっても全部の列でやり直した最後の時刻（地図の読み違いを固めないため）。 */
    @Volatile private var lastFullShelves = 0L
    /** 省いた処理の数（画面の表示用）。 */
    private val skipped = java.util.concurrent.ConcurrentHashMap<String, Int>()
    /**
     * 店の 3D の地図（ARCore の面と深度から、歩きながら裏で作る）。「地図」ボタンで、棚のラベルと一緒に見られる。
     */
    private val mapper = StoreMapper()
    private var mapRenderer: MapRenderer? = null
    @Volatile private var showMap = false
    private lateinit var mapOverlay: Map3DOverlay
    private val mapView = FloatArray(16)
    private val mapProjection = FloatArray(16)
    /** 棚の種類（7 分類）を AR 空間の棚の上に貯めたもの（アプリを閉じると消える）。 */
    private val shelfLabels = ShelfLabelMap()
    private val sceneBusy = AtomicBoolean(false)
    private val personBusy = AtomicBoolean(false)
    /** 次のフレームを高解像度で撮って渡す先（長押し・質問）。 */
    @Volatile private var pendingCapture: ((CapturedFrame) -> Unit)? = null

    private val recognizer by lazy { ProductRecognizer(this) }
    private val personDetector by lazy { MediaPipePersonDetector(this) }
    private val tracker = PersonTracker()
    private val lookup by lazy {
        ProductLookup(
            BuildConfig.CLOUD_VISION_API_KEY,
            CategoryJudge(CategoryRules.parse(assets.open("category_rules.json").bufferedReader().readText()), geminiClient()),
            getExternalFilesDir("lookup"),
        )
    }
    private val narrator = Narrator(geminiClient())
    private val judge = SceneJudge()
    private val recorder by lazy { Recorder(java.io.File(getExternalFilesDir(null), "records")) }
    /** 商品のジャンル（26 種類）→ 歩行中の判別と同じ 7 分類。場違いな商品を外すのに使う。 */
    private val groups by lazy { ShelfDescriber.Groups.parse(assets.open("coarse/categories.json").bufferedReader().readText()) }

    /** 近い商品（[NearProduct]）を最後に見た時刻。しばらく見えなければ、次に見えたときを新しい接近とする（sceneWorker だけで使う）。 */
    private var nearSeenAt = 0L
    private var lastScene = 0L
    private var lastPerson = 0L
    private var greeted = false
    @Volatile private var auto = true
    /** ② を話した場所と、その棚の種類（細分類）。ここから離れるか、別の種類の棚に向くか、棚 1 つの場面を離れるまでは繰り返さない。 */
    @Volatile private var describedAt: Vec3? = null
    @Volatile private var describedKinds: Set<String>? = null
    /** 自動の棚の説明を作っている間（LLM を待っている間）は、次を頼まない。 */
    private val describing = AtomicBoolean(false)
    private var leftOneShelfAt = 0L
    private var lastDescribeTry = 0L
    /** 説明した棚から別の棚に移ったか（1.5m 動かなくても、隣の棚に向きを変えたら説明し直す。sceneWorker だけで使う）。 */
    private val shelfChange = ShelfChange()
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
            contentDescription = "長押しで、手前の商品や棚を調べる。2回たたくと質問"
            isClickable = true
            isLongClickable = true
            // TalkBack から: ダブルタップ＝クリック（質問）、ダブルタップ長押し＝長押し
            setOnClickListener { startListening() }
            setOnLongClickListener { onLongPressed(); true }
        }
        debugOverlay = DebugOverlay(this, debug)
        setContentView(FrameLayout(this).apply {
            addView(gl)
            addView(debugOverlay, FrameLayout.LayoutParams(-1, -1))
            addView(touch, FrameLayout.LayoutParams(-1, -1))
            // 3D の地図を見る（確認用）。読み上げの操作の邪魔にならないよう、右上の小さなボタン
            mapOverlay = Map3DOverlay(this@AssistActivity, mapper.orbit, shelfLabels::snapshot) {
                "3D の地図  棚のラベル ${shelfLabels.size} か所  ${mapper.state.value.status}"
            }.apply { visibility = View.GONE }
            addView(mapOverlay, FrameLayout.LayoutParams(-1, -1))
            // 正面の棚の種類を言うボタン。見えない人でも見つけやすいよう、画面の下の端に大きく置く
            val d = resources.displayMetrics.density
            val shelfButton = android.widget.Button(this@AssistActivity).apply {
                text = "棚の種類"
                textSize = 24f
                contentDescription = "棚の種類。押すと、正面の棚の種類を言います"
                setOnClickListener { onShelfButton() }
            }
            addView(shelfButton, FrameLayout.LayoutParams(-1, (SHELF_BUTTON_DP * d).toInt(), Gravity.BOTTOM).apply {
                val m = (8 * d).toInt(); leftMargin = m; rightMargin = m; bottomMargin = (24 * d).toInt()
            })
            debugOverlay.bottomInset = (SHELF_BUTTON_DP + 32) * d   // 画面の文字がボタンに隠れないよう、上へずらす
            if (PRODUCT_ONLY) return@apply   // 地図を作らないので、地図のボタンも出さない
            val button = android.widget.Button(this@AssistActivity).apply {
                text = "地図"
                textSize = 14f
                alpha = 0.85f
                contentDescription = "3D の地図を見る（確認用）"
                setOnClickListener {
                    showMap = !showMap
                    mapper.setShowMap(showMap)   // 開くときに、歩いた範囲が入るよう視点を合わせる
                    mapOverlay.visibility = if (showMap) View.VISIBLE else View.GONE
                    text = if (showMap) "閉じる" else "地図"
                }
            }
            addView(button, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.END).apply {
                topMargin = (44 * resources.displayMetrics.density).toInt()
                rightMargin = (8 * resources.displayMetrics.density).toInt()
            })
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
                val config = Config(s).setFocusMode(Config.FocusMode.AUTO)
                    // 商品だけを計るときは、地図と棚のラベルにしか使わない面の検出を止める
                    .setPlaneFindingMode(if (PRODUCT_ONLY) Config.PlaneFindingMode.DISABLED else Config.PlaneFindingMode.VERTICAL)
                // 深度は、長押しで調べる近い商品の判定（NearProduct）に使うので、商品だけを計るときも使う
                depthSupported = s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
                if (depthSupported) config.depthMode = Config.DepthMode.AUTOMATIC
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
        sceneWorker.execute { recognizer }   // 優先度を下げたスレッドで読み込む（上の説明）
        coarseWorker.execute { coarse }      // ボタンを押したときにすぐ答えられるよう、先に読み込む
        gl.onResume()
        if (!greeted) {
            greeted = true
            voice.say("準備ができました。画面の下のボタンで、正面の棚の種類を言います。長押しで、手前の商品や棚を調べます。2回たたくと、質問できます。")
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
        for (w in listOf(sceneWorker, personWorker, lookupWorker, coarseWorker)) w.shutdownNow()
        mapper.close()
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
        mapRenderer = MapRenderer()
        mapper.onGlContextLost()
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
        // 3D の地図は、見ていない間も裏で作り続ける
        if (!PRODUCT_ONLY) mapper.onFrame(s, frame, frame.camera)
        if (showMap) drawMap() else background.draw(frame)
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
        updateMotion(pose, now)
        val rotation = sensorRotation   // 画面は縦に固定している
        pendingCapture?.let { target ->
            FrameCapture.capture(frame, camera, rotation, depthEnabled = depthSupported, fullResolution = true)?.let { f ->
                pendingCapture = null
                lookupWorker.execute { target(f) }
            }
        }
        if (PRODUCT_ONLY) {
            // 商品の解析だけを、前の解析が終わったらすぐ次を回す（地図・棚の種類・人の検出は止める）
            if (sceneBusy.compareAndSet(false, true)) {
                val t0 = SystemClock.elapsedRealtime()
                // 深度は、すぐ前の商品の判定（NearProduct）に使うので、商品だけを計るときも撮る（iOS 版と同じ）
                val f = FrameCapture.capture(frame, camera, rotation, depthEnabled = depthSupported, usualLongSide = SCENE_LONG_SIDE)
                if (f == null) sceneBusy.set(false) else {
                    val captureMs = SystemClock.elapsedRealtime() - t0
                    val heading = debug.headingDeg
                    sceneWorker.execute { try { runScene(f, emptyList(), heading, captureMs) } finally { sceneBusy.set(false) } }
                }
            }
            return
        }
        if (now - lastPerson >= PERSON_INTERVAL_MS && personBusy.compareAndSet(false, true)) {
            val f = FrameCapture.capture(frame, camera, rotation, depthEnabled = true)
            if (f == null) personBusy.set(false) else {
                lastPerson = now
                personWorker.execute { try { runPeople(f) } finally { personBusy.set(false) } }
            }
        }
        val fast = speedMps > FAST_MPS || turnDps > FAST_DPS
        if (now - lastScene >= SCENE_INTERVAL_MS && fast) {
            // 速く動いている間は画像がぶれて役に立たない
            lastScene = now
            skipped.merge("動きが速い", 1, Int::plus)
            debug.skipInfo = skipped.entries.joinToString("、") { "${it.key} ${it.value}" }
        }
        if (now - lastScene >= SCENE_INTERVAL_MS && sceneBusy.compareAndSet(false, true)) {
            val f = FrameCapture.capture(frame, camera, rotation, depthEnabled = true, usualLongSide = SCENE_LONG_SIDE)
            if (f == null) sceneBusy.set(false) else {
                lastScene = now
                val walls = walls(s)
                val heading = debug.headingDeg
                sceneWorker.execute { try { runScene(f, walls, heading) } finally { sceneBusy.set(false) } }
            }
        }
        if (now - lastCoarse >= COARSE_INTERVAL_MS && !fast && coarseBusy.compareAndSet(false, true)) {
            val f = FrameCapture.capture(frame, camera, rotation, depthEnabled = true, usualLongSide = SCENE_LONG_SIDE)
            if (f == null) coarseBusy.set(false) else {
                lastCoarse = now
                val walls = walls(s)
                coarseWorker.execute { try { runShelves(f, walls) } finally { coarseBusy.set(false) } }
            }
        }
    }

    // ---------------------------------------------------------------- ①② 写っているものの判定

    /**
     * 1 秒ごと: 商品の枠を出して写っているものを判定し、商品に名前を付け、画像と姿勢と結果を記録する。
     * 商品は地図に置かない（地図に置くのは棚の種類のラベルだけ）。
     */
    private fun runScene(f: CapturedFrame, walls: List<Wall>, heading: Float?, captureMs: Long = 0L) {
        val t0 = SystemClock.elapsedRealtime()
        val image = JpegEncoder.toUprightBitmap(f, SCENE_LONG_SIDE)
        val convertMs = SystemClock.elapsedRealtime() - t0
        try {
            val now = SystemClock.elapsedRealtime()
            val eye = cameraPosition(f)
            val boxes = recognizer.detect(image).map { it.first }
            val detectMs = SystemClock.elapsedRealtime() - now
            // 棚の種類は別の周期の最後の結果（古すぎれば使わない）
            val shelfView = latestShelves.takeIf { now - latestShelvesAt < SHELF_FRESH_MS }
            val shelves = shelfView?.segments.orEmpty()
            // スマホのすぐ前の商品（枠は使わず、画面の中央の深度だけで決める）
            val reading = if (depthSupported) NearProduct.measure(f.geometry) else null
            val near = reading?.pick
            judge.update(SceneEvidence(shelves.size, boxes, near != null))
            // 新しく近い商品が見えたら、前に調べた結果を忘れる（次の長押しで Cloud Vision を 1 回呼ぶ）
            if (near != null) {
                if (nearSeenAt == 0L || now - nearSeenAt > NEAR_GAP_MS) lookup.newApproach()
                nearSeenAt = now
            }
            debug.nearBox = near?.box
            debug.nearDistance = near?.distanceM
            debug.near = reading?.describe() ?: "深度が使えない機種なので、近い商品を判定できません（長押しでは画面の中央の枠で調べる）"
            // 商品の名前は、その場所の棚の種類（地図のラベル、なければ最後の判別の列）に合うものへ寄せる
            val products = recognizer.recognize(image, boxes, { b -> shelfFor(b, f, walls, eye, shelfView) }, groups::of)
            val recognizeMs = SystemClock.elapsedRealtime() - now - detectMs
            if (PRODUCT_ONLY) {
                timing.add(captureMs, convertMs, detectMs, recognizeMs, boxes.size, products.size)
                Log.i(TAG, "product only: ${timing.lastLine()}")
            } else {
                val placed = products.map { p -> Recorder.Product(p, null) }
                recorder.record(System.currentTimeMillis(), image, f.geometry, heading, walls.size, placed, shelfView?.columns)
            }
            debug.productBoxes = boxes
            debug.products = products
            debug.columnProducts = ShelfColumns.productsByColumn(products, shelfView?.columns?.size ?: DEFAULT_COLUMNS)
            debug.scene = judge.scene
            debug.shelfLabels = shelfLabels.visible(f.geometry)
            debug.sceneMs = if (PRODUCT_ONLY) "【商品だけ計測】\n${timing.summary()}" else "枠 ${detectMs}ms / 名前 ${recognizeMs}ms / 壁 ${walls.size}"
            debug.sceneAt = now
            debug.mapInfo = "棚のラベル ${shelfLabels.size} か所（今見えている ${debug.shelfLabels.size}）" +
                "  棚の種類で選び直した商品 ${recognizer.lastShelfCount}  3D: ${mapper.state.value.status}"
            debug.motion = "%.1fm/s %.0f°/s".format(speedMps, turnDps)
            Log.i(TAG, "scene: detect ${detectMs}ms, name ${recognizeMs}ms, boxes ${boxes.size}, named ${products.size} " +
                "(by shelf ${recognizer.lastShelfCount}), walls ${walls.size}, " +
                "shelves ${shelves.joinToString("|") { it.category }}, near ${reading?.describe() ?: "-"}")
            val camera = cameraPosition(f)
            if (judge.scene != Scene.ONE_SHELF && leftOneShelfAt == 0L) leftOneShelfAt = now
            if (judge.scene == Scene.ONE_SHELF) leftOneShelfAt = 0L
            if (leftOneShelfAt != 0L && now - leftOneShelfAt > FORGET_DESCRIPTION_MS) forgetDescription()

            // 説明し直すのは、説明した場所から MOVE_TO_REDESCRIBE_M 離れたとき（まだ説明していなければすぐ）か、別の種類の棚に移ったとき（iOS 版と同じ）
            val kinds = if (judge.scene == Scene.ONE_SHELF) ShelfDescriber.split(products, groups).mainKinds.toSet() else emptySet()
            val otherShelf = shelfChange.update(kinds, describedKinds, now)
            val describedPlace = describedAt
            val moved = (describedPlace?.distanceTo(camera) ?: Float.MAX_VALUE) > MOVE_TO_REDESCRIBE_M
            debug.shelfChange = shelfChange.describe(now) + (describedPlace?.let { "、説明した場所から %.1fm".format(it.distanceTo(camera)) } ?: "")
            if (auto && !voice.speaking && !describing.get() && judge.scene == Scene.ONE_SHELF &&
                now - lastDescribeTry > DESCRIBE_RETRY_MS && (moved || otherShelf)
            ) {
                lastDescribeTry = now
                describing.set(true)
                shelfChange.reset()
                Log.i(TAG, "describe trigger: ${if (moved) "moved" else "other shelf"}, ${debug.shelfChange}")
                // LLM を待つ間も解析を止めないよう、別のスレッドで話す
                val jpg = jpeg(image, NARRATE_LONG_SIDE)
                lookupWorker.execute {
                    try { describe(products, jpg, boxes.size, camera, requested = false) } finally { describing.set(false) }
                }
            }
        } catch (e: RuntimeException) {
            Log.e(TAG, "scene analysis failed", e)
        } finally {
            image.recycle()
        }
    }

    /** 前に説明した棚を忘れる（次に棚 1 つになったら、場所や種類に関係なく説明する）。 */
    private fun forgetDescription() {
        describedAt = null
        describedKinds = null
    }

    /**
     * ② 棚 1 つの商品を説明する。[products] はその場で認識したもの。
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
            describedKinds = split.mainKinds.toSet()
            lastDescription = text
            voice.say(text, interrupt = requested)
        } else if (requested) {
            voice.say("商品がよく分かりませんでした。棚の正面から、棚全体を写してください。")
        }
    }

    // ---------------------------------------------------------------- ③ ボタン: 正面の棚の種類

    private fun onShelfButton() {
        listening.cancel()
        voice.say("棚を見ています", interrupt = true, remember = false)
        pendingCapture = { f -> coarseWorker.execute { runShelfButton(f) } }
    }

    /** 撮った画像を列ごとに 7 分類し、正面・左・右の棚の種類を言う。 */
    private fun runShelfButton(f: CapturedFrame) {
        val c = coarse ?: run { debug.coarseMs = "モデルなし"; voice.say("棚の種類を見分けるモデルが入っていません。"); return }
        val image = JpegEncoder.toUprightBitmap(f, SCENE_LONG_SIDE)
        try {
            val t0 = SystemClock.elapsedRealtime()
            val r = c.classify(image)
            val segments = ShelfSegment.from(r.columns, r.confidence)
            val text = ShelfSpeech.describe(segments)
            val now = SystemClock.elapsedRealtime()
            debug.columns = r.columns
            debug.columnConfidence = r.confidence
            debug.columnFromMap = BooleanArray(r.columns.size)
            debug.shelves = segments
            debug.shelvesAt = now
            debug.coarseMs = "${now - t0}ms（ボタン）"
            debug.description = text
            Log.i(TAG, "shelf button: ${now - t0}ms, columns ${r.columns.joinToString(",")} -> $text")
            lastDescription = text
            voice.say(text, interrupt = true)
        } catch (e: RuntimeException) {
            Log.e(TAG, "shelf button failed", e)
            voice.say("棚の種類を見分けられませんでした。")
        } finally {
            image.recycle()
        }
    }

    // ---------------------------------------------------------------- ④ 長押し: 画面の中央の商品

    private fun onLongPressed() {
        listening.cancel()
        voice.say("調べています", interrupt = true, remember = false)
        pendingCapture = ::runLongPress
    }

    /**
     * 撮った画像で、手前の商品 1 つ（④）か、棚の広い範囲（②）か、その間（一歩下がってもらう）かを決める（iOS 版と同じ）。
     * 商品 1 つかどうかは、枠を使わず、画面の中央の深度だけで決める（[NearProduct]）。
     * ARCore で近い商品を判定できなかったとき（深度に対応していない機種、または深度で距離を測れなかったとき）だけ、
     * 棚でなければ画面の中央の商品の枠（[SceneJudge.centerProduct]）を調べる（[SceneJudge.focusWithFallback]）。
     */
    private fun runLongPress(f: CapturedFrame) {
        val image = JpegEncoder.toUprightBitmap(f, LOOKUP_LONG_SIDE)
        try {
            val boxes = recognizer.detect(image).map { it.first }
            val now = SystemClock.elapsedRealtime()
            val reading = if (depthSupported) NearProduct.measure(f.geometry) else null
            val near = reading?.pick
            val (focus, productBox, byCenterBox) = SceneJudge.focusWithFallback(boxes, near?.box, depthMeasured = reading?.centerM != null)
            // 画面の中央の枠で選んだときは接近が分からないので、押すたびに調べ直す
            if (byCenterBox) lookup.newApproach()
            val nearText = (reading?.describe() ?: "深度なし") + if (byCenterBox) "、ARCore で判定できず画面の中央の枠で選んだ" else ""
            Log.i(TAG, "long press: ${boxes.size} boxes, $nearText -> $focus")
            debug.longPress = "$focus（枠 ${boxes.size} 個、$nearText）"
            debug.nearBox = near?.box
            debug.focusBox = productBox
            debug.focusAt = now
            debug.productBoxes = boxes
            debug.sceneAt = now
            when (focus) {
                Focus.PRODUCT -> lookupProduct(image, productBox!!)
                Focus.SHELF -> {
                    val products = recognizer.recognize(image, boxes, null, groups::of)
                    describe(products, jpeg(image, NARRATE_LONG_SIDE), boxes.size, null, requested = true)
                }
                // ARCore の深度は、スマホを少し動かしたときの画像の違いから推定するので、動かさないと測れないことがある
                Focus.STEP_BACK ->
                    if (reading != null && reading.centerM == null) voice.say("距離を測れませんでした。スマホを少し左右に動かしてから、もう一度長押ししてください。")
                    else voice.say("棚は一歩下がって全体を、商品はスマホに近づけて写してください。")
                Focus.NOTHING -> voice.say("商品が見えません。棚か商品の方にスマホを向けてください。")
            }
        } catch (e: RuntimeException) {
            Log.e(TAG, "long press failed", e)
            voice.say("調べられませんでした。")
        } finally {
            image.recycle()
        }
    }

    /** ④ 手前の商品を Cloud Vision API で読み、商品名とカテゴリを言う。この接近ですでに調べていれば、API を呼ばずにその結果を言う。 */
    private fun lookupProduct(image: Bitmap, box: FloatArray) {
        lookup.cachedResult?.let {
            debug.lookup = "（この接近で調べた結果を使う）"
            speakLookup(it)
            return
        }
        if (!lookup.configured) {
            voice.say("商品を調べる設定がされていません。")
            return
        }
        val crop = cropWithMargin(image, box, 0.08f)
        val t0 = SystemClock.elapsedRealtime()
        val result = try { lookup.lookup(crop) } finally { if (crop !== image) crop.recycle() }
        val ms = SystemClock.elapsedRealtime() - t0
        result.exceptionOrNull()?.let { Log.w(TAG, "lookup failed", it) }
        debug.lookup = result.fold({ (it?.describe() ?: "分からない") + "（全体 ${ms}ms）" }, { "失敗: ${it.message}（Cloud Vision ${ms}ms）" })
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
            Command.SHELVES -> onShelfButton()
            Command.DESCRIBE, Command.LOOKUP -> onLongPressed()
            Command.PAUSE -> { auto = false; updateSettings(); voice.say("お知らせを止めます。人が近づいたときの音は鳴ります。", interrupt = true) }
            Command.RESUME -> { auto = true; updateSettings(); forgetDescription(); voice.say("お知らせを再開します。", interrupt = true) }
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

    /** 棚の種類を判別し、地図で直して画面に出し、地図に足す（[COARSE_INTERVAL_MS] ごと）。 */
    private fun runShelves(f: CapturedFrame, walls: List<Wall>) {
        val c = coarse ?: run { debug.coarseMs = "モデルなし"; return }
        val image = JpegEncoder.toUprightBitmap(f, SCENE_LONG_SIDE)
        try {
            val now = SystemClock.elapsedRealtime()
            val eye = cameraPosition(f)
            val n = c.head.columns
            val positions = (0 until n).map { k -> positionOf(floatArrayOf(k.toFloat() / n, 0.3f, (k + 1f) / n, 0.7f), f, walls) }
            val priors = positions.map { p -> p?.let { shelfLabels.at(it, eye) } }
            // 地図で十分に分かっている列はモデルにかけない。ときどきは全部の列をかけ直す
            val full = now - lastFullShelves > REVERIFY_MS
            if (full) lastFullShelves = now
            val known = if (full) emptyMap() else priors.withIndex()
                .filter { (_, p) -> p != null && p.votes >= ShelfLabelMap.MIN_VOTES && p.share >= KNOWN_SHELF_SHARE }
                .associate { (k, p) -> k to p!!.label }
            if (known.size == n) {
                skipped.merge("棚の判別（地図で全部の列が分かる）", 1, Int::plus)
                debug.skipInfo = skipped.entries.joinToString("、") { "${it.key} ${it.value}" }
            }
            val r = c.classify(image, known)
            val view = mapShelves(r, f, walls, eye, now, positions, priors, known.keys)
            latestShelves = view
            latestShelvesAt = now
            debug.columns = view.columns
            debug.columnConfidence = r.confidence
            debug.columnFromMap = view.fromMap
            debug.shelves = view.segments
            debug.shelvesAt = now
            debug.coarseMs = "${SystemClock.elapsedRealtime() - now}ms（モデルにかけた列 ${n - known.size}/$n${if (full) "、やり直し" else ""}）"
            Log.i(TAG, "shelves: ${SystemClock.elapsedRealtime() - now}ms, classified ${n - known.size}/$n${if (full) " (full)" else ""}, columns ${r.columns.joinToString(",")}, " +
                "corrected by map ${view.fromMap.count { it }}, labels on map ${shelfLabels.size}")
        } catch (e: RuntimeException) {
            Log.e(TAG, "shelf classification failed", e)
        } finally {
            image.recycle()
        }
    }

    /** カメラの動きの速さを、前のフレームとの差からならして求める。 */
    private fun updateMotion(pose: com.google.ar.core.Pose, now: Long) {
        val z = pose.zAxis
        val cur = floatArrayOf(pose.tx(), pose.ty(), pose.tz(), -z[0], -z[1], -z[2])
        val prev = lastPose
        val dt = (now - lastPoseAt) / 1000f
        if (prev != null && dt in 0.005f..0.5f) {
            val v = distance(prev, cur) / dt
            val w = angleDeg(prev, cur) / dt
            speedMps += (v - speedMps) * MOTION_SMOOTHING
            turnDps += (w - turnDps) * MOTION_SMOOTHING
        }
        lastPose = cur
        lastPoseAt = now
    }

    private fun distance(a: FloatArray, b: FloatArray) =
        sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]))

    private fun angleDeg(a: FloatArray, b: FloatArray): Float {
        val d = (a[3] * b[3] + a[4] * b[4] + a[5] * b[5]).coerceIn(-1f, 1f)
        return Math.toDegrees(kotlin.math.acos(d).toDouble()).toFloat()
    }

    /** 3D の地図を、[StoreMapper.orbit] の視点で描く（GL スレッド）。棚のラベルも重ねる。 */
    private fun drawMap() {
        val map = mapRenderer ?: return
        if (viewH == 0) return
        glClearColor(0.05f, 0.05f, 0.08f, 1f)
        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        android.opengl.GLES20.glEnable(android.opengl.GLES20.GL_DEPTH_TEST)
        val eye = mapper.orbit.eye()
        val center = mapper.orbit.center()
        android.opengl.Matrix.setLookAtM(mapView, 0, eye.x, eye.y, eye.z, center.x, center.y, center.z, 0f, 1f, 0f)
        android.opengl.Matrix.perspectiveM(mapProjection, 0, MAP_FOV_DEG, viewW.toFloat() / viewH, 0.1f, 1000f)
        val vp = FloatArray(16)
        android.opengl.Matrix.multiplyMM(vp, 0, mapProjection, 0, mapView, 0)
        map.draw(mapper, vp, mapProjection[5], viewH, mapper.ceilingClipY, showTrail = true, people = emptyList(), showPoints = false)
        map.drawShelfLabels(shelfLabels.snapshot().map { it.position to (SHELF_COLORS[it.label] ?: 0xFF888888.toInt()) }, vp)
        mapOverlay.viewProjection = vp
        android.opengl.GLES20.glDisable(android.opengl.GLES20.GL_DEPTH_TEST)
        glClearColor(0f, 0f, 0f, 1f)
    }

    private fun walls(s: Session): List<Wall> = s.getAllTrackables(Plane::class.java)
        .filter { it.type == Plane.Type.VERTICAL && it.trackingState == TrackingState.TRACKING && it.subsumedBy == null }
        .map { pl ->
            val c = pl.centerPose
            val n = c.yAxis
            Wall(Vec3(c.tx(), c.ty(), c.tz()), Vec3(n[0], n[1], n[2]), maxOf(pl.extentX, pl.extentZ) / 2)
        }

    /** 1 枚の棚の種類。[columns] は地図で直した後の列、[priors] は列の場所の地図のラベル。 */
    private class ShelfView(
        val columns: List<String>, val fromMap: BooleanArray, val segments: List<ShelfSegment>,
        val priors: List<ShelfPrior?>,
    )

    /**
     * 8 列の分類を、地図に貯めた棚のラベルで直してから、この 1 枚の判別を地図に足す。
     * 足すのは直す前の判別（地図で直した結果を地図に足すと、最初の読み違いが固まってしまう）。
     */
    private fun mapShelves(
        r: CoarseResult, f: CapturedFrame, walls: List<Wall>, eye: Vec3, now: Long,
        positions: List<Vec3?>, priors: List<ShelfPrior?>, fromMapOnly: Set<Int>,
    ): ShelfView {
        val n = r.columns.size
        val (columns, fromMap) = ShelfLabelMap.correct(r.columns, priors)
        for (c in fromMapOnly) fromMap[c] = true
        for (c in 0 until n) {
            // 地図の答えをそのまま使った列は票を足さない（自分の答えで票を増やすと、読み違いが固まる）
            if (c in fromMapOnly) continue
            val p = positions[c] ?: continue
            if (r.columns[c] != ShelfSegment.NO_PRODUCT) shelfLabels.add(r.columns[c], p, eye, now)
        }
        return ShelfView(columns, fromMap, ShelfSegment.from(columns, r.confidence, fromMap), priors)
    }

    /** 商品の枠の場所の棚の種類。地図に十分な票があればそれを、なければこの 1 枚の列の分類（1 票ぶん）。 */
    private fun shelfFor(b: FloatArray, f: CapturedFrame, walls: List<Wall>, eye: Vec3, view: ShelfView?): ShelfPrior? {
        val fromMap = positionOf(b, f, walls)?.let { shelfLabels.at(it, eye) }?.takeIf { it.votes >= ShelfLabelMap.MIN_VOTES }
        if (fromMap != null) return fromMap
        val cols = view?.columns ?: return null
        val c = (((b[0] + b[2]) / 2) * cols.size).toInt().coerceIn(0, cols.size - 1)
        return cols[c].takeIf { it != ShelfSegment.NO_PRODUCT }?.let { ShelfPrior(it, 1f, 1, mapOf(it to 1f)) }
    }

    /** 枠（正立画像の 0..1）の商品の、世界座標の位置。深度か壁で決める（[ShelfMap.place]）。 */
    private fun positionOf(b: FloatArray, f: CapturedFrame, walls: List<Wall>): Vec3? {
        val g = f.geometry
        val cx = (b[0] + b[2]) / 2
        val cy = (b[1] + b[3]) / 2
        val depth = Unprojector.medianDepthMeters(NormalizedBox(b[0], b[1], b[2], b[3]), g)
        val along = Unprojector.worldAt(cx, cy, 1f, g)
        return ShelfMap.place(cameraPosition(f), along, depth?.let { Unprojector.worldAt(cx, cy, it, g) }, walls)
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

    /** Gemini（キーがなければ null）。話す文を作る・カテゴリを決めるだけなので、速い lite のモデルから試す（flash は 6〜9 秒、lite は 2 秒ほど）。 */
    private fun geminiClient(): GeminiClient? = BuildConfig.GEMINI_API_KEY.takeIf { it.isNotEmpty() }?.let { key ->
        val models = BuildConfig.GEMINI_MODELS.split(",").map(String::trim).filter(String::isNotEmpty).sortedBy { m -> "lite" !in m }
        if (models.isEmpty()) null else GeminiClient(key, models)
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
        /** 商品の解析の間隔。棚の種類（[COARSE_INTERVAL_MS]）を先に、商品は後から分かればよい。 */
        const val SCENE_INTERVAL_MS = 2000L
        /**
         * 一時的な計測用: 商品の解析だけを動かし、3D の地図・棚の種類（7 分類）・人の検出は止める。
         * 解析は前の解析が終わったらすぐ次を回し、かかった時間を画面とログ（タグ Assist）に出す。元に戻すときは false にする。
         */
        const val PRODUCT_ONLY = true
        const val DEFAULT_COLUMNS = 8
        const val MAP_FOV_DEG = 50f
        const val COARSE_INTERVAL_MS = 1000L
        /** 棚の種類を先に出したいので 4 本（Pixel 8a で最も速い）。商品の解析は優先度を下げて譲る。 */
        const val COARSE_THREADS = 4
        /** 棚の種類の判別を、商品の名前の選び直しに使える古さ。 */
        const val SHELF_FRESH_MS = 3000L
        /** これより速く動いている（歩く・振り向く）間は、商品の解析と棚の判別を休む。画像がぶれて役に立たない。 */
        const val FAST_MPS = 1.0f
        const val FAST_DPS = 60f
        const val MOTION_SMOOTHING = 0.2f
        /** 地図で分かっていても、この間隔で全部の列をやり直して地図に足す。 */
        const val REVERIFY_MS = 30_000L
        /** 棚の判別を省くのに要る、地図のいちばん多い分類の割合（列を直す [ShelfLabelMap.MIN_SHARE] より厳しく）。 */
        const val KNOWN_SHELF_SHARE = 0.7f
        const val PERSON_INTERVAL_MS = 200L
        const val SCENE_LONG_SIDE = 1280
        const val PERSON_LONG_SIDE = 640
        const val LOOKUP_LONG_SIDE = 1600
        /** 近い商品がこれだけ見えなければ、次に見えたときを新しい接近とする。 */
        const val NEAR_GAP_MS = 2000L
        const val MAX_CAMERA_IMAGE_WIDTH = 1920
        const val NARRATE_LONG_SIDE = 768
        const val QUESTION_LONG_SIDE = 1280
        const val FORGET_DESCRIPTION_MS = 3000L
        const val MOVE_TO_REDESCRIBE_M = 1.5f
        /** 棚の説明を試してから、次に試すまでの間（説明できなかったときも含む）。 */
        const val DESCRIBE_RETRY_MS = 1500L
        /** 棚の種類のボタンの高さ（dp）。 */
        const val SHELF_BUTTON_DP = 88
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
