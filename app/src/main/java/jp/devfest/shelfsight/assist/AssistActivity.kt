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
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import com.example.store3d.CameraBackground
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.UnavailableException
import jp.devfest.shelfsight.BuildConfig
import jp.devfest.shelfsight.ar.CapturedFrame
import jp.devfest.shelfsight.ar.FrameCapture
import jp.devfest.shelfsight.ar.JpegEncoder
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.perception.MediaPipePersonDetector
import jp.devfest.shelfsight.perception.PersonDistance
import jp.devfest.shelfsight.perception.PersonMotion
import jp.devfest.shelfsight.perception.PersonObservation
import jp.devfest.shelfsight.perception.PersonTracker
import jp.devfest.shelfsight.walking.CoarseClassifier
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sqrt

/**
 * 視覚障害のある人のための、コンビニの棚と商品の案内。胸の前にスマホを立てて持ち、イヤホンで聞く。
 *
 * ① 棚が複数写っている: 歩行中の判別（画面を 8 列に分け、列ごとに 7 分類）で、カテゴリが変わる所を棚の境界とし、
 *    それぞれの棚の方向から棚の種類を読み上げる
 * ② 棚が 1 つ写っている: 商品を認識し、段ごとに何があるかを話す（場違いな商品は言わない）
 * ③ 商品が 1 つ大きく写っている: 長押しで Cloud Vision API に聞き、社名と商品名を言う（1 回の接近で 1 回だけ呼ぶ）
 * ④ 人が近づいてくる: その人の方向からピコピコ鳴らす
 *
 * 操作は、長押し（商品を調べる）と、タップ（話しかける）だけ。TalkBack が有効なら、ダブルタップとダブルタップ長押しになる。
 */
class AssistActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var gl: GLSurfaceView
    private lateinit var status: TextView
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
    @Volatile private var lookupRequested = false

    /** 歩行中の判別のモデル。大きいのでリポジトリに含めておらず、入っていないビルドでは null（① だけが止まる）。 */
    private val coarse by lazy { CoarseClassifier.fromAssets(this, BuildConfig.COARSE_MODEL) }
    private val recognizer by lazy { ProductRecognizer(this) }
    private val personDetector by lazy { MediaPipePersonDetector(this) }
    private val tracker = PersonTracker()
    private val lookup = ProductLookup(BuildConfig.CLOUD_VISION_API_KEY)
    private val judge = SceneJudge()
    private val shelfMemory = ShelfMemory()

    private var lastScene = 0L
    private var lastPerson = 0L
    private var greeted = false
    @Volatile private var auto = true
    @Volatile private var describeRequested = false
    @Volatile private var shelvesRequested = false
    /** ② を話した場所。ここから離れるか、棚 1 つの場面を離れるまでは繰り返さない。 */
    private var describedAt: Vec3? = null
    private var leftOneShelfAt = 0L
    private var lastDescribeTry = 0L
    /** 今の接近で、大きく写った商品を最後に見た時刻と、そのときの場所。 */
    private var approachSeenAt = 0L
    private var approachFrom: Vec3? = null
    /** 最後に見えた、大きく写った商品の枠と、その時刻。 */
    @Volatile private var mainProduct: Pair<FloatArray, Long>? = null
    @Volatile private var latestShelves: List<PlacedShelf> = emptyList()

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
        status = TextView(this).apply {
            setTextColor(-1); setBackgroundColor(0x88000000.toInt()); setPadding(24, 24, 24, 24); textSize = 14f
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        // 画面全体を 1 つのボタンにする（どこを触ってもよい）
        val touch = View(this).apply {
            contentDescription = "タップで話しかける。長押しで商品を調べる"
            isClickable = true
            isLongClickable = true
            setOnClickListener { startListening() }
            setOnLongClickListener { requestLookup(); true }
        }
        setContentView(FrameLayout(this).apply {
            addView(gl)
            addView(status, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))
            addView(touch, FrameLayout.LayoutParams(-1, -1))
        })
        status.text = "準備しています"
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
                val config = Config(s).setFocusMode(Config.FocusMode.AUTO).setPlaneFindingMode(Config.PlaneFindingMode.DISABLED)
                if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) config.depthMode = Config.DepthMode.AUTOMATIC
                s.configure(config)
                val cm = getSystemService(CameraManager::class.java)
                sensorRotation = cm.getCameraCharacteristics(s.cameraConfig.cameraId).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            }
        } catch (e: UnavailableException) {
            voice.say("このスマホでは、カメラの位置合わせが使えません。")
            status.text = "ARCore が使えません: $e"
            return
        }
        session!!.resume()
        gl.onResume()
        if (!greeted) {
            greeted = true
            sceneWorker.execute { if (coarse == null) Log.w(TAG, "coarse/${BuildConfig.COARSE_MODEL}.onnx is missing: shelf categories are off") }
            voice.say("準備ができました。画面をタップすると話しかけられます。長押しで、商品を調べます。")
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
        if (camera.trackingState != TrackingState.TRACKING) return

        // 聞く人の向き: スマホを胸の前に立てて持つので、カメラの前（-z）を水平にしたもの
        val pose = camera.pose
        val z = pose.zAxis
        val fx = -z[0]
        val fz = -z[2]
        val len = sqrt(fx * fx + fz * fz)
        if (len > 0.3f) listener = Listener(Vec3(pose.tx(), pose.ty(), pose.tz()), fx / len, fz / len)

        val now = SystemClock.elapsedRealtime()
        val rotation = sensorRotation   // 画面は縦に固定している
        if (lookupRequested) {
            FrameCapture.capture(frame, camera, rotation, depthEnabled = false, fullResolution = true)?.let { f ->
                lookupRequested = false
                lookupWorker.execute { runLookup(f) }
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
                sceneWorker.execute { try { runScene(f) } finally { sceneBusy.set(false) } }
            }
        }
    }

    // ---------------------------------------------------------------- ①② 写っているものの判定

    private fun runScene(f: CapturedFrame) {
        val image = JpegEncoder.toUprightBitmap(f, SCENE_LONG_SIDE)
        try {
            val now = SystemClock.elapsedRealtime()
            val shelves = coarse?.classify(image)?.let { r -> ShelfMemory.shelves(r.columns, r.confidence) }.orEmpty().map { s ->
                s.copy(position = Unprojector.unproject(NormalizedBox(s.left, 0.3f, s.right, 0.7f), f.geometry).position)
            }
            val boxes = recognizer.detect(image).map { it.first }
            val switched = judge.update(SceneEvidence(shelves.size, boxes))
            val main = SceneJudge.mainProduct(boxes)
            main?.let { mainProduct = it to now }
            latestShelves = shelves
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

            val forceShelves = shelvesRequested.also { shelvesRequested = false }
            val forceDescribe = describeRequested.also { describeRequested = false }
            when {
                forceDescribe -> describe(image, boxes, camera, requested = true)
                forceShelves -> announceShelves(shelves, requested = true)
                !auto || voice.speaking -> Unit
                judge.scene == Scene.SHELVES -> announceShelves(shelves, requested = false)
                judge.scene == Scene.ONE_SHELF && now - lastDescribeTry > DESCRIBE_RETRY_MS &&
                    (describedAt?.distanceTo(camera) ?: Float.MAX_VALUE) > MOVE_TO_REDESCRIBE_M -> {
                    lastDescribeTry = now
                    describe(image, boxes, camera, requested = false)
                }
            }
            runOnUiThread {
                status.text = "場面: ${judge.scene}  棚: ${shelves.joinToString(" | ") { it.category }}  商品の枠: ${boxes.size}"
            }
        } catch (e: RuntimeException) {
            Log.e(TAG, "scene analysis failed", e)
        } finally {
            image.recycle()
        }
    }

    /** ① 棚の種類を、それぞれの棚の方向から言う。最近同じ場所で言った棚は言わない。 */
    private fun announceShelves(shelves: List<PlacedShelf>, requested: Boolean) {
        val now = SystemClock.elapsedRealtime()
        val toSay = if (requested) shelves else shelves.filter { shelfMemory.isNew(it, now) }
        if (toSay.isEmpty()) {
            if (requested) voice.say("棚が見つかりません。棚の方にスマホを向けてください。", interrupt = true)
            return
        }
        // 左から順に、その位置から聞こえるように
        toSay.sortedBy { it.left }.forEachIndexed { i, s ->
            shelfMemory.remember(s, now)
            val pos = s.position
            voice.say(s.spoken, position = { pos }, interrupt = requested && i == 0)
        }
    }

    /** ② 棚 1 つの商品を、段ごとに説明する。 */
    private fun describe(image: Bitmap, boxes: List<FloatArray>, camera: Vec3, requested: Boolean) {
        if (requested) voice.say("見ています", interrupt = true, remember = false)
        val products = recognizer.recognize(image, boxes)
        val text = ShelfDescriber.describe(products)
        Log.i(TAG, "describe: ${products.size} products -> $text")
        if (text != null) {
            describedAt = camera
            voice.say(text, interrupt = requested)
        } else if (requested) {
            voice.say("商品がよく分かりませんでした。棚の正面から、棚全体を写してください。")
        }
    }

    // ---------------------------------------------------------------- ③ 商品を調べる

    private fun requestLookup() {
        listening.cancel()
        lookup.cachedResult?.let { cached ->
            speakLookup(cached, interrupt = true)   // この接近ですでに調べてある
            return
        }
        if (!lookup.configured) {
            voice.say("商品を調べる設定がされていません。", interrupt = true)
            return
        }
        voice.say("調べています", interrupt = true, remember = false)
        lookupRequested = true
    }

    private fun runLookup(f: CapturedFrame) {
        val image = JpegEncoder.toUprightBitmap(f, LOOKUP_LONG_SIDE)
        try {
            // 大きく写った商品だけを送る（最近見つかっていなければ、画面の真ん中）
            val box = mainProduct?.takeIf { SystemClock.elapsedRealtime() - it.second < MAIN_PRODUCT_FRESH_MS }?.first
                ?: floatArrayOf(0.15f, 0.15f, 0.85f, 0.85f)
            val crop = cropWithMargin(image, box, 0.08f)
            val result = try { lookup.lookup(crop) } finally { if (crop !== image) crop.recycle() }
            result.exceptionOrNull()?.let { Log.w(TAG, "lookup failed", it) }
            speakLookup(result, interrupt = false)
        } finally {
            image.recycle()
        }
    }

    private fun speakLookup(result: Result<ProductAnswer?>, interrupt: Boolean) {
        val text = result.fold(
            onSuccess = { it?.spoken() ?: "商品が分かりませんでした。商品の正面を、カメラに向けてください。" },
            onFailure = { "調べられませんでした。通信を確認してください。" },
        )
        voice.say(text, interrupt = interrupt)
    }

    // ---------------------------------------------------------------- ④ 人の接近

    private fun runPeople(f: CapturedFrame) {
        val image = JpegEncoder.toUprightBitmap(f, PERSON_LONG_SIDE)
        val boxes = try { personDetector.detect(image) } catch (e: RuntimeException) {
            Log.e(TAG, "person detection failed", e); emptyList()
        } finally { image.recycle() }
        val t = SystemClock.elapsedRealtime() / 1000.0
        val camera = cameraPosition(f)
        val observations = boxes.map { box ->
            val d = PersonDistance.estimate(box, f.geometry)
            val pos = d?.let { Unprojector.worldAt((box.left + box.right) / 2f, (box.top + box.bottom) / 2f, it, f.geometry) }
            PersonObservation(box, d, pos)
        }
        tracker.update(t, observations, camera)
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
    }

    // ---------------------------------------------------------------- 声の操作

    private fun startListening() {
        if (!listening.available) {
            voice.say("このスマホでは、音声認識が使えません。", interrupt = true)
            return
        }
        voice.stop()
        listening.listen(onReady = { mixer.play(SpatialMixer.Sound(SpatialMixer.makeTick(), gain = 0.6f)) }) { text ->
            Log.i(TAG, "heard: $text")
            handle(text?.let(Commands::parse), text)
        }
    }

    private fun handle(command: Command?, heard: String?) {
        when (command) {
            Command.SHELVES -> { shelvesRequested = true; if (latestShelves.isEmpty()) voice.say("見ています", interrupt = true, remember = false) }
            Command.DESCRIBE -> describeRequested = true
            Command.LOOKUP -> requestLookup()
            Command.PAUSE -> { auto = false; voice.say("お知らせを止めます。人が近づいたときの音は鳴ります。", interrupt = true) }
            Command.RESUME -> { auto = true; shelfMemory.clear(); describedAt = null; voice.say("お知らせを再開します。", interrupt = true) }
            Command.REPEAT -> voice.say(voice.lastSpoken.ifEmpty { "まだ何も話していません。" }, interrupt = true)
            Command.HELP -> voice.say(Commands.HELP_TEXT, interrupt = true)
            null -> voice.say(if (heard == null) "聞き取れませんでした。" else "分かりませんでした。「何がある？」「これは何？」のように話しかけてください。", interrupt = true)
        }
    }

    // ---------------------------------------------------------------- 補助

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
        const val SCENE_INTERVAL_MS = 900L
        const val PERSON_INTERVAL_MS = 200L
        const val SCENE_LONG_SIDE = 1280
        const val PERSON_LONG_SIDE = 640
        const val LOOKUP_LONG_SIDE = 1600
        const val MAX_CAMERA_IMAGE_WIDTH = 1920
        const val MAIN_PRODUCT_FRESH_MS = 2500L
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
