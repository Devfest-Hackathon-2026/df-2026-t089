package com.example.store3d

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLES20.*
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableException
import java.io.File
import java.util.concurrent.Executors
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MainActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var gl: GLSurfaceView
    private lateinit var status: TextView
    private var session: Session? = null
    private var installRequested = false
    private var sensorRot = 90
    private var viewW = 0; private var viewH = 0; private var geometryChanged = false

    private val bg = CameraBackground()
    private val drawer = BoxDrawer()
    private val view = FloatArray(16); private val proj = FloatArray(16)

    private val worker = Executors.newSingleThreadExecutor()
    private val analyzer by lazy { Analyzer(this) }
    private val lock = Any()
    private var planes = emptyList<PlaneSnap>()
    private val labelRays = mutableListOf<LabelRay>()
    private val boundaryRays = mutableListOf<Ray>()
    @Volatile private var preview = emptyList<Box>()
    @Volatile private var busy = false
    private var lastShot = 0L; private var lastRecon = 0L

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        gl = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        status = TextView(this).apply { setTextColor(-1); setBackgroundColor(0x88000000.toInt()); setPadding(24, 24, 24, 24); textSize = 14f }
        val bar = LinearLayout(this).apply {
            gravity = Gravity.CENTER
            addView(Button(context).apply { text = "完了して3D化"; setOnClickListener { finishScan() } })
            addView(Button(context).apply { text = "前回のモデル"; setOnClickListener { openLatest() } })
        }
        setContentView(FrameLayout(this).apply {
            addView(gl)
            addView(status, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))
            addView(bar, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM).apply { bottomMargin = 48 })
        })
        status.text = "棚の正面にカメラを向け、通路をゆっくり歩いてください"
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 0); return
        }
        if (session == null) try {
            if (ArCoreApk.getInstance().requestInstall(this, !installRequested) == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                installRequested = true; return
            }
            session = Session(this).also { s ->
                s.configure(Config(s).setPlaneFindingMode(Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL).setFocusMode(Config.FocusMode.AUTO))
                val cm = getSystemService(CameraManager::class.java)
                sensorRot = cm.getCameraCharacteristics(s.cameraConfig.cameraId).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            }
        } catch (e: UnavailableException) {
            Toast.makeText(this, "ARCoreが使えません: $e", Toast.LENGTH_LONG).show(); finish(); return
        }
        session!!.resume()
        gl.onResume()
    }

    override fun onPause() {
        super.onPause()
        gl.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.shutdown()
        session?.close()
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        if (res.firstOrNull() != PackageManager.PERMISSION_GRANTED) { Toast.makeText(this, "カメラ権限が必要です", Toast.LENGTH_LONG).show(); finish() }
    }

    override fun onSurfaceCreated(g: GL10?, c: EGLConfig?) {
        glClearColor(0f, 0f, 0f, 1f)
        bg.init(); drawer.init()
    }

    override fun onSurfaceChanged(g: GL10?, w: Int, h: Int) {
        glViewport(0, 0, w, h); viewW = w; viewH = h; geometryChanged = true
    }

    override fun onDrawFrame(g: GL10?) {
        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        s.setCameraTextureName(bg.texId)
        if (geometryChanged) { s.setDisplayGeometry(display!!.rotation, viewW, viewH); geometryChanged = false }
        val frame = try { s.update() } catch (e: Exception) { return }
        bg.draw(frame)
        val cam = frame.camera
        if (cam.trackingState != TrackingState.TRACKING) return
        cam.getProjectionMatrix(proj, 0, 0.05f, 50f)
        cam.getViewMatrix(view, 0)
        drawer.draw(preview, view, proj, 0.45f)

        val now = SystemClock.uptimeMillis()
        if (busy || now - lastShot < SHOT_INTERVAL_MS) return
        val shot = try {
            frame.acquireCameraImage().use { img ->
                val k = cam.imageIntrinsics
                FrameShot(yuvToArgb(img), img.width, img.height, sensorRot, cam.pose, k.focalLength[0], k.focalLength[1], k.principalPoint[0], k.principalPoint[1])
            }
        } catch (e: NotYetAvailableException) { return }
        val snap = s.getAllTrackables(Plane::class.java)
            .filter { it.trackingState != TrackingState.STOPPED && it.subsumedBy == null && it.type != Plane.Type.HORIZONTAL_DOWNWARD_FACING }
            .map(::snapshot)
        lastShot = now; busy = true
        worker.execute { process(shot, snap) }
    }

    private fun snapshot(p: Plane): PlaneSnap {
        val poly = p.polygon.let { b -> FloatArray(b.limit()).also { b.rewind(); b.get(it) } }
        val pose = p.centerPose
        val world = FloatArray(poly.size / 2 * 3)
        for (i in 0 until poly.size / 2) pose.transformPoint(floatArrayOf(poly[i * 2], 0f, poly[i * 2 + 1])).copyInto(world, i * 3)
        return PlaneSnap(p.type == Plane.Type.VERTICAL, world, pose.yAxis)
    }

    private fun process(shot: FrameShot, snap: List<PlaneSnap>) {
        try {
            val r = analyzer.analyze(shot)
            val n: Int
            synchronized(lock) { planes = snap; labelRays += r.labelRays; boundaryRays += r.boundaryRays; n = labelRays.size }
            val now = SystemClock.uptimeMillis()
            if (now - lastRecon > RECON_INTERVAL_MS) { lastRecon = now; preview = rebuild() }
            runOnUiThread {
                status.text = "平面 ${snap.size}  観測 $n  棚 ${preview.count { it.kind != Kind.FLOOR }}\n${r.summary}"
            }
        } finally { busy = false }
    }

    private fun rebuild() = synchronized(lock) { Reconstructor.build(planes, labelRays.toList(), boundaryRays.toList(), analyzer.labels) }

    private fun finishScan() = worker.execute {
        val boxes = rebuild()
        if (boxes.isEmpty()) { runOnUiThread { Toast.makeText(this, "まだ平面が見つかっていません", Toast.LENGTH_SHORT).show() }; return@execute }
        val dir = File(getExternalFilesDir(null), "scan_${System.currentTimeMillis()}")
        synchronized(lock) { Storage.save(dir, planes, labelRays, boundaryRays, analyzer.labels, boxes) }
        runOnUiThread { startActivity(Intent(this, ViewerActivity::class.java).putExtra("dir", dir.path)) }
    }

    private fun openLatest() {
        val dir = getExternalFilesDir(null)?.listFiles { f -> File(f, "model.json").exists() }?.maxByOrNull { it.name }
        if (dir == null) Toast.makeText(this, "保存済みモデルはありません", Toast.LENGTH_SHORT).show()
        else startActivity(Intent(this, ViewerActivity::class.java).putExtra("dir", dir.path))
    }

    companion object {
        const val SHOT_INTERVAL_MS = 500L
        const val RECON_INTERVAL_MS = 2000L
    }
}
