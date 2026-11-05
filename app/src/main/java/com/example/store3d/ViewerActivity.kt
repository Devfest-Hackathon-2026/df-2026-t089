package com.example.store3d

import android.app.Activity
import android.opengl.GLES20.*
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.style.ForegroundColorSpan
import android.view.Gravity
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.widget.FrameLayout
import android.widget.TextView
import java.io.File
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/** Orbit viewer for a saved model.json (drag = rotate, pinch = zoom). */
class ViewerActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var boxes: List<Box>
    private val drawer = BoxDrawer()
    private val view = FloatArray(16); private val proj = FloatArray(16)
    private val target = FloatArray(3)
    @Volatile private var yaw = 0.6f
    @Volatile private var pitch = 0.9f
    @Volatile private var dist = 8f
    private var lastX = 0f; private var lastY = 0f

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        val dir = File(intent.getStringExtra("dir")!!)
        boxes = Storage.loadModel(File(dir, "model.json"))
        boxes.firstOrNull { it.kind == Kind.FLOOR }?.let {
            it.center.copyInto(target); dist = max(it.size[0], it.size[2]) * 1.1f
        }
        val scale = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean { dist = (dist / d.scaleFactor).coerceIn(1f, 100f); return true }
        })
        val gl = object : GLSurfaceView(this) {
            override fun onTouchEvent(e: MotionEvent): Boolean {
                scale.onTouchEvent(e)
                if (e.actionMasked == MotionEvent.ACTION_MOVE && e.pointerCount == 1 && !scale.isInProgress) {
                    yaw -= (e.x - lastX) * 0.01f
                    pitch = (pitch + (e.y - lastY) * 0.01f).coerceIn(0.1f, 1.5f)
                }
                lastX = e.x; lastY = e.y
                return true
            }
        }.apply {
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@ViewerActivity)
        }
        val counts = boxes.filter { it.kind != Kind.FLOOR }.groupingBy { it.label }.eachCount()
        val legend = SpannableStringBuilder("${dir.name}\n${dir.path}\n")
        counts.forEach { (l, n) ->
            val s = legend.length; legend.append("■ "); legend.setSpan(ForegroundColorSpan(colorOf(l)), s, s + 1, 0); legend.append("$l ×$n\n")
        }
        val text = TextView(this).apply { this.text = legend; setTextColor(-1); setBackgroundColor(0x88000000.toInt()); setPadding(24, 24, 24, 24); textSize = 12f }
        setContentView(FrameLayout(this).apply {
            addView(gl)
            addView(text, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.START))
        })
    }

    override fun onSurfaceCreated(g: GL10?, c: EGLConfig?) {
        glClearColor(0.12f, 0.13f, 0.15f, 1f)
        glEnable(GL_DEPTH_TEST)
        drawer.init()
    }

    override fun onSurfaceChanged(g: GL10?, w: Int, h: Int) {
        glViewport(0, 0, w, h)
        Matrix.perspectiveM(proj, 0, 50f, w / h.toFloat(), 0.1f, 200f)
    }

    override fun onDrawFrame(g: GL10?) {
        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        val ex = target[0] + dist * cos(pitch) * sin(yaw)
        val ey = target[1] + dist * sin(pitch)
        val ez = target[2] + dist * cos(pitch) * cos(yaw)
        Matrix.setLookAtM(view, 0, ex, ey, ez, target[0], target[1], target[2], 0f, 1f, 0f)
        drawer.draw(boxes, view, proj, 1f)
    }
}
