package jp.devfest.shelfsight.assist

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.opengl.Matrix
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import jp.devfest.shelfsight.mapping.OrbitCamera
import jp.devfest.shelfsight.model.Vec3

/**
 * 3D の地図（店の面と、棚の種類のラベル）を見ている間、カメラ映像の画面の上に重ねる透明な層。
 * 指で視点を動かし（1 本指で回す、2 本指で拡大・縮小と移動）、棚のラベルの名前と凡例を文字で描く。
 * 地図そのものは GL スレッドが [OrbitCamera] の視点で描く。
 */
class Map3DOverlay(
    context: Context,
    private val orbit: OrbitCamera,
    private val labels: () -> List<ShelfMark>,
    private val status: () -> String,
) : View(context) {
    /** GL スレッドが最後に描いたときの、世界 → 画面の行列。ラベルの文字の位置を合わせる。 */
    @Volatile var viewProjection: FloatArray? = null

    private val density = resources.displayMetrics.density
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 12 * density }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 10 * density }
    private var lastX = 0f
    private var lastY = 0f
    private var pointers = 0
    private val pinch = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean { orbit.zoom(d.scaleFactor); return true }
    })

    init {
        isClickable = true   // 地図を見ている間は、カメラの画面の操作（長押し・2 回たたく）にしない
    }

    @android.annotation.SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        pinch.onTouchEvent(e)
        val x = (0 until e.pointerCount).map { e.getX(it) }.average().toFloat()
        val y = (0 until e.pointerCount).map { e.getY(it) }.average().toFloat()
        if (e.actionMasked == MotionEvent.ACTION_MOVE && e.pointerCount == pointers) {
            if (e.pointerCount == 1) orbit.rotate(x - lastX, y - lastY) else orbit.pan(x - lastX, y - lastY)
        }
        pointers = if (e.actionMasked == MotionEvent.ACTION_UP || e.actionMasked == MotionEvent.ACTION_POINTER_UP) -1 else e.pointerCount
        lastX = x
        lastY = y
        return true
    }

    override fun onDraw(canvas: Canvas) {
        val marks = labels()
        val vp = viewProjection
        if (vp != null) for (m in marks) {
            val s = toScreen(m.position, vp) ?: continue
            val label = "${m.label.take(2)} ${(m.share * 100).toInt()}%/${m.total}"
            fill.color = (SHELF_COLORS[m.label] ?: Color.DKGRAY) and 0x00FFFFFF or 0xC0000000.toInt()
            val w = small.measureText(label)
            canvas.drawRect(s.first + 8 * density, s.second - small.textSize, s.first + 12 * density + w, s.second + 4 * density, fill)
            canvas.drawText(label, s.first + 10 * density, s.second, small)
        }
        // 凡例
        // ステータスバーと「閉じる」ボタンの下から
        val top = 92 * density
        fill.color = 0xB0000000.toInt()
        canvas.drawRect(0f, top, width.toFloat(), top + 64 * density, fill)
        canvas.drawText(status(), 12 * density, top + 20 * density, text)
        val counts = marks.groupingBy { it.label }.eachCount()
        var x = 12 * density
        var y = top + 40 * density
        for ((name, color) in SHELF_COLORS) {
            if (name == ShelfSegment.NO_PRODUCT) continue
            fill.color = color
            canvas.drawCircle(x + 5 * density, y - 4 * density, 5 * density, fill)
            val label = "$name ${counts[name] ?: 0}"
            canvas.drawText(label, x + 13 * density, y, small)
            x += small.measureText(label) + 24 * density
            if (x > width - 140 * density) { x = 12 * density; y += 16 * density }
        }
        postInvalidateDelayed(REFRESH_MS)
    }

    private fun toScreen(p: Vec3, vp: FloatArray): Pair<Float, Float>? {
        val out = FloatArray(4)
        Matrix.multiplyMV(out, 0, vp, 0, floatArrayOf(p.x, p.y, p.z, 1f), 0)
        if (out[3] <= 0f) return null
        val nx = out[0] / out[3]
        val ny = out[1] / out[3]
        if (nx !in -1f..1f || ny !in -1f..1f) return null
        return (nx + 1f) / 2f * width to (1f - ny) / 2f * height
    }

    private companion object {
        const val REFRESH_MS = 200L
    }
}
