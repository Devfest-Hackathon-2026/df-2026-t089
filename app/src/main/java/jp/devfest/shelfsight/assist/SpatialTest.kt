package jp.devfest.shelfsight.assist

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/** 立体音響のテストの音源。同じ場所で [CHIMES] 回鳴らし、次の場所へゆっくり移る、を繰り返す。 */
internal object SpatialDemo {
    const val CHIMES = 5
    private const val CHIME_S = 0.7
    private const val HOLD_S = CHIMES * CHIME_S
    private const val MOVE_S = 3.0
    private const val CYCLE_S = HOLD_S + MOVE_S

    /** 回る場所（右が正の方位 °、上が正の仰角 °、距離 m）。隣どうしは少しずつ離し、最初は正面にして画面に映す。 */
    private val SPOTS = listOf(
        Triple(0.0, 0.0, 2.0),
        Triple(40.0, 25.0, 1.5),
        Triple(100.0, -20.0, 2.5),
        Triple(170.0, 10.0, 1.5),
        Triple(-120.0, 35.0, 3.0),
        Triple(-50.0, -25.0, 1.0),
        Triple(-10.0, 40.0, 2.5),
        Triple(20.0, -30.0, 3.5),
    )
    val SPOT_COUNT = SPOTS.size

    /** 今鳴らす番号（鳴らす回ごとに変わる）。移動中は null。 */
    fun chimeSlot(seconds: Double): Int? {
        val cycle = floor(seconds / CYCLE_S)
        val phase = seconds - cycle * CYCLE_S
        return if (phase < HOLD_S) cycle.toInt() * CHIMES + (phase / CHIME_S).toInt() else null
    }

    /** 最初に向いていた方向を基準にした音源の位置（x が右、y が上、z が正面、m）。 */
    fun offset(seconds: Double): Vec3 {
        val cycle = floor(seconds / CYCLE_S)
        val phase = seconds - cycle * CYCLE_S
        val from = SPOTS[cycle.toInt() % SPOTS.size]
        val to = SPOTS[(cycle.toInt() + 1) % SPOTS.size]
        val t = ((phase - HOLD_S) / MOVE_S).coerceIn(0.0, 1.0).let { it * it * (3 - 2 * it) }   // 動き始めと止まる前をゆっくりに
        val turn = ((to.first - from.first + 540.0) % 360.0) - 180.0   // 近い方の回りで移る
        val azimuth = Math.toRadians(from.first + turn * t)
        val elevation = Math.toRadians(from.second + (to.second - from.second) * t)
        val distance = from.third + (to.third - from.third) * t
        val horizontal = distance * cos(elevation)
        return Vec3((horizontal * sin(azimuth)).toFloat(), (distance * sin(elevation)).toFloat(), (horizontal * cos(azimuth)).toFloat())
    }

    /** [origin] の場所と向きを基準にした世界座標。 */
    fun world(origin: Listener, seconds: Double): Vec3 {
        val o = offset(seconds)
        val p = origin.position
        return Vec3(p.x - origin.forwardZ * o.x + origin.forwardX * o.z, p.y + o.y, p.z + origin.forwardX * o.x + origin.forwardZ * o.z)
    }
}

/** カメラ映像に重ねて、音源の場所に印を出す。画面の外にあるときは、その方向の端に矢印を出す。 */
internal class SpatialTestView(context: Context, private val state: () -> Pair<Listener, Vec3>, private val sounding: () -> Boolean) : View(context) {
    /** 世界座標 → クリップ座標の行列。ARCore が追跡していなければ null。 */
    @Volatile var viewProjection: FloatArray? = null
    private val density = resources.displayMetrics.density
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3 * density }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 20 * resources.displayMetrics.scaledDensity; textAlign = Paint.Align.CENTER }
    private val strip = Paint().apply { color = 0xB0000000.toInt() }
    private val arrow = Path()

    init {
        contentDescription = "立体音響のテスト。同じ場所から5回鳴らしたあと、ゆっくり別の場所へ移ります"
        isClickable = true   // 下の画面の「たたく」操作に渡さない
    }

    override fun onDraw(canvas: Canvas) {
        val (l, position) = state()
        val color = if (sounding()) 0xFFFFD23F.toInt() else 0xFFAEB4BD.toInt()
        fill.color = color
        ring.color = color
        viewProjection?.let { drawMarker(canvas, it, position) }

        val az = Binaural.azimuth(l, position)
        val horizontal = Binaural.horizontalDistance(l, position)
        val rise = position.y - l.position.y
        val deg = Math.toDegrees(az.toDouble()).roundToInt()
        val up = Math.toDegrees(atan2(rise, horizontal).toDouble()).roundToInt()
        val bottom = height - 40 * density
        canvas.drawRect(0f, bottom - 44 * density, width.toFloat(), bottom, strip)
        val where = "${if (deg >= 0) "右" else "左"} ${abs(deg)}°   ${if (up >= 0) "上" else "下"} ${abs(up)}°   ${"%.1f".format(sqrt(horizontal * horizontal + rise * rise))}m"
        canvas.drawText(if (viewProjection == null) "位置合わせ中  $where" else where, width / 2f, bottom - 14 * density, text)
        if (isShown) postInvalidateOnAnimation()
    }

    private fun drawMarker(canvas: Canvas, m: FloatArray, p: Vec3) {
        val w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15]
        val x = (m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]) / w
        val y = (m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]) / w
        if (w > 0f && abs(x) <= 1f && abs(y) <= 1f) {
            val radius = (28f / w).coerceIn(9f, 40f) * density
            val sx = (x + 1f) / 2f * width
            val sy = (1f - y) / 2f * height
            canvas.drawCircle(sx, sy, radius, fill)
            canvas.drawCircle(sx, sy, radius * 1.7f, ring)
            return
        }
        // 後ろにあるときは投影が反転するので、向きを戻す
        val sign = if (w > 0f) 1f else -1f
        val dx = x * sign * width
        val dy = -y * sign * height
        val inset = 44 * density
        val halfW = width / 2f - inset
        val halfH = height / 2f - TOP_INSET_DP * density
        val k = min(halfW / max(abs(dx), 1f), halfH / max(abs(dy), 1f))
        canvas.save()
        canvas.translate(width / 2f + dx * k, height / 2f + dy * k)
        canvas.rotate(Math.toDegrees(atan2(dy, dx).toDouble()).toFloat())
        val s = 26 * density
        arrow.reset(); arrow.moveTo(s, 0f); arrow.lineTo(-s * 0.6f, s * 0.7f); arrow.lineTo(-s * 0.6f, -s * 0.7f); arrow.close()
        canvas.drawPath(arrow, fill)
        canvas.restore()
    }

    private companion object {
        const val TOP_INSET_DP = 110
    }
}
