package jp.devfest.shelfsight.assist

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import jp.devfest.shelfsight.model.Vec3
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/** 立体音響のテストで鳴らす音源の動き。周りを回りながら、上下し、遠ざかったり近づいたりする。 */
internal object SpatialDemo {
    private const val ORBIT_S = 8.0
    private const val ELEVATION_S = 5.0
    private const val DISTANCE_S = 11.0
    private const val MAX_ELEVATION_DEG = 50.0
    const val MIN_DISTANCE_M = 0.7
    const val MAX_DISTANCE_M = 4.0

    /** 聞く人から見た音源の位置（m）。世界に固定した向きで回るので、スマホを回すと聞こえる方向も変わる。 */
    fun offset(seconds: Double): Vec3 {
        val azimuth = 2 * PI * seconds / ORBIT_S
        val elevation = Math.toRadians(MAX_ELEVATION_DEG) * sin(2 * PI * seconds / ELEVATION_S)
        val middle = (MIN_DISTANCE_M + MAX_DISTANCE_M) / 2
        val distance = middle + (MAX_DISTANCE_M - MIN_DISTANCE_M) / 2 * sin(2 * PI * seconds / DISTANCE_S)
        val horizontal = distance * cos(elevation)
        return Vec3((horizontal * sin(azimuth)).toFloat(), (distance * sin(elevation)).toFloat(), (-horizontal * cos(azimuth)).toFloat())
    }
}

/** テスト中の音源を、上から見た図（上が聞く人の正面）と高さの棒で見せる。 */
internal class SpatialTestView(context: Context, private val state: () -> Pair<Listener, Vec3>) : View(context) {
    private val density = resources.displayMetrics.density
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF565D68.toInt(); style = Paint.Style.STROKE; strokeWidth = density }
    private val me = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFF0EFEC.toInt() }
    private val source = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFFFD23F.toInt() }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 22 * resources.displayMetrics.scaledDensity; textAlign = Paint.Align.CENTER }
    private val small = Paint(text).apply { textSize = 13 * resources.displayMetrics.scaledDensity; color = 0xFFAEB4BD.toInt() }
    private val head = Path()

    init {
        setBackgroundColor(0xFF14171C.toInt())
        contentDescription = "立体音響のテスト。音があなたの周りを回りながら、上下し、遠ざかったり近づいたりします"
        isClickable = true   // 下の画面の「たたく」操作に渡さない
    }

    override fun onDraw(canvas: Canvas) {
        val (l, position) = state()
        val cx = width / 2f
        val cy = height * 0.42f
        val radius = width * 0.40f
        val perMeter = radius / SpatialDemo.MAX_DISTANCE_M.toFloat()
        for (m in 1..4) {
            canvas.drawCircle(cx, cy, m * perMeter, line)
            canvas.drawText("${m}m", cx, cy - m * perMeter - 4 * density, small)
        }
        val s = 12 * density
        head.reset(); head.moveTo(cx, cy - s); head.lineTo(cx + s * 0.75f, cy + s * 0.7f); head.lineTo(cx - s * 0.75f, cy + s * 0.7f); head.close()
        canvas.drawPath(head, me)

        val az = Binaural.azimuth(l, position)
        val horizontal = Binaural.horizontalDistance(l, position)
        val height3d = position.y - l.position.y
        val elevation = atan2(height3d, horizontal)
        val distance = sqrt(horizontal * horizontal + height3d * height3d)
        // 上にあるほど大きく描く
        val dotRadius = (11 + 7 * sin(elevation)) * density
        canvas.drawCircle(cx + horizontal * sin(az) * perMeter, cy - horizontal * cos(az) * perMeter, dotRadius, source)

        val barX = width - 22 * density
        val barH = radius
        canvas.drawLine(barX, cy - barH, barX, cy + barH, line)
        canvas.drawLine(barX - 8 * density, cy, barX + 8 * density, cy, line)
        canvas.drawCircle(barX, cy - sin(elevation) * barH, 8 * density, source)

        val deg = Math.toDegrees(az.toDouble()).roundToInt()
        val side = if (deg >= 0) "右" else "左"
        val up = Math.toDegrees(elevation.toDouble()).roundToInt()
        canvas.drawText("$side ${abs(deg)}°   ${if (up >= 0) "上" else "下"} ${abs(up)}°   ${"%.1f".format(distance)}m", cx, cy + radius + 56 * density, text)
        canvas.drawText("スマホを回すと、聞こえる方向も変わります", cx, cy + radius + 88 * density, small)
        if (isShown) postInvalidateOnAnimation()
    }
}
