package jp.devfest.shelfsight.assist

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.os.SystemClock
import android.view.View
import jp.devfest.shelfsight.ar.Unprojector
import jp.devfest.shelfsight.model.NormalizedBox

/** 画面に出す内部状態。各スレッドが書き、[DebugOverlay] が読むだけ。 */
class AssistDebug {
    class PersonView(val box: NormalizedBox, val ownHand: Boolean, val label: String)

    // カメラ・向き
    @Volatile var tracking = "準備中"
    @Volatile var headingDeg: Float? = null
    /** センサー画像の正規化座標 (0,0), (1,0), (0,1) が画面のどこに来るか（画面の正規化座標）。 */
    @Volatile var corners: FloatArray? = null
    @Volatile var rotation = 90

    // ① 写っているもの
    @Volatile var scene = Scene.NONE
    @Volatile var productBoxes: List<FloatArray> = emptyList()
    @Volatile var sceneMs = ""
    @Volatile var sceneAt = 0L

    // 地図（壁の上に貯めた商品）
    @Volatile var mapView: List<RecognizedProduct> = emptyList()
    @Volatile var mapInfo = ""

    // ② 棚の説明
    @Volatile var kept: List<RecognizedProduct> = emptyList()
    @Volatile var removed: List<RecognizedProduct> = emptyList()
    @Volatile var unnamed = 0
    @Volatile var mainGroup: String? = null
    @Volatile var description = ""
    @Volatile var describeInfo = ""
    @Volatile var productsAt = 0L

    // ③ 長押し
    @Volatile var longPress = ""
    @Volatile var focusBox: FloatArray? = null
    @Volatile var focusAt = 0L
    @Volatile var lookup = ""

    // ④ 人
    @Volatile var people: List<PersonView> = emptyList()
    @Volatile var peopleInfo = ""
    @Volatile var peopleAt = 0L

    // 声と設定
    @Volatile var voice = ""
    @Volatile var settings = ""
}

/**
 * カメラの映像の上に、認識の状況をすべて描く。
 * 映像の上: 商品の枠（緑＝説明に使った、橙＝場違いとして外した、灰＝最後の解析の枠、青＝地図に貯めた今見えている商品と見えた回数）、
 * 長押しで調べた範囲（水色）、人の枠（赤＝人、紫の点線＝自分の手として外した）。下: 文字の一覧。
 */
class DebugOverlay(context: Context, private val d: AssistDebug) : View(context) {
    private val density = resources.displayMetrics.density
    private val box = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2 * density }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 11 * density }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 9 * density }
    private val dashed = DashPathEffect(floatArrayOf(8 * density, 6 * density), 0f)

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
        setWillNotDraw(false)
    }

    override fun onDraw(canvas: Canvas) {
        val now = SystemClock.elapsedRealtime()
        drawImageLayer(canvas, now)
        drawPanel(canvas, 0f)
        postInvalidateDelayed(REFRESH_MS)
    }

    /** 正立画像の正規化座標 → 画面のピクセル。 */
    private fun toView(u: Float, v: Float): Pair<Float, Float>? {
        val c = d.corners ?: return null
        val (sx, sy) = Unprojector.uprightToSensor(u, v, d.rotation)
        val x = c[0] + (c[2] - c[0]) * sx + (c[4] - c[0]) * sy
        val y = c[1] + (c[3] - c[1]) * sx + (c[5] - c[1]) * sy
        return x * width to y * height
    }

    private fun rect(canvas: Canvas, l: Float, t: Float, r: Float, b: Float, color: Int, label: String? = null, dash: Boolean = false) {
        val pts = listOf(toView(l, t), toView(r, t), toView(r, b), toView(l, b))
        if (pts.any { it == null }) return
        val xs = pts.map { it!!.first }
        val ys = pts.map { it!!.second }
        box.color = color
        box.pathEffect = if (dash) dashed else null
        canvas.drawRect(xs.min(), ys.min(), xs.max(), ys.max(), box)
        if (label != null) {
            fill.color = (color and 0x00FFFFFF) or 0xB0000000.toInt()
            val w = small.measureText(label)
            canvas.drawRect(xs.min(), ys.min() - small.textSize - 2, xs.min() + w + 4, ys.min(), fill)
            canvas.drawText(label, xs.min() + 2, ys.min() - 3, small)
        }
    }

    private fun drawImageLayer(canvas: Canvas, now: Long) {
        // 商品の枠（最後の解析）
        if (now - d.sceneAt < SCENE_MS) for (b in d.productBoxes) rect(canvas, b[0], b[1], b[2], b[3], GRAY)
        // 地図に貯めた、今見えている商品（青、数字は見えた回数）
        if (now - d.sceneAt < SCENE_MS) for (p in d.mapView) rect(canvas, p.left, p.top, p.right, p.bottom, BLUE, "${short(p.name)} ${p.sim.toInt()}")
        // 説明に使った商品と、外した商品
        if (now - d.productsAt < PRODUCTS_MS) {
            for (p in d.kept) rect(canvas, p.left, p.top, p.right, p.bottom, GREEN, short(p.name))
            for (p in d.removed) rect(canvas, p.left, p.top, p.right, p.bottom, ORANGE, "×" + short(p.name))
        }
        // 長押しで調べた範囲
        if (now - d.focusAt < PRODUCTS_MS) d.focusBox?.let { b ->
            box.strokeWidth = 4 * density
            rect(canvas, b[0], b[1], b[2], b[3], CYAN, "調べた範囲")
            box.strokeWidth = 2 * density
        }
        // 人
        if (now - d.peopleAt < STALE_MS) for (p in d.people) {
            rect(canvas, p.box.left, p.box.top, p.box.right, p.box.bottom, if (p.ownHand) PURPLE else RED, p.label, dash = p.ownHand)
        }
    }

    private fun drawPanel(canvas: Canvas, top: Float) {
        val now = SystemClock.elapsedRealtime()
        val age = { t: Long -> if (t == 0L) "-" else "${(now - t) / 1000}秒前" }
        val lines = buildList {
            add("追跡: ${d.tracking}  向き: ${d.headingDeg?.let { "%.0f°".format(it) } ?: "-"}  ${d.settings}")
            add("場面: ${d.scene}  商品の枠: ${d.productBoxes.size}  ${d.sceneMs}（${age(d.sceneAt)}）")
            add("地図: ${d.mapInfo.ifEmpty { "-" }}")
            add("② 主な分類: ${d.mainGroup ?: "-"}  使った ${d.kept.size} / 外した ${d.removed.size} / 名前なし ${d.unnamed}  ${d.describeInfo}（${age(d.productsAt)}）")
            if (d.removed.isNotEmpty()) add("   外した: " + d.removed.groupBy { it.genre ?: "?" }.entries.joinToString("、") { "${it.key}×${it.value.size}" })
            if (d.description.isNotEmpty()) add("   説明: ${d.description}")
            add("③ 長押し: ${d.longPress.ifEmpty { "-" }}（${age(d.focusAt)}）")
            if (d.lookup.isNotEmpty()) add("   商品: ${d.lookup}")
            add("④ 人: ${d.peopleInfo.ifEmpty { "-" }}")
            add("声: ${d.voice}")
        }
        val lh = text.textSize * 1.25f
        val wrapped = lines.flatMap { wrap(it, width - 16f) }
        val panelTop = height - lh * wrapped.size - 12
        fill.color = 0xB0000000.toInt()
        canvas.drawRect(0f, maxOf(top, panelTop), width.toFloat(), height.toFloat(), fill)
        wrapped.forEachIndexed { i, l -> canvas.drawText(l, 8f, panelTop + lh * (i + 1), text) }
    }

    private fun wrap(line: String, max: Float): List<String> {
        val out = mutableListOf<String>()
        var rest = line
        while (rest.isNotEmpty()) {
            val n = text.breakText(rest, true, max, null).coerceAtLeast(1)
            out += rest.take(n)
            rest = rest.drop(n)
        }
        return out
    }

    private fun short(name: String) = name.substringAfter(' ').take(10)
    private fun pct(p: Float) = "${(p * 100).toInt()}%"

    private companion object {
        const val REFRESH_MS = 100L
        const val STALE_MS = 3000L
        /** 全枠を MobileCLIP で認識すると、枠が多い棚では 1 周 20 秒ほどかかるので、次の結果が出るまで枠を残す。 */
        const val SCENE_MS = 40000L
        const val PRODUCTS_MS = 8000L
        const val GRAY = 0xFF9E9E9E.toInt()
        const val BLUE = 0xFF448AFF.toInt()
        const val GREEN = 0xFF4CAF50.toInt()
        const val ORANGE = 0xFFFF9800.toInt()
        const val CYAN = 0xFF00E5FF.toInt()
        const val RED = 0xFFF44336.toInt()
        const val PURPLE = 0xFFE040FB.toInt()
    }
}
