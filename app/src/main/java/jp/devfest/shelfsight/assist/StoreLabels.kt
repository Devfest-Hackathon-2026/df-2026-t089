package jp.devfest.shelfsight.assist

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.view.View

/** 棚の上に出すジャンル。[u], [v] は店の座標、[top] は棚の高さ。 */
internal class ShelfLabel(val u: Double, val v: Double, val top: Double, val text: String)

/** 画面の座標に直した文字。[shelf] なら棚のジャンル、違えば商品名。先頭のものほど優先して描く。 */
internal class ScreenLabel(val x: Float, val y: Float, val text: String, val shelf: Boolean)

internal object StoreLabels {
    /** 面ごとに、同じジャンルが続くモジュールを 1 つにまとめる。テーマが決まらないモジュールは一番多いカテゴリの名前にする。 */
    fun shelves(view: StoreView): List<ShelfLabel> = view.fixtures.flatMap { f ->
        f.faces.flatMap { (side, modules) ->
            // 両面の棚は、それぞれの面の側に寄せて出す
            val across = f.c + if (f.faces.size == 2) (if (side == "+") f.d / 4 else -f.d / 4) else 0.0
            val names = modules.indices.map { i -> f.themes[side]?.getOrNull(i) ?: modules[i].firstOrNull()?.first }
            var first = 0
            val out = ArrayList<ShelfLabel>()
            for (i in 1..names.size) {
                if (i < names.size && names[i] == names[first]) continue
                names[first]?.let { name ->
                    val along = f.a + (first + i) / 2.0 * f.w
                    out += if (f.axis == 0) ShelfLabel(across, along, f.h, name) else ShelfLabel(along, across, f.h, name)
                }
                first = i
            }
            out
        }
    }
}

/** 3D モデルの上に重ねる文字。GL スレッドが [labels] を入れ、UI スレッドが描く。 */
internal class StoreLabelsView(context: Context) : View(context) {
    @Volatile var labels: List<ScreenLabel> = emptyList()
    /** 上のタブの下端。棚のジャンルがタブに隠れないようにする。 */
    var topInset = 0
    private val density = resources.displayMetrics.scaledDensity
    private val shelfText = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 14 * density; typeface = Typeface.DEFAULT_BOLD }
    private val itemText = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 11 * density }
    private val shelfBack = Paint().apply { color = 0xCC1B6FD0.toInt() }
    private val itemBack = Paint().apply { color = 0xB0000000.toInt() }

    init { importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO }

    override fun onDraw(canvas: Canvas) {
        val taken = ArrayList<RectF>()
        val pad = 3 * density
        for (l in labels) {
            val paint = if (l.shelf) shelfText else itemText
            val half = paint.measureText(l.text) / 2 + pad
            val height = paint.textSize + 2 * pad
            val box = RectF(l.x - half, l.y - height, l.x + half, l.y)   // 点や棚のすぐ上に出す
            if (l.shelf) box.offset(
                if (box.left < 0) -box.left else minOf(0f, width - box.right),
                // 下の 3 分の 1 は店の説明の文章で隠れる
                if (box.top < topInset) topInset - box.top else minOf(0f, this.height * 2f / 3 - box.bottom),
            )
            if (taken.any { RectF.intersects(it, box) }) continue   // 重なる文字は読めないので描かない
            taken += box
            canvas.drawRoundRect(box, pad, pad, if (l.shelf) shelfBack else itemBack)
            canvas.drawText(l.text, box.left + pad, box.bottom - pad - paint.descent() / 2, paint)
        }
    }
}
