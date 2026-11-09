package jp.devfest.shelfsight.assist

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Color
import android.view.View
import android.widget.CheckBox
import android.widget.LinearLayout

/** 店のモデルの画面で表示を切り替えられる項目。[color] は描画の色（RGB）で、凡例にも使う。 */
internal enum class StoreLayer(val label: String, val color: Int, val topViewOnly: Boolean = false) {
    SHELF("棚", 0xc8ccd2),
    GLASS("ガラス棚", 0x4fd1c5),
    FLOOR("床", 0x565d68),
    POINT("商品の点", 0xff9800),
    SIGHT("視線", 0x63e6ff),
    CATEGORY("商品カテゴリ", 0xffffff),
    GENRE("棚のジャンル", 0x1b6fd0),
    PERSON("人", 0x5ad1e6),
    PHONE("スマホ", 0xf0efec),
    MIRROR("ガラス補正", 0xff8c00),
    RANSAC("RANSAC", 0xff4fd8, topViewOnly = true),
    SAVED("保存マップ", 0xb388ff),
}

/** 項目ごとのチェックを縦に並べた凡例。チェックの色が描画の色になる。 */
internal class StoreLegend(context: Context, private val onChange: (Set<StoreLayer>) -> Unit) : LinearLayout(context) {
    private var shown = StoreLayer.values().toSet()
    private val topViewRows = ArrayList<View>()

    init {
        orientation = VERTICAL
        setBackgroundColor(0xA0000000.toInt())
        for (layer in StoreLayer.values()) {
            val row = CheckBox(context).apply {
                text = layer.label
                textSize = 12f
                setTextColor(Color.WHITE)
                buttonTintList = ColorStateList.valueOf(0xFF000000.toInt() or layer.color)
                isChecked = true
                minHeight = 0
                minimumHeight = 0
                setOnCheckedChangeListener { _, checked ->
                    shown = if (checked) shown + layer else shown - layer
                    onChange(shown)
                }
            }
            if (layer.topViewOnly) {
                row.visibility = GONE
                topViewRows += row
            }
            addView(row)
        }
    }

    /** RANSAC のように 2D でだけ描く項目の行を出し入れする。 */
    fun showTopViewRows(show: Boolean) {
        for (row in topViewRows) row.visibility = if (show) VISIBLE else GONE
    }
}
