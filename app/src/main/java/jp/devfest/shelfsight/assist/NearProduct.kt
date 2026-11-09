package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.Unprojector
import kotlin.math.abs

/**
 * スマホのすぐ前（画面の中央）に商品があるかを、ARCore の深度だけで決める（iOS 版の NearProduct と同じ）。
 * 商品の枠は使わない。枠の大きさで近さを決めることもしない（近づくと枠が出なかったり、ばらばらになったりするため）。
 * ARCore の深度は、スマホを少し動かしたときの画像の違いから推定する（LiDAR がなくてもよい）。
 *
 * 1. 画面の中央（[CENTER_FRACTION] 四方）の深度の中央値を、スマホから商品までの距離とする。
 *    中央の点の [MIN_VALID_FRACTION] 以上で深度が取れ、距離が [MAX_DISTANCE_M] 以内なら、近い商品がある
 * 2. 調べる範囲（Cloud Vision に送る切り抜き）は、中央から、近い商品と同じくらいの距離（差が [SAME_TOL_M] 以内）の点を
 *    つないだ範囲。ただし棚の面は全部同じくらいの距離なので、実際の大きさで [MAX_SIZE_M] 四方までにする
 */
object NearProduct {
    /** 選んだ商品。[box] は調べる範囲（正立画像の 0..1、[left, top, right, bottom]）。 */
    class Pick(val box: FloatArray, val distanceM: Float) {
        fun describe() = "近い商品 ${"%.2f".format(distanceM)}m"
    }

    /** 1 回の判定の結果。近い商品がなくても、中央の距離と深度の取れた割合は分かる（画面の表示とログ用）。 */
    class Reading(val pick: Pick?, val centerM: Float?, val validFraction: Float) {
        fun describe(): String = pick?.describe()
            ?: "なし（中央 ${centerM?.let { "%.2fm".format(it) } ?: "-"}、深度の取れた点 ${(validFraction * 100).toInt()}%）"
    }

    const val MAX_DISTANCE_M = 0.5f
    const val SAME_TOL_M = 0.10f
    const val MAX_SIZE_M = 0.30f
    const val CENTER_FRACTION = 0.2f
    const val MIN_VALID_FRACTION = 0.5f
    /** 画面全体を GRID × GRID の点で調べる。 */
    const val GRID = 32

    fun measure(g: CaptureGeometry): Reading {
        if (g.depth == null) return Reading(null, null, 0f)
        val n = GRID
        // 正立画像の上の点ごとの深度（取れなければ null）
        val depth = arrayOfNulls<Float>(n * n)
        for (row in 0 until n) for (col in 0 until n) {
            depth[row * n + col] = Unprojector.depthAt((col + 0.5f) / n, (row + 0.5f) / n, g)
        }
        // 1. 画面の中央の距離
        val lo = ((0.5f - CENTER_FRACTION / 2) * n).toInt()
        val hi = ((0.5f + CENTER_FRACTION / 2) * n).toInt()
        val center = ArrayList<Float>()
        for (row in lo until hi) for (col in lo until hi) depth[row * n + col]?.let { center += it }
        val valid = center.size.toFloat() / ((hi - lo) * (hi - lo))
        if (center.isEmpty() || valid < MIN_VALID_FRACTION) return Reading(null, null, valid)
        center.sort()
        val d = center[center.size / 2]
        if (d > MAX_DISTANCE_M) return Reading(null, d, valid)

        // 2. 中央から、同じくらいの距離の点をつなぐ（上下左右の隣へ）
        val seen = BooleanArray(n * n)
        val stack = ArrayDeque<Int>()
        for (row in lo until hi) for (col in lo until hi) {
            val i = row * n + col
            val di = depth[i]
            if (di != null && abs(di - d) <= SAME_TOL_M) { seen[i] = true; stack.addLast(i) }
        }
        var minCol = n; var minRow = n; var maxCol = -1; var maxRow = -1
        while (stack.isNotEmpty()) {
            val i = stack.removeLast()
            val row = i / n
            val col = i % n
            minCol = minOf(minCol, col); maxCol = maxOf(maxCol, col)
            minRow = minOf(minRow, row); maxRow = maxOf(maxRow, row)
            for ((r, c) in listOf(row - 1 to col, row + 1 to col, row to col - 1, row to col + 1)) {
                if (r !in 0 until n || c !in 0 until n) continue
                val j = r * n + c
                val dj = depth[j]
                if (!seen[j] && dj != null && abs(dj - d) <= SAME_TOL_M) { seen[j] = true; stack.addLast(j) }
            }
        }
        // 実際の大きさで MAX_SIZE_M 四方まで（中央を真ん中にして切る）
        val (hu, hv) = halfSize(MAX_SIZE_M / 2, d, g)
        val box = floatArrayOf(
            maxOf(minCol.toFloat() / n, 0.5f - hu), maxOf(minRow.toFloat() / n, 0.5f - hv),
            minOf((maxCol + 1).toFloat() / n, 0.5f + hu), minOf((maxRow + 1).toFloat() / n, 0.5f + hv),
        )
        return Reading(Pick(box, d), d, valid)
    }

    /** カメラから [distanceM] 離れたところの [halfM] が、正立画像の正規化座標でどれだけの幅・高さになるか。 */
    fun halfSize(halfM: Float, distanceM: Float, g: CaptureGeometry): Pair<Float, Float> {
        val rotated = g.uprightRotationDegrees == 90 || g.uprightRotationDegrees == 270
        // 正立画像の横・縦の焦点距離と大きさ（センサー画像を 90 度回すと横と縦が入れ替わる）
        val fu = if (rotated) g.focalY else g.focalX
        val fv = if (rotated) g.focalX else g.focalY
        val wu = (if (rotated) g.imageHeight else g.imageWidth).toFloat()
        val wv = (if (rotated) g.imageWidth else g.imageHeight).toFloat()
        return halfM / distanceM * fu / wu to halfM / distanceM * fv / wv
    }
}
