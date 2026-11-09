package jp.devfest.shelfsight.assist

import android.opengl.GLES20
import android.opengl.Matrix
import jp.devfest.shelfsight.ar.GlUtil
import jp.devfest.shelfsight.mapping.OrbitCamera
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.perception.PersonMarker
import jp.devfest.shelfsight.perception.PersonMotion

/** 推定した店（[StoreView]）を線画で、認識した商品を点で、自分を向き付きのスマホで描く。GL スレッドから使う。 */
internal class StoreModelRenderer {
    private val program = GlUtil.createProgram(VERTEX, FRAGMENT)
    private val position = GLES20.glGetAttribLocation(program, "a_Position")
    private val mvp = GLES20.glGetUniformLocation(program, "u_Mvp")
    private val color = GLES20.glGetUniformLocation(program, "u_Color")
    private val boxLines = GlUtil.floatBuffer(BOX_LINES)
    private val arrowLines = GlUtil.floatBuffer(ARROW_LINES)
    private val pointProgram = GlUtil.createProgram(POINT_VERTEX, POINT_FRAGMENT)
    private val pointPosition = GLES20.glGetAttribLocation(pointProgram, "a_Position")
    private val pointMvp = GLES20.glGetUniformLocation(pointProgram, "u_Mvp")
    private val pointColor = GLES20.glGetUniformLocation(pointProgram, "u_Color")
    private val pointSize = GLES20.glGetUniformLocation(pointProgram, "u_Size")
    private val circleLines = GlUtil.floatBuffer(CIRCLE_LINES)
    private val uprightLine = GlUtil.floatBuffer(UPRIGHT_LINE)
    private val pointBuffer = GlUtil.floatBuffer(FloatArray(ShelfMap.MAX_ITEMS * 3))
    private val ransacBuffer = GlUtil.floatBuffer(FloatArray(MultiLineRansac.MAX_LINES * 6))
    private val restoreLineBuffer = GlUtil.floatBuffer(FloatArray(ShelfMap.MAX_ITEMS * 6))
    private val productArrowBuffer = GlUtil.floatBuffer(FloatArray(ShelfMap.MAX_ITEMS * ProductViewArrow.FLOATS))
    private val viewProjection = FloatArray(16)
    private val model = FloatArray(16)
    private val result = FloatArray(16)

    /**
     * @param phone 端末の姿勢（画面の右 +x、上 +y、カメラの向き -z の 4×4 行列）。[floorY] はそこから見積もった床の高さ。
     * @return 画面に重ねる文字（棚のジャンルが先、商品名は見えた回数の多い順）
     */
    fun draw(view: StoreView?, orbit: OrbitCamera, width: Int, height: Int, phone: FloatArray?, floorY: Float, items: List<MapItem>, people: List<PersonMarker>, topView: Boolean = false,
        layers: Set<StoreLayer> = StoreLayer.values().toSet(), saved: StoreView? = null, pending: List<Vec3> = emptyList()): List<ScreenLabel> {
        val e = begin(orbit, width, height, 50f, topView)
        useLines(boxLines)
        if (view?.room != null) {
            val r = view.room
            if (StoreLayer.FLOOR in layers) box(view.yaw, r[0], r[1], r[2], r[3], floorY, 0f, FLOOR)
            for (f in view.fixtures) {
                if ((if (f.kind == "glass") StoreLayer.GLASS else StoreLayer.SHELF) !in layers) continue
                val lo = f.c - f.d / 2; val hi = f.c + f.d / 2
                for (i in 0 until f.n) {
                    val a = f.moduleStart(i); val b = f.moduleEnd(i)
                    val col = if (f.kind == "glass") GLASS else SHELF
                    if (f.axis == 0) box(view.yaw, lo, a, hi, b, floorY, f.h.toFloat(), col)
                    else box(view.yaw, a, lo, b, hi, floorY, f.h.toFloat(), col)
                }
            }
        }
        // shortcut: 保存した地図は前回の ARCore の座標のままで今の場所と合っていない。店の点との位置合わせを入れたら案内にも使う
        if (saved != null && StoreLayer.SAVED in layers) for (f in saved.fixtures) {
            val lo = f.c - f.d / 2; val hi = f.c + f.d / 2
            for (i in 0 until f.n) {
                val a = f.moduleStart(i); val b = f.moduleEnd(i)
                if (f.axis == 0) box(saved.yaw, lo, a, hi, b, floorY, f.h.toFloat(), SAVED) else box(saved.yaw, a, lo, b, hi, floorY, f.h.toFloat(), SAVED)
            }
        }
        if (phone != null && StoreLayer.PHONE in layers) drawPhone(phone)
        if (StoreLayer.PERSON in layers) for (p in people) drawPerson(p.position.x, floorY, p.position.z, if (p.motion == PersonMotion.APPROACHING) APPROACHING else OTHER_PERSON)
        if (StoreLayer.SIGHT in layers) drawProductViewArrows(items)
        val correcting = StoreLayer.MIRROR in layers
        val restored = items.map { if (correcting) GlassProjection.corrected(it, view) else null }
        drawRestoreLines(items, restored)
        if (topView && correcting) drawMirrorLines(view, floorY)
        if (topView && StoreLayer.RANSAC in layers) drawRansac(view, floorY)
        GLES20.glDisableVertexAttribArray(position)
        if (StoreLayer.POINT in layers) drawPending(pending)
        drawPoints(items, restored, StoreLayer.POINT in layers)
        if (topView && correcting) drawMirrorPoints(view, floorY)
        return (if (StoreLayer.PERSON in layers) personLabels(people, phone, floorY, width, height) else emptyList()) +
            labels(view, items, e, floorY, width, height, StoreLayer.GENRE in layers, StoreLayer.CATEGORY in layers)
    }

    private fun personLabels(people: List<PersonMarker>, phone: FloatArray?, floorY: Float, width: Int, height: Int): List<ScreenLabel> = people.mapNotNull { p ->
        screen(p.position.x, floorY + PERSON_LABEL_Y, p.position.z, width, height, SHELF_LABEL_REACH)?.let { at ->
            val metres = phone?.let { "%.1fm".format(Math.hypot((p.position.x - it[12]).toDouble(), (p.position.z - it[14]).toDouble())) } ?: "?"
            ScreenLabel(at[0], at[1], "人 $metres ${p.motion.labelJa}", true)
        }
    }

    /** 立体音響のテスト: 正面（-z）を向いて立つ人の線画と、その人から見た音源の位置を描く。 */
    fun drawDemo(orbit: OrbitCamera, width: Int, height: Int, listener: Listener, source: Vec3, sounding: Boolean) {
        begin(orbit, width, height, DEMO_FOV_DEG)
        useLines(circleLines)
        for (r in 1..4) {
            Matrix.setIdentityM(model, 0)
            Matrix.scaleM(model, 0, r.toFloat(), 1f, r.toFloat())
            drawLines(FLOOR, CIRCLE_LINES)
        }
        drawPerson(0f, 0f, 0f, ME)
        val az = Binaural.azimuth(listener, source)
        val horizontal = Binaural.horizontalDistance(listener, source)
        val x = horizontal * Math.sin(az.toDouble()).toFloat()
        val y = HEAD_Y + source.y - listener.position.y
        val z = -horizontal * Math.cos(az.toDouble()).toFloat()
        val col = if (sounding) POINT else MOVING
        useLines(uprightLine)
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, x, 0f, z)
        Matrix.scaleM(model, 0, 1f, y, 1f)
        drawLines(col, UPRIGHT_LINE)   // 床からの線で高さを見せる
        GLES20.glDisableVertexAttribArray(position)
        pointBuffer.clear()
        pointBuffer.put(x).put(y).put(z)
        pointBuffer.position(0)
        drawPointBuffer(1, col, SOURCE_SIZE)
    }

    /** 足元が ([x], [floorY], [z]) の人の線画。呼んだあとは箱の線が選ばれている。 */
    private fun drawPerson(x: Float, floorY: Float, z: Float, col: FloatArray) {
        useLines(circleLines)
        // 頭は 3 方向の輪で球に見せる
        for ((angle, ax, az) in listOf(Triple(0f, 1f, 0f), Triple(90f, 1f, 0f), Triple(90f, 0f, 1f))) {
            Matrix.setIdentityM(model, 0)
            Matrix.translateM(model, 0, x, floorY + HEAD_Y, z)
            if (angle != 0f) Matrix.rotateM(model, 0, angle, ax, 0f, az)
            Matrix.scaleM(model, 0, HEAD_R, HEAD_R, HEAD_R)
            drawLines(col, CIRCLE_LINES)
        }
        useLines(boxLines)
        for (p in BODY) {
            Matrix.setIdentityM(model, 0)
            Matrix.translateM(model, 0, x + p[0], floorY + p[1], z + p[2])
            Matrix.scaleM(model, 0, p[3], p[4], p[5])
            drawLines(col, BOX_LINES)
        }
    }

    /** 画面を消して線の描画を始める。@return 視点の位置 */
    private fun begin(orbit: OrbitCamera, width: Int, height: Int, fovDeg: Float, topView: Boolean = false): Vec3 {
        GLES20.glClearColor(0.08f, 0.09f, 0.11f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)   // 線画なので隠れる面がない
        val projection = FloatArray(16)
        val lookAt = FloatArray(16)
        val aspect = width.toFloat() / height.coerceAtLeast(1)
        var e = orbit.eye()
        val c = orbit.center()
        if (topView) {   // 真上からの正射影。ズームは視点の距離、回転は視点の水平の向きをそのまま使う
            val distance = e.distanceTo(c)
            val half = distance * Math.tan(Math.toRadians(fovDeg / 2.0)).toFloat()
            val flat = Math.hypot((c.x - e.x).toDouble(), (c.z - e.z).toDouble()).toFloat().coerceAtLeast(1e-4f)
            Matrix.orthoM(projection, 0, -half * aspect, half * aspect, -half, half, -200f, 200f)
            Matrix.setLookAtM(lookAt, 0, c.x, c.y + distance, c.z, c.x, c.y, c.z, (c.x - e.x) / flat, 0f, (c.z - e.z) / flat)
            e = Vec3(c.x, c.y + distance, c.z)
        } else {
            Matrix.perspectiveM(projection, 0, fovDeg, aspect, 0.1f, 200f)
            Matrix.setLookAtM(lookAt, 0, e.x, e.y, e.z, c.x, c.y, c.z, 0f, 1f, 0f)
        }
        Matrix.multiplyMM(viewProjection, 0, projection, 0, lookAt, 0)
        GLES20.glUseProgram(program)
        GLES20.glLineWidth(LINE_WIDTH)
        GLES20.glEnableVertexAttribArray(position)
        return e
    }

    private fun useLines(lines: java.nio.FloatBuffer) {
        lines.position(0)
        GLES20.glVertexAttribPointer(position, 3, GLES20.GL_FLOAT, false, 0, lines)
    }

    /** まだ分類できていない点を暗く描く。分類できると [drawPoints] の明るい色に変わる。 */
    private fun drawPending(points: List<Vec3>) {
        val shown = points.take(ShelfMap.MAX_ITEMS)
        if (shown.isEmpty()) return
        pointBuffer.clear()
        for (p in shown) pointBuffer.put(p.x).put(p.y).put(p.z)
        pointBuffer.position(0)
        drawPointBuffer(shown.size, PENDING_POINT, POINT_SIZE)
    }

    /** ガラス越しの元の点は薄いオレンジ、ガラス面へ戻した点はふつうの色で描く。 */
    private fun drawPoints(items: List<MapItem>, restored: List<Vec3?>, plain: Boolean) {
        val n = minOf(items.size, ShelfMap.MAX_ITEMS)
        fun draw(col: FloatArray, at: (Int) -> Vec3?) {
            pointBuffer.clear()
            var count = 0
            for (i in 0 until n) at(i)?.let { pointBuffer.put(it.x).put(it.y).put(it.z); count++ }
            pointBuffer.position(0)
            if (count > 0) drawPointBuffer(count, col, POINT_SIZE)
        }
        if (plain) draw(POINT) { i -> items[i].position.takeIf { restored[i] == null } }
        draw(RAW_POINT) { i -> items[i].position.takeIf { restored[i] != null } }
        draw(POINT) { i -> restored[i] }
    }

    /** 2D 表示に、ガラスと判断して動かした点の元の位置と新しい位置を結ぶ細い線を描く。 */
    private fun drawMirrorLines(view: StoreView?, floorY: Float) {
        val moved = view?.mirror.orEmpty().take(ShelfMap.MAX_ITEMS)
        if (moved.isEmpty()) return
        restoreLineBuffer.clear()
        for (m in moved) restoreLineBuffer.put(m[0].toFloat()).put(floorY).put(m[1].toFloat()).put(m[2].toFloat()).put(floorY).put(m[3].toFloat())
        restoreLineBuffer.position(0)
        useLines(restoreLineBuffer)
        Matrix.setIdentityM(model, 0)
        Matrix.multiplyMM(result, 0, viewProjection, 0, model, 0)
        GLES20.glUniformMatrix4fv(mvp, 1, false, result, 0)
        GLES20.glUniform3fv(color, 1, RESTORE_LINE, 0)
        GLES20.glLineWidth(THIN_LINE_WIDTH)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, moved.size * 2)
        GLES20.glLineWidth(LINE_WIDTH)
        useLines(boxLines)
    }

    private fun drawMirrorPoints(view: StoreView?, floorY: Float) {
        val moved = view?.mirror.orEmpty().take(ShelfMap.MAX_ITEMS)
        if (moved.isEmpty()) return
        fun draw(col: FloatArray, x: Int, z: Int) {
            pointBuffer.clear()
            for (m in moved) pointBuffer.put(m[x].toFloat()).put(floorY).put(m[z].toFloat())
            pointBuffer.position(0)
            drawPointBuffer(moved.size, col, POINT_SIZE)
        }
        draw(RAW_POINT, 0, 1)
        draw(MIRROR_POINT, 2, 3)
    }

    /** 2D 表示に重ねる、RANSAC の線分。 */
    private fun drawRansac(view: StoreView?, floorY: Float) {
        val lines = view?.ransac.orEmpty()
        if (lines.isEmpty()) return
        ransacBuffer.clear()
        for (l in lines) ransacBuffer.put(l[0].toFloat()).put(floorY).put(l[1].toFloat()).put(l[2].toFloat()).put(floorY).put(l[3].toFloat())
        ransacBuffer.position(0)
        useLines(ransacBuffer)
        Matrix.setIdentityM(model, 0)
        Matrix.multiplyMM(result, 0, viewProjection, 0, model, 0)
        GLES20.glUniformMatrix4fv(mvp, 1, false, result, 0)
        GLES20.glUniform3fv(color, 1, RANSAC, 0)
        GLES20.glLineWidth(RANSAC_LINE_WIDTH)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, lines.size * 2)
        GLES20.glLineWidth(LINE_WIDTH)
        useLines(boxLines)
    }

    /** 元の点と、ガラス面へ戻した点を細い線でつなぐ。 */
    private fun drawRestoreLines(items: List<MapItem>, restored: List<Vec3?>) {
        restoreLineBuffer.clear()
        var lines = 0
        for (i in 0 until minOf(items.size, ShelfMap.MAX_ITEMS)) restored[i]?.let { r ->
            val p = items[i].position
            restoreLineBuffer.put(p.x).put(p.y).put(p.z).put(r.x).put(r.y).put(r.z)
            lines++
        }
        if (lines == 0) return
        restoreLineBuffer.position(0)
        useLines(restoreLineBuffer)
        Matrix.setIdentityM(model, 0)
        Matrix.multiplyMM(result, 0, viewProjection, 0, model, 0)
        GLES20.glUniformMatrix4fv(mvp, 1, false, result, 0)
        GLES20.glUniform3fv(color, 1, RESTORE_LINE, 0)
        GLES20.glLineWidth(THIN_LINE_WIDTH)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, lines * 2)
        GLES20.glLineWidth(LINE_WIDTH)
        useLines(boxLines)
    }

    /** 商品を見た場所から点まで、細い線で視線を描く。 */
    private fun drawProductViewArrows(items: List<MapItem>) {
        productArrowBuffer.clear()
        var n = 0
        for (item in items) {
            val camera = item.cameraAt ?: continue
            if (n == ShelfMap.MAX_ITEMS) break
            ProductViewArrow.append(item.position, camera, productArrowBuffer)
            n++
        }
        if (n == 0) return
        productArrowBuffer.position(0)
        useLines(productArrowBuffer)
        Matrix.setIdentityM(model, 0)
        Matrix.multiplyMM(result, 0, viewProjection, 0, model, 0)
        GLES20.glUniformMatrix4fv(mvp, 1, false, result, 0)
        GLES20.glUniform3fv(color, 1, PRODUCT_VIEW, 0)
        GLES20.glLineWidth(THIN_LINE_WIDTH)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, n * ProductViewArrow.FLOATS / 3)
        GLES20.glLineWidth(LINE_WIDTH)
        useLines(boxLines)
    }

    private fun drawPointBuffer(n: Int, col: FloatArray, size: Float) {
        GLES20.glUseProgram(pointProgram)
        GLES20.glUniformMatrix4fv(pointMvp, 1, false, viewProjection, 0)
        GLES20.glUniform3fv(pointColor, 1, col, 0)
        GLES20.glUniform1f(pointSize, size)
        GLES20.glEnableVertexAttribArray(pointPosition)
        GLES20.glVertexAttribPointer(pointPosition, 3, GLES20.GL_FLOAT, false, 0, pointBuffer)
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, n)
        GLES20.glDisableVertexAttribArray(pointPosition)
    }

    private fun labels(view: StoreView?, items: List<MapItem>, eye: Vec3, floorY: Float, width: Int, height: Int, genres: Boolean, categories: Boolean): List<ScreenLabel> {
        val out = ArrayList<ScreenLabel>()
        if (genres && view?.room != null) {
            val cs = Math.cos(view.yaw); val sn = Math.sin(view.yaw)
            for (s in StoreLabels.shelves(view)) {
                val x = (s.u * cs - s.v * sn).toFloat()
                val z = (s.u * sn + s.v * cs).toFloat()
                screen(x, floorY + s.top.toFloat() + SHELF_LABEL_LIFT_M, z, width, height, SHELF_LABEL_REACH)?.let { out += ScreenLabel(it[0], it[1], s.text, true) }
            }
        }
        // 近づいた点だけ名前を出す
        if (categories) for (item in items.filter { it.position.distanceTo(eye) < NAME_RANGE_M }.sortedByDescending { it.seen }) {
            val p = item.position
            screen(p.x, p.y, p.z, width, height)?.let { out += ScreenLabel(it[0], it[1] - POINT_SIZE / 2, item.name, false) }
        }
        return out
    }

    /** 世界座標の点の画面上の位置（ピクセル）。視点の後ろや、画面の [reach] 倍の範囲より外なら null。 */
    private fun screen(x: Float, y: Float, z: Float, width: Int, height: Int, reach: Float = 1f): FloatArray? {
        val m = viewProjection
        val w = m[3] * x + m[7] * y + m[11] * z + m[15]
        if (w <= 0f) return null
        val sx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w
        val sy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w
        if (sx !in -reach..reach || sy !in -reach..reach) return null
        return floatArrayOf((sx + 1f) / 2f * width, (1f - sy) / 2f * height)
    }

    private fun box(yaw: Double, u0: Double, v0: Double, u1: Double, v1: Double, y: Float, height: Float, col: FloatArray) {
        val cu = (u0 + u1) / 2; val cv = (v0 + v1) / 2
        val cs = Math.cos(yaw); val sn = Math.sin(yaw)
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, (cu * cs - cv * sn).toFloat(), y + height / 2, (cu * sn + cv * cs).toFloat())
        Matrix.rotateM(model, 0, -Math.toDegrees(yaw).toFloat(), 0f, 1f, 0f)
        Matrix.scaleM(model, 0, (u1 - u0).toFloat(), height, (v1 - v0).toFloat())
        drawLines(col, BOX_LINES)
    }

    /** 実際の位置と傾きで浮かぶスマホと、カメラの向きの矢印。店の中で見えるよう、実物より大きく描く。 */
    private fun drawPhone(pose: FloatArray) {
        Matrix.scaleM(model, 0, pose, 0, PHONE_W, PHONE_H, PHONE_D)
        drawLines(ME, BOX_LINES)
        useLines(arrowLines)
        System.arraycopy(pose, 0, model, 0, 16)
        drawLines(ARROW, ARROW_LINES)
        useLines(boxLines)
    }

    private fun drawLines(col: FloatArray, lines: FloatArray) {
        Matrix.multiplyMM(result, 0, viewProjection, 0, model, 0)
        GLES20.glUniformMatrix4fv(mvp, 1, false, result, 0)
        GLES20.glUniform3fv(color, 1, col, 0)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, lines.size / 3)
    }

    private companion object {
        fun rgb(hex: Int) = floatArrayOf((hex shr 16 and 255) / 255f, (hex shr 8 and 255) / 255f, (hex and 255) / 255f)
        val SHELF = rgb(StoreLayer.SHELF.color)
        val FLOOR = rgb(StoreLayer.FLOOR.color)
        val ME = rgb(StoreLayer.PHONE.color)
        val ARROW = rgb(0xff7a45)
        val POINT = rgb(StoreLayer.POINT.color)
        val PENDING_POINT = rgb(0x6e4a12)
        val RAW_POINT = rgb(0xffd9a8)
        val RESTORE_LINE = rgb(StoreLayer.MIRROR.color)
        val GLASS = rgb(StoreLayer.GLASS.color)
        val RANSAC = rgb(StoreLayer.RANSAC.color)
        val MIRROR_POINT = rgb(0xff5a36)
        val SAVED = rgb(StoreLayer.SAVED.color)
        val PRODUCT_VIEW = rgb(StoreLayer.SIGHT.color)
        const val LINE_WIDTH = 2f
        const val THIN_LINE_WIDTH = 1f
        const val RANSAC_LINE_WIDTH = 5f
        const val POINT_SIZE = 12f
        const val NAME_RANGE_M = 3f
        const val SHELF_LABEL_LIFT_M = 0.1f
        /** 中心が少し画面の外に出た棚も、文字を画面の端に寄せて出す。 */
        const val SHELF_LABEL_REACH = 1.6f
        const val PHONE_W = 0.24f
        const val PHONE_H = 0.5f
        const val PHONE_D = 0.04f
        const val ARROW_LENGTH_M = 1f
        const val ARROW_HEAD_M = 0.18f
        val MOVING = rgb(0xaeb4bd)
        val APPROACHING = rgb(0xff4d4d)
        val OTHER_PERSON = rgb(StoreLayer.PERSON.color)
        const val PERSON_LABEL_Y = 1.9f
        const val SOURCE_SIZE = 44f
        /** 縦長の画面でも左右の音源が入るよう、店のモデルより広く写す。 */
        const val DEMO_FOV_DEG = 70f
        const val HEAD_Y = 1.6f
        const val HEAD_R = 0.12f

        /** 人の体の箱（中心 x, y, z、大きさ x, y, z）。胴・両腕・両脚と、正面が分かる鼻。正面は -z。 */
        val BODY = listOf(
            floatArrayOf(0f, 1.18f, 0f, 0.42f, 0.62f, 0.22f),
            floatArrayOf(-0.3f, 1.16f, 0f, 0.1f, 0.62f, 0.12f),
            floatArrayOf(0.3f, 1.16f, 0f, 0.1f, 0.62f, 0.12f),
            floatArrayOf(-0.11f, 0.43f, 0f, 0.15f, 0.86f, 0.16f),
            floatArrayOf(0.11f, 0.43f, 0f, 0.15f, 0.86f, 0.16f),
            floatArrayOf(0f, 1.6f, -0.15f, 0.04f, 0.05f, 0.08f),
        )

        /** 半径 1 で y = 0 の水平な円。 */
        val CIRCLE_LINES: FloatArray = run {
            val v = ArrayList<Float>()
            val n = 48
            for (i in 0 until n) for (k in listOf(i, i + 1)) {
                val a = 2 * Math.PI * k / n
                v += listOf(Math.cos(a).toFloat(), 0f, Math.sin(a).toFloat())
            }
            v.toFloatArray()
        }

        val UPRIGHT_LINE = floatArrayOf(0f, 0f, 0f, 0f, 1f, 0f)

        /** 中心が原点の単位立方体の 12 本の辺（x, y, z）。 */
        val BOX_LINES: FloatArray = run {
            val v = ArrayList<Float>()
            val ends = listOf(-0.5f, 0.5f)
            for (a in ends) for (b in ends) {
                v += listOf(-0.5f, a, b, 0.5f, a, b)
                v += listOf(a, -0.5f, b, a, 0.5f, b)
                v += listOf(a, b, -0.5f, a, b, 0.5f)
            }
            v.toFloatArray()
        }

        /** 端末の中心からカメラの向き（-z）へ伸びる矢印。 */
        val ARROW_LINES: FloatArray = run {
            val tip = -ARROW_LENGTH_M
            val back = tip + ARROW_HEAD_M
            val h = ARROW_HEAD_M * 0.6f
            floatArrayOf(
                0f, 0f, 0f, 0f, 0f, tip,
                0f, 0f, tip, h, 0f, back,
                0f, 0f, tip, -h, 0f, back,
                0f, 0f, tip, 0f, h, back,
                0f, 0f, tip, 0f, -h, back,
            )
        }

        const val VERTEX = """
            uniform mat4 u_Mvp;
            attribute vec3 a_Position;
            void main() {
                gl_Position = u_Mvp * vec4(a_Position, 1.0);
            }
        """
        const val FRAGMENT = """
            precision mediump float;
            uniform vec3 u_Color;
            void main() {
                gl_FragColor = vec4(u_Color, 1.0);
            }
        """
        const val POINT_VERTEX = """
            uniform mat4 u_Mvp;
            uniform float u_Size;
            attribute vec3 a_Position;
            void main() {
                gl_Position = u_Mvp * vec4(a_Position, 1.0);
                gl_PointSize = u_Size;
            }
        """
        const val POINT_FRAGMENT = """
            precision mediump float;
            uniform vec3 u_Color;
            void main() {
                vec2 d = gl_PointCoord - vec2(0.5);
                if (dot(d, d) > 0.25) discard;
                gl_FragColor = vec4(u_Color, 1.0);
            }
        """
    }
}
