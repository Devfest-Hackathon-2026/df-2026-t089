package jp.devfest.shelfsight.assist

import android.opengl.GLES20
import android.opengl.Matrix
import jp.devfest.shelfsight.ar.GlUtil
import jp.devfest.shelfsight.mapping.OrbitCamera
import jp.devfest.shelfsight.model.Vec3

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
    private val pointBuffer = GlUtil.floatBuffer(FloatArray(ShelfMap.MAX_ITEMS * 3))
    private val viewProjection = FloatArray(16)
    private val model = FloatArray(16)
    private val result = FloatArray(16)

    /**
     * @param phone 端末の姿勢（画面の右 +x、上 +y、カメラの向き -z の 4×4 行列）。[floorY] はそこから見積もった床の高さ。
     * @return 画面に重ねる文字（棚のジャンルが先、商品名は見えた回数の多い順）
     */
    fun draw(view: StoreView?, orbit: OrbitCamera, width: Int, height: Int, phone: FloatArray?, floorY: Float, items: List<MapItem>): List<ScreenLabel> {
        GLES20.glClearColor(0.08f, 0.09f, 0.11f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)   // 線画なので隠れる面がない
        val projection = FloatArray(16)
        val lookAt = FloatArray(16)
        Matrix.perspectiveM(projection, 0, 50f, width.toFloat() / height.coerceAtLeast(1), 0.1f, 200f)
        val e = orbit.eye(); val c = orbit.center()
        Matrix.setLookAtM(lookAt, 0, e.x, e.y, e.z, c.x, c.y, c.z, 0f, 1f, 0f)
        Matrix.multiplyMM(viewProjection, 0, projection, 0, lookAt, 0)
        GLES20.glUseProgram(program)
        GLES20.glLineWidth(LINE_WIDTH)
        GLES20.glEnableVertexAttribArray(position)
        useLines(boxLines)
        if (view?.room != null) {
            val r = view.room
            box(view.yaw, r[0], r[1], r[2], r[3], floorY, 0f, FLOOR)
            for (f in view.fixtures) {
                val lo = f.c - f.d / 2; val hi = f.c + f.d / 2
                for (i in 0 until f.n) {
                    val a = f.a + i * f.w; val b = a + f.w
                    if (f.axis == 0) box(view.yaw, lo, a, hi, b, floorY, f.h.toFloat(), SHELF)
                    else box(view.yaw, a, lo, b, hi, floorY, f.h.toFloat(), SHELF)
                }
            }
        }
        if (phone != null) drawPhone(phone)
        GLES20.glDisableVertexAttribArray(position)
        drawPoints(items)
        return labels(view, items, e, floorY, width, height)
    }

    private fun useLines(lines: java.nio.FloatBuffer) {
        lines.position(0)
        GLES20.glVertexAttribPointer(position, 3, GLES20.GL_FLOAT, false, 0, lines)
    }

    private fun drawPoints(items: List<MapItem>) {
        val n = minOf(items.size, ShelfMap.MAX_ITEMS)
        if (n == 0) return
        pointBuffer.clear()
        for (i in 0 until n) items[i].position.let { pointBuffer.put(it.x).put(it.y).put(it.z) }
        pointBuffer.position(0)
        GLES20.glUseProgram(pointProgram)
        GLES20.glUniformMatrix4fv(pointMvp, 1, false, viewProjection, 0)
        GLES20.glUniform3fv(pointColor, 1, POINT, 0)
        GLES20.glEnableVertexAttribArray(pointPosition)
        GLES20.glVertexAttribPointer(pointPosition, 3, GLES20.GL_FLOAT, false, 0, pointBuffer)
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, n)
        GLES20.glDisableVertexAttribArray(pointPosition)
    }

    private fun labels(view: StoreView?, items: List<MapItem>, eye: Vec3, floorY: Float, width: Int, height: Int): List<ScreenLabel> {
        val out = ArrayList<ScreenLabel>()
        if (view?.room != null) {
            val cs = Math.cos(view.yaw); val sn = Math.sin(view.yaw)
            for (s in StoreLabels.shelves(view)) {
                val x = (s.u * cs - s.v * sn).toFloat()
                val z = (s.u * sn + s.v * cs).toFloat()
                screen(x, floorY + s.top.toFloat() + SHELF_LABEL_LIFT_M, z, width, height, SHELF_LABEL_REACH)?.let { out += ScreenLabel(it[0], it[1], s.text, true) }
            }
        }
        // 近づいた点だけ名前を出す
        for (item in items.filter { it.position.distanceTo(eye) < NAME_RANGE_M }.sortedByDescending { it.seen }) {
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
        val SHELF = rgb(0xc8ccd2)
        val FLOOR = rgb(0x565d68)
        val ME = rgb(0xf0efec)
        val ARROW = rgb(0xff7a45)
        val POINT = rgb(0xffd23f)
        const val LINE_WIDTH = 2f
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
            attribute vec3 a_Position;
            void main() {
                gl_Position = u_Mvp * vec4(a_Position, 1.0);
                gl_PointSize = 12.0;
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
