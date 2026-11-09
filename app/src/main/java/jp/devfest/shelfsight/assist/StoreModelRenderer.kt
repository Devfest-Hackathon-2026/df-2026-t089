package jp.devfest.shelfsight.assist

import android.opengl.GLES20
import android.opengl.Matrix
import jp.devfest.shelfsight.ar.GlUtil
import jp.devfest.shelfsight.mapping.OrbitCamera
import jp.devfest.shelfsight.model.Vec3

/** 推定した店（[StoreView]）を箱の集まりとして描く。GL スレッドから使う。 */
internal class StoreModelRenderer {
    private val program = GlUtil.createProgram(VERTEX, FRAGMENT)
    private val position = GLES20.glGetAttribLocation(program, "a_Position")
    private val shade = GLES20.glGetAttribLocation(program, "a_Shade")
    private val mvp = GLES20.glGetUniformLocation(program, "u_Mvp")
    private val color = GLES20.glGetUniformLocation(program, "u_Color")
    private val cube = GlUtil.floatBuffer(CUBE)
    private val viewProjection = FloatArray(16)
    private val model = FloatArray(16)
    private val result = FloatArray(16)

    /** @param camera 今いる場所。[floorY] はそこから見積もった床の高さ。 */
    fun draw(view: StoreView?, orbit: OrbitCamera, width: Int, height: Int, camera: Vec3?, floorY: Float) {
        GLES20.glClearColor(0.08f, 0.09f, 0.11f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        val projection = FloatArray(16)
        val lookAt = FloatArray(16)
        Matrix.perspectiveM(projection, 0, 50f, width.toFloat() / height.coerceAtLeast(1), 0.1f, 200f)
        val e = orbit.eye(); val c = orbit.center()
        Matrix.setLookAtM(lookAt, 0, e.x, e.y, e.z, c.x, c.y, c.z, 0f, 1f, 0f)
        Matrix.multiplyMM(viewProjection, 0, projection, 0, lookAt, 0)
        GLES20.glUseProgram(program)
        GLES20.glEnableVertexAttribArray(position)
        GLES20.glEnableVertexAttribArray(shade)
        cube.position(0)
        GLES20.glVertexAttribPointer(position, 3, GLES20.GL_FLOAT, false, 16, cube)
        cube.position(3)
        GLES20.glVertexAttribPointer(shade, 1, GLES20.GL_FLOAT, false, 16, cube)
        if (view?.room != null) {
            val r = view.room
            box(view.yaw, r[0], r[1], r[2], r[3], floorY - 0.02f, 0.02f, FLOOR)
            for (f in view.fixtures) {
                val lo = f.c - f.d / 2; val hi = f.c + f.d / 2
                val col = COLORS.getValue(f.kind)
                // モジュールの境目が見えるよう、1 つずつ少し隙間を空けて描く
                for (i in 0 until f.n) {
                    val a = f.a + i * f.w + GAP; val b = f.a + (i + 1) * f.w - GAP
                    if (f.axis == 0) box(view.yaw, lo, a, hi, b, floorY, f.h.toFloat(), col)
                    else box(view.yaw, a, lo, b, hi, floorY, f.h.toFloat(), col)
                }
            }
        }
        if (camera != null) marker(camera, floorY)
        GLES20.glDisableVertexAttribArray(position)
        GLES20.glDisableVertexAttribArray(shade)
    }

    private fun box(yaw: Double, u0: Double, v0: Double, u1: Double, v1: Double, y: Float, height: Float, col: FloatArray) {
        val cu = (u0 + u1) / 2; val cv = (v0 + v1) / 2
        val cs = Math.cos(yaw); val sn = Math.sin(yaw)
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, (cu * cs - cv * sn).toFloat(), y + height / 2, (cu * sn + cv * cs).toFloat())
        Matrix.rotateM(model, 0, -Math.toDegrees(yaw).toFloat(), 0f, 1f, 0f)
        Matrix.scaleM(model, 0, (u1 - u0).toFloat(), height, (v1 - v0).toFloat())
        drawCube(col)
    }

    private fun marker(p: Vec3, floorY: Float) {
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, p.x, floorY + 0.8f, p.z)
        Matrix.scaleM(model, 0, 0.3f, 1.6f, 0.3f)
        drawCube(ME)
    }

    private fun drawCube(col: FloatArray) {
        Matrix.multiplyMM(result, 0, viewProjection, 0, model, 0)
        GLES20.glUniformMatrix4fv(mvp, 1, false, result, 0)
        GLES20.glUniform3fv(color, 1, col, 0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, CUBE.size / 4)
    }

    private companion object {
        const val GAP = 0.01
        fun rgb(hex: Int) = floatArrayOf((hex shr 16 and 255) / 255f, (hex shr 8 and 255) / 255f, (hex and 255) / 255f)
        val COLORS = mapOf(
            "island" to rgb(0x3987e5), "wall" to rgb(0xd95926), "case" to rgb(0x199e70),
            "end" to rgb(0xc98500), "register" to rgb(0xd55181),
        )
        val FLOOR = rgb(0x2d333c)
        val ME = rgb(0xf0efec)

        /** 単位立方体の三角形（x, y, z, 明るさ）。上の面を明るく、横の面は向きで明るさを変えて形を見せる。 */
        val CUBE: FloatArray = run {
            val v = ArrayList<Float>()
            fun quad(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray, s: Float) {
                for (p in listOf(a, b, c, a, c, d)) { v += p[0] - 0.5f; v += p[1] - 0.5f; v += p[2] - 0.5f; v += s }
            }
            fun p(x: Int, y: Int, z: Int) = floatArrayOf(x.toFloat(), y.toFloat(), z.toFloat())
            quad(p(0, 1, 0), p(0, 1, 1), p(1, 1, 1), p(1, 1, 0), 1f)
            quad(p(0, 0, 0), p(1, 0, 0), p(1, 0, 1), p(0, 0, 1), 0.4f)
            quad(p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1), 0.8f)
            quad(p(0, 0, 0), p(0, 1, 0), p(1, 1, 0), p(1, 0, 0), 0.6f)
            quad(p(1, 0, 0), p(1, 1, 0), p(1, 1, 1), p(1, 0, 1), 0.7f)
            quad(p(0, 0, 0), p(0, 0, 1), p(0, 1, 1), p(0, 1, 0), 0.5f)
            v.toFloatArray()
        }

        const val VERTEX = """
            uniform mat4 u_Mvp;
            attribute vec3 a_Position;
            attribute float a_Shade;
            varying float v_Shade;
            void main() {
                gl_Position = u_Mvp * vec4(a_Position, 1.0);
                v_Shade = a_Shade;
            }
        """
        const val FRAGMENT = """
            precision mediump float;
            uniform vec3 u_Color;
            varying float v_Shade;
            void main() {
                gl_FragColor = vec4(u_Color * v_Shade, 1.0);
            }
        """
    }
}
