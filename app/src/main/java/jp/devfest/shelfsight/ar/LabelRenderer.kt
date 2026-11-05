package jp.devfest.shelfsight.ar

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.opengl.GLES20
import android.opengl.GLUtils
import android.opengl.Matrix
import jp.devfest.shelfsight.model.ProductCategory
import jp.devfest.shelfsight.model.Vec3

/** 3D 空間の 1 点に、立方体のマーカーと常にカメラを向く文字ラベルを描く。 */
internal class LabelRenderer {
    private data class TextureKey(val category: ProductCategory, val hasMeasuredDepth: Boolean)

    private val textures = HashMap<TextureKey, Int>()

    private val labelProgram = GlUtil.createProgram(LABEL_VERTEX_SHADER, LABEL_FRAGMENT_SHADER)
    private val labelPosition = GLES20.glGetAttribLocation(labelProgram, "a_Position")
    private val labelTexCoord = GLES20.glGetAttribLocation(labelProgram, "a_TexCoord")
    private val labelMvp = GLES20.glGetUniformLocation(labelProgram, "u_Mvp")
    private val labelTexture = GLES20.glGetUniformLocation(labelProgram, "u_Texture")

    private val cubeProgram = GlUtil.createProgram(CUBE_VERTEX_SHADER, CUBE_FRAGMENT_SHADER)
    private val cubePosition = GLES20.glGetAttribLocation(cubeProgram, "a_Position")
    private val cubeNormal = GLES20.glGetAttribLocation(cubeProgram, "a_Normal")
    private val cubeMvp = GLES20.glGetUniformLocation(cubeProgram, "u_Mvp")
    private val cubeColor = GLES20.glGetUniformLocation(cubeProgram, "u_Color")

    private val quadPositions = GlUtil.floatBuffer(floatArrayOf(-0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f))
    private val quadTexCoords = GlUtil.floatBuffer(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))
    private val cubePositions = GlUtil.floatBuffer(CUBE_POSITIONS)
    private val cubeNormals = GlUtil.floatBuffer(CUBE_NORMALS)

    private val model = FloatArray(16)
    private val mvp = FloatArray(16)

    /** 手前のラベルが読めるよう、呼び出し側は奥から順に描く。 */
    fun drawCube(category: ProductCategory, position: Vec3, scale: Float, viewProjection: FloatArray) {
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, position.x, position.y, position.z)
        val size = CUBE_SIZE_M * scale
        Matrix.scaleM(model, 0, size, size, size)
        Matrix.multiplyMM(mvp, 0, viewProjection, 0, model, 0)

        GLES20.glUseProgram(cubeProgram)
        GLES20.glUniformMatrix4fv(cubeMvp, 1, false, mvp, 0)
        GLES20.glUniform4f(
            cubeColor,
            Color.red(category.argb) / 255f,
            Color.green(category.argb) / 255f,
            Color.blue(category.argb) / 255f,
            1f,
        )
        GLES20.glVertexAttribPointer(cubePosition, 3, GLES20.GL_FLOAT, false, 0, cubePositions)
        GLES20.glVertexAttribPointer(cubeNormal, 3, GLES20.GL_FLOAT, false, 0, cubeNormals)
        GLES20.glEnableVertexAttribArray(cubePosition)
        GLES20.glEnableVertexAttribArray(cubeNormal)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, CUBE_POSITIONS.size / 3)
        GLES20.glDisableVertexAttribArray(cubePosition)
        GLES20.glDisableVertexAttribArray(cubeNormal)
    }

    fun drawLabel(
        category: ProductCategory,
        hasMeasuredDepth: Boolean,
        position: Vec3,
        scale: Float,
        view: FloatArray,
        viewProjection: FloatArray,
    ) {
        val height = LABEL_HEIGHT_M * scale
        val width = height * TEXTURE_WIDTH / TEXTURE_HEIGHT
        val lift = height * 0.5f + CUBE_SIZE_M * scale

        // ビュー行列の各行はカメラの右・上・後ろ方向をワールド座標で表している。
        model[0] = view[0] * width; model[1] = view[4] * width; model[2] = view[8] * width; model[3] = 0f
        model[4] = view[1] * height; model[5] = view[5] * height; model[6] = view[9] * height; model[7] = 0f
        model[8] = view[2]; model[9] = view[6]; model[10] = view[10]; model[11] = 0f
        model[12] = position.x + view[1] * lift
        model[13] = position.y + view[5] * lift
        model[14] = position.z + view[9] * lift
        model[15] = 1f
        Matrix.multiplyMM(mvp, 0, viewProjection, 0, model, 0)

        GLES20.glDisable(GLES20.GL_DEPTH_TEST) // 視認性を優先し、物に隠れても表示する
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA) // Bitmap は乗算済みアルファ
        GLES20.glUseProgram(labelProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureFor(category, hasMeasuredDepth))
        GLES20.glUniform1i(labelTexture, 0)
        GLES20.glUniformMatrix4fv(labelMvp, 1, false, mvp, 0)
        GLES20.glVertexAttribPointer(labelPosition, 2, GLES20.GL_FLOAT, false, 0, quadPositions)
        GLES20.glVertexAttribPointer(labelTexCoord, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        GLES20.glEnableVertexAttribArray(labelPosition)
        GLES20.glEnableVertexAttribArray(labelTexCoord)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(labelPosition)
        GLES20.glDisableVertexAttribArray(labelTexCoord)
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    private fun textureFor(category: ProductCategory, hasMeasuredDepth: Boolean): Int =
        textures.getOrPut(TextureKey(category, hasMeasuredDepth)) {
            val text = if (hasMeasuredDepth) category.labelJa else "${category.labelJa}（距離は推定）"
            val bitmap = drawLabelBitmap(text, category.argb)
            val ids = IntArray(1)
            GLES20.glGenTextures(1, ids, 0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, ids[0])
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
            bitmap.recycle()
            ids[0]
        }

    private fun drawLabelBitmap(text: String, argb: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(TEXTURE_WIDTH, TEXTURE_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val background = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = argb }
        val rect = RectF(0f, 0f, TEXTURE_WIDTH.toFloat(), TEXTURE_HEIGHT.toFloat())
        canvas.drawRoundRect(rect, 28f, 28f, background)

        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textAlign = Paint.Align.CENTER
            textSize = 80f
            isFakeBoldText = true
        }
        val available = TEXTURE_WIDTH - 40f
        val measured = paint.measureText(text)
        if (measured > available) paint.textSize *= available / measured
        val baseline = TEXTURE_HEIGHT / 2f - (paint.descent() + paint.ascent()) / 2f
        canvas.drawText(text, TEXTURE_WIDTH / 2f, baseline, paint)
        return bitmap
    }

    private companion object {
        const val TEXTURE_WIDTH = 512
        const val TEXTURE_HEIGHT = 128
        const val CUBE_SIZE_M = 0.05f
        const val LABEL_HEIGHT_M = 0.07f

        const val LABEL_VERTEX_SHADER = """
            uniform mat4 u_Mvp;
            attribute vec4 a_Position;
            attribute vec2 a_TexCoord;
            varying vec2 v_TexCoord;
            void main() {
                gl_Position = u_Mvp * a_Position;
                v_TexCoord = a_TexCoord;
            }
        """
        const val LABEL_FRAGMENT_SHADER = """
            precision mediump float;
            uniform sampler2D u_Texture;
            varying vec2 v_TexCoord;
            void main() {
                gl_FragColor = texture2D(u_Texture, v_TexCoord);
            }
        """
        const val CUBE_VERTEX_SHADER = """
            uniform mat4 u_Mvp;
            attribute vec4 a_Position;
            attribute vec3 a_Normal;
            varying float v_Light;
            void main() {
                gl_Position = u_Mvp * a_Position;
                v_Light = 0.65 + 0.35 * dot(a_Normal, normalize(vec3(0.4, 0.8, 0.45)));
            }
        """
        const val CUBE_FRAGMENT_SHADER = """
            precision mediump float;
            uniform vec4 u_Color;
            varying float v_Light;
            void main() {
                gl_FragColor = vec4(u_Color.rgb * v_Light, u_Color.a);
            }
        """

        /** 辺の長さ 1 の立方体。面ごとに 2 枚の三角形で構成する。 */
        val CUBE_FACES = listOf(
            floatArrayOf(0f, 0f, 1f) to floatArrayOf(-.5f, -.5f, .5f, .5f, -.5f, .5f, .5f, .5f, .5f, -.5f, .5f, .5f),
            floatArrayOf(0f, 0f, -1f) to floatArrayOf(.5f, -.5f, -.5f, -.5f, -.5f, -.5f, -.5f, .5f, -.5f, .5f, .5f, -.5f),
            floatArrayOf(1f, 0f, 0f) to floatArrayOf(.5f, -.5f, .5f, .5f, -.5f, -.5f, .5f, .5f, -.5f, .5f, .5f, .5f),
            floatArrayOf(-1f, 0f, 0f) to floatArrayOf(-.5f, -.5f, -.5f, -.5f, -.5f, .5f, -.5f, .5f, .5f, -.5f, .5f, -.5f),
            floatArrayOf(0f, 1f, 0f) to floatArrayOf(-.5f, .5f, .5f, .5f, .5f, .5f, .5f, .5f, -.5f, -.5f, .5f, -.5f),
            floatArrayOf(0f, -1f, 0f) to floatArrayOf(-.5f, -.5f, -.5f, .5f, -.5f, -.5f, .5f, -.5f, .5f, -.5f, -.5f, .5f),
        )
        val TRIANGLE_CORNERS = intArrayOf(0, 1, 2, 0, 2, 3)
        val CUBE_POSITIONS: FloatArray = CUBE_FACES.flatMap { (_, corners) ->
            TRIANGLE_CORNERS.flatMap { c -> listOf(corners[c * 3], corners[c * 3 + 1], corners[c * 3 + 2]) }
        }.toFloatArray()
        val CUBE_NORMALS: FloatArray = CUBE_FACES.flatMap { (normal, _) ->
            List(TRIANGLE_CORNERS.size) { normal.toList() }.flatten()
        }.toFloatArray()
    }
}
