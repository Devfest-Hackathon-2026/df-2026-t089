package com.example.store3d

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.opengl.GLES11Ext
import android.opengl.GLES20.*
import android.opengl.GLUtils
import android.opengl.Matrix
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

private fun floats(vararg v: Float): FloatBuffer =
    ByteBuffer.allocateDirect(v.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(v); position(0) }

private fun program(vs: String, fs: String): Int {
    fun sh(type: Int, src: String) = glCreateShader(type).also {
        glShaderSource(it, src); glCompileShader(it)
        val ok = IntArray(1); glGetShaderiv(it, GL_COMPILE_STATUS, ok, 0)
        check(ok[0] != 0) { glGetShaderInfoLog(it) }
    }
    return glCreateProgram().also { glAttachShader(it, sh(GL_VERTEX_SHADER, vs)); glAttachShader(it, sh(GL_FRAGMENT_SHADER, fs)); glLinkProgram(it) }
}

fun colorOf(label: String): Int = when (label) {
    UNKNOWN -> Color.rgb(150, 150, 150)
    "床" -> Color.rgb(70, 72, 78)
    else -> Color.HSVToColor(floatArrayOf(((label.hashCode() * 47L) % 360 + 360) % 360f, 0.55f, 0.95f))
}

/** Full-screen ARCore camera image. */
class CameraBackground {
    var texId = 0; private set
    private var prog = 0
    private val ndc = floats(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
    private val uv = floats(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)

    fun init() {
        val t = IntArray(1); glGenTextures(1, t, 0); texId = t[0]
        glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId)
        glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        prog = program(
            "attribute vec2 p; attribute vec2 t; varying vec2 v; void main(){ v=t; gl_Position=vec4(p,0.,1.); }",
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float; varying vec2 v; uniform samplerExternalOES s; void main(){ gl_FragColor=texture2D(s,v); }")
    }

    fun draw(frame: Frame) {
        if (frame.hasDisplayGeometryChanged())
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, ndc, Coordinates2d.TEXTURE_NORMALIZED, uv)
        if (frame.timestamp == 0L) return
        glDisable(GL_DEPTH_TEST); glDepthMask(false)
        glUseProgram(prog)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId)
        val p = glGetAttribLocation(prog, "p"); val t = glGetAttribLocation(prog, "t")
        glVertexAttribPointer(p, 2, GL_FLOAT, false, 0, ndc); glEnableVertexAttribArray(p)
        glVertexAttribPointer(t, 2, GL_FLOAT, false, 0, uv); glEnableVertexAttribArray(t)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(p); glDisableVertexAttribArray(t)
        glDepthMask(true); glEnable(GL_DEPTH_TEST)
    }
}

/** Shaded boxes + a text label on each shelf front (or freezer top). */
class BoxDrawer {
    private var boxProg = 0
    private var lblProg = 0
    private val textures = HashMap<String, Pair<Int, Float>>() // texture id, width/height
    private val cube: FloatBuffer
    private val quad = floats(-.5f, -.5f, 0f, 0f, 1f, .5f, -.5f, 0f, 1f, 1f, -.5f, .5f, 0f, 0f, 0f, .5f, .5f, 0f, 1f, 0f)
    private val m = FloatArray(16); private val mvp = FloatArray(16); private val vp = FloatArray(16)

    init {
        val v = ArrayList<Float>()
        // 6 faces (no culling, so winding does not matter)
        for (ax in 0..2) for (sg in intArrayOf(-1, 1)) {
            val u = (ax + 1) % 3; val w = (ax + 2) % 3
            val corners = listOf(-1 to -1, 1 to -1, 1 to 1, -1 to -1, 1 to 1, -1 to 1)
            for ((a, b) in corners) {
                val p = FloatArray(3); p[ax] = sg * .5f; p[u] = a * .5f; p[w] = b * .5f
                val n = FloatArray(3); n[ax] = sg.toFloat()
                v += p.toList(); v += n.toList()
            }
        }
        cube = floats(*v.toFloatArray())
    }

    fun init() {
        textures.clear()
        boxProg = program(
            "uniform mat4 mvp; uniform mat4 model; attribute vec3 p; attribute vec3 n; varying float l;" +
                "void main(){ vec3 w=normalize(mat3(model)*n); l=0.45+0.55*max(dot(w,normalize(vec3(0.4,1.0,0.3))),0.); gl_Position=mvp*vec4(p,1.); }",
            "precision mediump float; uniform vec4 c; varying float l; void main(){ gl_FragColor=vec4(c.rgb*l,c.a); }")
        lblProg = program(
            "uniform mat4 mvp; attribute vec3 p; attribute vec2 t; varying vec2 v; void main(){ v=t; gl_Position=mvp*vec4(p,1.); }",
            "precision mediump float; uniform sampler2D s; varying vec2 v; void main(){ gl_FragColor=texture2D(s,v); }")
    }

    fun draw(boxes: List<Box>, view: FloatArray, proj: FloatArray, alpha: Float) {
        Matrix.multiplyMM(vp, 0, proj, 0, view, 0)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(boxProg)
        val p = glGetAttribLocation(boxProg, "p"); val n = glGetAttribLocation(boxProg, "n")
        cube.position(0); glVertexAttribPointer(p, 3, GL_FLOAT, false, 24, cube); glEnableVertexAttribArray(p)
        cube.position(3); glVertexAttribPointer(n, 3, GL_FLOAT, false, 24, cube); glEnableVertexAttribArray(n)
        for (b in boxes) {
            Matrix.setIdentityM(m, 0)
            Matrix.translateM(m, 0, b.center[0], b.center[1], b.center[2])
            Matrix.rotateM(m, 0, Math.toDegrees(b.yaw.toDouble()).toFloat(), 0f, 1f, 0f)
            Matrix.scaleM(m, 0, b.size[0], b.size[1], b.size[2])
            Matrix.multiplyMM(mvp, 0, vp, 0, m, 0)
            glUniformMatrix4fv(glGetUniformLocation(boxProg, "mvp"), 1, false, mvp, 0)
            glUniformMatrix4fv(glGetUniformLocation(boxProg, "model"), 1, false, m, 0)
            val c = colorOf(b.label)
            glUniform4f(glGetUniformLocation(boxProg, "c"), Color.red(c) / 255f, Color.green(c) / 255f, Color.blue(c) / 255f, alpha)
            glDrawArrays(GL_TRIANGLES, 0, 36)
        }
        glDisableVertexAttribArray(p); glDisableVertexAttribArray(n)

        glUseProgram(lblProg)
        val lp = glGetAttribLocation(lblProg, "p"); val lt = glGetAttribLocation(lblProg, "t")
        quad.position(0); glVertexAttribPointer(lp, 3, GL_FLOAT, false, 20, quad); glEnableVertexAttribArray(lp)
        quad.position(3); glVertexAttribPointer(lt, 2, GL_FLOAT, false, 20, quad); glEnableVertexAttribArray(lt)
        for (b in boxes) {
            if (b.kind == Kind.FLOOR) continue
            val (tex, aspect) = texture(b.label)
            val w = minOf(b.size[0] * 0.9f, 0.18f * aspect); val h = w / aspect
            Matrix.setIdentityM(m, 0)
            Matrix.translateM(m, 0, b.center[0], b.center[1], b.center[2])
            Matrix.rotateM(m, 0, Math.toDegrees(b.yaw.toDouble()).toFloat(), 0f, 1f, 0f)
            if (b.kind == Kind.FREEZER) {
                Matrix.translateM(m, 0, 0f, b.size[1] / 2 + 0.01f, 0f); Matrix.rotateM(m, 0, -90f, 1f, 0f, 0f)
            } else Matrix.translateM(m, 0, 0f, 0f, b.size[2] / 2 + 0.01f)
            Matrix.scaleM(m, 0, w, h, 1f)
            Matrix.multiplyMM(mvp, 0, vp, 0, m, 0)
            glUniformMatrix4fv(glGetUniformLocation(lblProg, "mvp"), 1, false, mvp, 0)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        }
        glDisableVertexAttribArray(lp); glDisableVertexAttribArray(lt)
        glDisable(GL_BLEND)
    }

    private fun texture(label: String) = textures.getOrPut(label) {
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 48f; color = Color.WHITE }
        val bmp = Bitmap.createBitmap(paint.measureText(label).toInt() + 24, 68, Bitmap.Config.ARGB_8888)
        Canvas(bmp).apply { drawColor(Color.argb(200, 20, 20, 20)); drawText(label, 12f, 50f, paint) }
        val t = IntArray(1); glGenTextures(1, t, 0)
        glBindTexture(GL_TEXTURE_2D, t[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GL_TEXTURE_2D, 0, bmp, 0)
        Pair(t[0], bmp.width / bmp.height.toFloat()).also { bmp.recycle() }
    }
}
