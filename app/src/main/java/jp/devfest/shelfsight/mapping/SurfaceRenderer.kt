package jp.devfest.shelfsight.mapping

import android.opengl.GLES20
import jp.devfest.shelfsight.ar.GlUtil
import kotlin.math.abs

/** 面を、塗りつぶしと輪郭線で描く。点で補正する前の平面は ARCore の凸多角形のまま描く。 */
internal class SurfaceRenderer {
    private val program = GlUtil.createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
    private val positionHandle = GLES20.glGetAttribLocation(program, "a_Position")
    private val viewProjectionHandle = GLES20.glGetUniformLocation(program, "u_ViewProjection")
    private val modelHandle = GLES20.glGetUniformLocation(program, "u_Model")
    private val colorHandle = GLES20.glGetUniformLocation(program, "u_Color")
    private val clipYHandle = GLES20.glGetUniformLocation(program, "u_ClipY")
    private val model = FloatArray(16)

    /** 床と棚の上面を不透明で描く。点より奥に寄せて、重ねたときに点が隠れないようにする。 */
    fun drawHorizontal(fused: List<DrawableSurface>, patches: List<SurfacePatch>, viewProjection: FloatArray, clipY: Float) {
        begin(viewProjection, clipY)
        for (surface in fused) {
            if (surface.kind != SurfaceKind.WALL) draw(surface, colorOf(surface.kind), 1f)
        }
        for (patch in patches) {
            if (patch.kind != SurfaceKind.WALL) draw(patch, colorOf(patch.kind), 1f)
        }
        end()
    }

    /** 壁は向こう側の棚が見えるよう半透明にする。深度を書かないので描く順序に関係なく重ねられる。 */
    fun drawWalls(fused: List<DrawableSurface>, patches: List<SurfacePatch>, viewProjection: FloatArray, clipY: Float) {
        begin(viewProjection, clipY)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glDepthMask(false)
        for (surface in fused) {
            if (surface.kind == SurfaceKind.WALL) draw(surface, WALL_COLOR, WALL_ALPHA)
        }
        for (patch in patches) {
            if (patch.kind == SurfaceKind.WALL) draw(patch, WALL_COLOR, WALL_ALPHA)
        }
        GLES20.glDepthMask(true)
        GLES20.glDisable(GLES20.GL_BLEND)
        end()
    }

    private fun begin(viewProjection: FloatArray, clipY: Float) {
        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(viewProjectionHandle, 1, false, viewProjection, 0)
        GLES20.glUniform1f(clipYHandle, clipY)
        GLES20.glEnable(GLES20.GL_POLYGON_OFFSET_FILL)
        GLES20.glPolygonOffset(1f, 1f)
        GLES20.glLineWidth(OUTLINE_WIDTH_PX)
        GLES20.glEnableVertexAttribArray(positionHandle)
    }

    private fun end() {
        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glDisable(GLES20.GL_POLYGON_OFFSET_FILL)
    }

    private fun colorOf(kind: SurfaceKind) = when (kind) {
        SurfaceKind.FLOOR -> FLOOR_COLOR
        SurfaceKind.RAISED -> RAISED_COLOR
        SurfaceKind.WALL -> WALL_COLOR
    }

    private fun draw(surface: DrawableSurface, color: FloatArray, alpha: Float) {
        surface.writeModel(model)
        GLES20.glUniformMatrix4fv(modelHandle, 1, false, model, 0)
        setFill(color, alpha, surface.normal[0], surface.normal[1], surface.normal[2])
        surface.triangles.position(0)
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 0, surface.triangles)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, surface.triangleVertices)
        GLES20.glUniform4f(colorHandle, color[0], color[1], color[2], 1f)
        surface.outline.position(0)
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 0, surface.outline)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, surface.outlineVertices)
    }

    private fun draw(patch: SurfacePatch, color: FloatArray, alpha: Float) {
        if (patch.vertexCount < 3) return
        val m = patch.worldFromPlane
        GLES20.glUniformMatrix4fv(modelHandle, 1, false, m, 0)
        setFill(color, alpha, m[4], m[5], m[6])
        patch.buffer.position(0)
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 0, patch.buffer)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, patch.vertexCount)
        GLES20.glUniform4f(colorHandle, color[0], color[1], color[2], 1f)
        GLES20.glDrawArrays(GLES20.GL_LINE_LOOP, 0, patch.vertexCount)
    }

    /** 面の向きで明るさを変えて、同じ色の壁どうしでも向きの違いが分かるようにする。 */
    private fun setFill(color: FloatArray, alpha: Float, nx: Float, ny: Float, nz: Float) {
        val shade = SHADE_BASE + SHADE_RANGE * abs(nx * LIGHT_X + ny * LIGHT_Y + nz * LIGHT_Z)
        GLES20.glUniform4f(colorHandle, color[0] * shade, color[1] * shade, color[2] * shade, alpha)
    }

    private companion object {
        val FLOOR_COLOR = floatArrayOf(0.30f, 0.36f, 0.46f)
        val RAISED_COLOR = floatArrayOf(0.25f, 0.62f, 0.60f)
        val WALL_COLOR = floatArrayOf(0.85f, 0.85f, 0.80f)
        const val WALL_ALPHA = 0.55f
        const val OUTLINE_WIDTH_PX = 3f
        const val SHADE_BASE = 0.6f
        const val SHADE_RANGE = 0.4f
        const val LIGHT_X = 0.53f
        const val LIGHT_Y = 0.80f
        const val LIGHT_Z = 0.27f

        const val VERTEX_SHADER = """
            uniform mat4 u_ViewProjection;
            uniform mat4 u_Model;
            attribute vec3 a_Position;
            varying float v_WorldY;
            void main() {
                vec4 world = u_Model * vec4(a_Position, 1.0);
                v_WorldY = world.y;
                gl_Position = u_ViewProjection * world;
            }
        """
        const val FRAGMENT_SHADER = """
            precision mediump float;
            uniform vec4 u_Color;
            uniform float u_ClipY;
            varying float v_WorldY;
            void main() {
                if (v_WorldY > u_ClipY) discard;
                gl_FragColor = u_Color;
            }
        """
    }
}
