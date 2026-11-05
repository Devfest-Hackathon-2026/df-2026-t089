package jp.devfest.shelfsight.mapping

import android.opengl.GLES20
import android.opengl.Matrix
import java.nio.IntBuffer
import jp.devfest.shelfsight.ar.GlUtil

/** 地図の点群、歩いた軌跡、現在地を描く。AR 表示と地図表示で共通に使う。 */
internal class MapRenderer {
    private val program = GlUtil.createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
    private val positionHandle = GLES20.glGetAttribLocation(program, "a_Position")
    private val colorHandle = GLES20.glGetAttribLocation(program, "a_Color")
    private val viewProjectionHandle = GLES20.glGetUniformLocation(program, "u_ViewProjection")
    private val modelHandle = GLES20.glGetUniformLocation(program, "u_Model")
    private val pointScaleHandle = GLES20.glGetUniformLocation(program, "u_PointScale")
    private val fixedPointSizeHandle = GLES20.glGetUniformLocation(program, "u_FixedPointSize")
    private val clipYHandle = GLES20.glGetUniformLocation(program, "u_ClipY")

    private val identity = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
    private var trailWorld = FloatArray(0)
    private var trailBuffer = GlUtil.floatBuffer(FloatArray(0))
    private val markerBuffer = GlUtil.floatBuffer(FloatArray(6))

    /**
     * @param projectionScaleY 射影行列の [5] の要素。点の大きさを実寸に合わせるために使う
     * @param clipY この高さより上の点は描かない
     */
    fun draw(
        mapper: StoreMapper,
        viewProjection: FloatArray,
        projectionScaleY: Float,
        viewportHeight: Int,
        clipY: Float,
        showTrail: Boolean,
    ) {
        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(viewProjectionHandle, 1, false, viewProjection, 0)
        GLES20.glUniform1f(
            pointScaleHandle,
            viewportHeight * projectionScaleY / 2f * VoxelGrid.DEFAULT_VOXEL_SIZE_M,
        )
        GLES20.glUniform1f(fixedPointSizeHandle, 0f)
        GLES20.glUniform1f(clipYHandle, clipY)
        drawChunks(mapper)

        if (showTrail) {
            GLES20.glUniform1f(clipYHandle, Float.MAX_VALUE)
            GLES20.glUniformMatrix4fv(modelHandle, 1, false, identity, 0)
            drawTrail(mapper)
            drawCurrentPosition(mapper)
        }
    }

    private fun drawChunks(mapper: StoreMapper) {
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glEnableVertexAttribArray(colorHandle)
        for (chunk in mapper.chunks) {
            upload(chunk)
            if (chunk.uploadedVertexCount == 0) continue
            GLES20.glUniformMatrix4fv(modelHandle, 1, false, chunk.worldFromChunk, 0)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, chunk.vbo)
            GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, VERTEX_BYTES, 0)
            GLES20.glVertexAttribPointer(colorHandle, 4, GLES20.GL_UNSIGNED_BYTE, true, VERTEX_BYTES, 12)
            GLES20.glDrawArrays(GLES20.GL_POINTS, 0, chunk.uploadedVertexCount)
        }
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        GLES20.glDisableVertexAttribArray(colorHandle)
        GLES20.glDisableVertexAttribArray(positionHandle)
    }

    private fun upload(chunk: MapChunk) {
        val version = chunk.vertexVersion
        if (chunk.uploadedVersion == version) return
        val vertices = chunk.vertices
        if (chunk.vbo == 0) {
            val ids = IntArray(1)
            GLES20.glGenBuffers(1, ids, 0)
            chunk.vbo = ids[0]
        }
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, chunk.vbo)
        GLES20.glBufferData(
            GLES20.GL_ARRAY_BUFFER,
            vertices.size * Int.SIZE_BYTES,
            IntBuffer.wrap(vertices),
            GLES20.GL_DYNAMIC_DRAW,
        )
        chunk.uploadedVertexCount = vertices.size / VoxelGrid.INTS_PER_VERTEX
        chunk.uploadedVersion = version
    }

    private fun drawTrail(mapper: StoreMapper) {
        val size = mapper.trailSize
        if (size < 2) return
        val floats = size * 3
        if (trailWorld.size < floats) {
            trailWorld = FloatArray(floats * 2)
            trailBuffer = GlUtil.floatBuffer(trailWorld)
        }
        mapper.writeTrailWorld(trailWorld)
        trailBuffer.position(0)
        trailBuffer.put(trailWorld, 0, floats)
        trailBuffer.position(0)

        GLES20.glVertexAttrib4f(colorHandle, 1f, 0.84f, 0.1f, 1f)
        GLES20.glLineWidth(TRAIL_WIDTH_PX)
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 0, trailBuffer)
        GLES20.glDrawArrays(GLES20.GL_LINE_STRIP, 0, size)
        GLES20.glDisableVertexAttribArray(positionHandle)
    }

    private fun drawCurrentPosition(mapper: StoreMapper) {
        val p = mapper.currentPosition ?: return
        val f = mapper.currentForward
        markerBuffer.position(0)
        markerBuffer.put(p.x).put(p.y).put(p.z)
        markerBuffer.put(p.x + f.x * HEADING_LENGTH_M)
        markerBuffer.put(p.y + f.y * HEADING_LENGTH_M)
        markerBuffer.put(p.z + f.z * HEADING_LENGTH_M)
        markerBuffer.position(0)

        GLES20.glDisable(GLES20.GL_DEPTH_TEST) // 現在地は棚の裏にいても見えるようにする
        GLES20.glVertexAttrib4f(colorHandle, 0.1f, 0.9f, 1f, 1f)
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 0, markerBuffer)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2)
        GLES20.glUniform1f(fixedPointSizeHandle, MARKER_SIZE_PX)
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, 1)
        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    private companion object {
        const val VERTEX_BYTES = VoxelGrid.INTS_PER_VERTEX * Int.SIZE_BYTES
        const val TRAIL_WIDTH_PX = 6f
        const val MARKER_SIZE_PX = 36f
        const val HEADING_LENGTH_M = 0.8f

        const val VERTEX_SHADER = """
            uniform mat4 u_ViewProjection;
            uniform mat4 u_Model;
            uniform float u_PointScale;
            uniform float u_FixedPointSize;
            uniform float u_ClipY;
            attribute vec3 a_Position;
            attribute vec4 a_Color;
            varying vec4 v_Color;
            void main() {
                vec4 world = u_Model * vec4(a_Position, 1.0);
                gl_Position = u_ViewProjection * world;
                if (world.y > u_ClipY) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // クリップ範囲の外へ出して描かない
                }
                gl_PointSize = u_FixedPointSize > 0.0
                    ? u_FixedPointSize
                    : clamp(u_PointScale / gl_Position.w, 2.0, 24.0);
                v_Color = a_Color;
            }
        """
        const val FRAGMENT_SHADER = """
            precision mediump float;
            varying vec4 v_Color;
            void main() {
                gl_FragColor = v_Color;
            }
        """
    }
}
