package jp.devfest.shelfsight.mapping

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.util.Log
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.SessionPausedException
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import jp.devfest.shelfsight.ar.BackgroundRenderer

/** ARCore の更新と描画を行う。地図を表示している間も、裏で記録と追跡を続ける。 */
class MappingRenderer(
    private val mapper: StoreMapper,
    private val displayRotation: () -> Int,
) : GLSurfaceView.Renderer {

    @Volatile private var session: Session? = null
    private var boundSession: Session? = null

    private var background: BackgroundRenderer? = null
    private var mapRenderer: MapRenderer? = null
    private var viewWidth = 0
    private var viewHeight = 0

    private val view = FloatArray(16)
    private val projection = FloatArray(16)
    private val viewProjection = FloatArray(16)

    fun attachSession(session: Session?) {
        this.session = session
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        background = BackgroundRenderer()
        mapRenderer = MapRenderer()
        mapper.onGlContextLost()
        boundSession = null
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewWidth = width
        viewHeight = height
        boundSession = null
    }

    override fun onDrawFrame(gl: GL10?) {
        val showMap = mapper.state.value.showMap
        if (showMap) GLES20.glClearColor(0.05f, 0.05f, 0.08f, 1f) else GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        val current = session ?: return
        val backgroundRenderer = background ?: return
        val map = mapRenderer ?: return

        if (boundSession !== current) {
            current.setCameraTextureName(backgroundRenderer.textureId)
            current.setDisplayGeometry(displayRotation(), viewWidth, viewHeight)
            boundSession = current
        }

        val frame = try {
            current.update()
        } catch (e: CameraNotAvailableException) {
            Log.w(TAG, "Camera not available", e)
            return
        } catch (_: SessionPausedException) {
            return
        }
        val camera = frame.camera
        mapper.onFrame(current, frame, camera)

        if (showMap) {
            drawMapView(map)
        } else {
            backgroundRenderer.draw(frame)
            if (camera.trackingState == TrackingState.TRACKING) {
                camera.getViewMatrix(view, 0)
                camera.getProjectionMatrix(projection, 0, NEAR_M, FAR_M)
                Matrix.multiplyMM(viewProjection, 0, projection, 0, view, 0)
                // カメラ映像に重ね、記録できた範囲が分かるようにする。
                map.draw(mapper, viewProjection, projection[5], viewHeight, Float.MAX_VALUE, showTrail = false)
            }
        }
    }

    private fun drawMapView(map: MapRenderer) {
        if (viewHeight == 0) return
        val eye = mapper.orbit.eye()
        val center = mapper.orbit.center()
        Matrix.setLookAtM(view, 0, eye.x, eye.y, eye.z, center.x, center.y, center.z, 0f, 1f, 0f)
        Matrix.perspectiveM(projection, 0, FIELD_OF_VIEW_DEGREES, viewWidth.toFloat() / viewHeight, NEAR_M, MAP_FAR_M)
        Matrix.multiplyMM(viewProjection, 0, projection, 0, view, 0)
        map.draw(mapper, viewProjection, projection[5], viewHeight, mapper.ceilingClipY, showTrail = true)
    }

    private companion object {
        const val TAG = "MappingRenderer"
        const val NEAR_M = 0.1f
        const val FAR_M = 100f
        const val MAP_FAR_M = 1000f
        const val FIELD_OF_VIEW_DEGREES = 50f
    }
}
