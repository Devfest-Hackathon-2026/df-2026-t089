package jp.devfest.shelfsight.ar

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.util.Log
import com.google.ar.core.Anchor
import com.google.ar.core.Camera
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.SessionPausedException
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import jp.devfest.shelfsight.model.ProductCategory
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.scan.LabelMerging
import jp.devfest.shelfsight.scan.ShelfScanner
import kotlin.math.max

/**
 * ARCore のフレーム更新と描画を行う。Anchor の作成を含め、ARCore の操作はこの GL スレッドで行う。
 */
class ShelfArRenderer(
    private val scanner: ShelfScanner?,
    private val displayRotation: () -> Int,
    private val onLabelPlaced: (ProductCategory) -> Unit,
) : GLSurfaceView.Renderer {

    private class Label(val category: ProductCategory, val anchor: Anchor, val hasMeasuredDepth: Boolean) {
        val position: Vec3
            get() = anchor.pose.let { Vec3(it.tx(), it.ty(), it.tz()) }
    }

    private class ActiveSession(val session: Session, val uprightRotationDegrees: Int)

    @Volatile private var active: ActiveSession? = null
    private var boundSession: Session? = null

    private var background: BackgroundRenderer? = null
    private var labelRenderer: LabelRenderer? = null
    private var viewWidth = 0
    private var viewHeight = 0

    private val labels = ArrayList<Label>()
    private val view = FloatArray(16)
    private val projection = FloatArray(16)
    private val viewProjection = FloatArray(16)
    private val cameraToWorld = FloatArray(16)

    fun attachSession(session: Session?, uprightRotationDegrees: Int = 0) {
        active = session?.let { ActiveSession(it, uprightRotationDegrees) }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        background = BackgroundRenderer()
        labelRenderer = LabelRenderer()
        boundSession = null // 新しい GL コンテキストのテクスチャを設定し直す
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewWidth = width
        viewHeight = height
        boundSession = null // 画面サイズを ARCore に伝え直す
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val current = active ?: return
        val session = current.session
        val backgroundRenderer = background ?: return

        if (boundSession !== session) {
            session.setCameraTextureName(backgroundRenderer.textureId)
            session.setDisplayGeometry(displayRotation(), viewWidth, viewHeight)
            boundSession = session
        }

        val frame = try {
            session.update()
        } catch (e: CameraNotAvailableException) {
            Log.w(TAG, "Camera not available", e)
            return
        } catch (_: SessionPausedException) {
            return
        }

        backgroundRenderer.draw(frame)
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) return

        captureIfNeeded(frame, camera, current)
        placeNewLabels(session)
        drawLabels(camera)
    }

    private fun captureIfNeeded(frame: Frame, camera: Camera, current: ActiveSession) {
        val activeScanner = scanner ?: return
        camera.pose.toMatrix(cameraToWorld, 0)
        if (!activeScanner.wantsCapture(cameraToWorld)) return
        val depthEnabled = current.session.config.depthMode != Config.DepthMode.DISABLED
        FrameCapture.capture(frame, camera, current.uprightRotationDegrees, depthEnabled)
            ?.let(activeScanner::submit)
    }

    private fun placeNewLabels(session: Session) {
        val results = scanner?.results ?: return
        while (true) {
            val detection = results.poll() ?: break
            val p = detection.position
            val duplicate = LabelMerging.isDuplicate(
                detection.category, p, labels, { it.category }, { it.position },
            )
            if (duplicate) continue

            if (labels.size >= MAX_LABELS) labels.removeAt(0).anchor.detach()
            val anchor = session.createAnchor(Pose.makeTranslation(p.x, p.y, p.z))
            labels += Label(detection.category, anchor, detection.hasMeasuredDepth)
            onLabelPlaced(detection.category)
        }
    }

    private fun drawLabels(camera: Camera) {
        val renderer = labelRenderer ?: return
        if (labels.isEmpty()) return
        camera.getViewMatrix(view, 0)
        camera.getProjectionMatrix(projection, 0, NEAR_M, FAR_M)
        Matrix.multiplyMM(viewProjection, 0, projection, 0, view, 0)

        val cameraPose = camera.pose
        val eye = Vec3(cameraPose.tx(), cameraPose.ty(), cameraPose.tz())
        val visible = labels
            .filter { it.anchor.trackingState == TrackingState.TRACKING }
            .map { it to it.position }
            .sortedByDescending { (_, position) -> position.distanceTo(eye) }

        for ((label, position) in visible) {
            // 遠くのラベルも読めるよう、距離に応じて拡大する。
            val scale = max(1f, position.distanceTo(eye))
            renderer.drawCube(label.category, position, scale, viewProjection)
            renderer.drawLabel(label.category, label.hasMeasuredDepth, position, scale, view, viewProjection)
        }
    }

    private companion object {
        const val TAG = "ShelfArRenderer"
        const val NEAR_M = 0.1f
        const val FAR_M = 100f

        /** Anchor が多いと ARCore の負荷が上がるため、古いものから外す。 */
        const val MAX_LABELS = 30
    }
}
