package jp.devfest.shelfsight

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.display.DisplayManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.view.Display
import android.view.Surface
import android.view.View
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.FatalException
import com.google.ar.core.exceptions.UnavailableException
import jp.devfest.shelfsight.mapping.MappingRenderer
import jp.devfest.shelfsight.mapping.StoreMapper
import jp.devfest.shelfsight.perception.MediaPipeHandDetector
import jp.devfest.shelfsight.perception.MediaPipePersonDetector
import jp.devfest.shelfsight.perception.PerceptionPipeline
import jp.devfest.shelfsight.ui.MappingScreen

class MainActivity : ComponentActivity() {
    private val mapper = StoreMapper()
    private lateinit var perception: PerceptionPipeline

    private lateinit var surfaceView: GLSurfaceView
    private lateinit var renderer: MappingRenderer
    private var session: Session? = null
    private var installRequested = false
    private var permissionDenied = false

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            permissionDenied = !granted
            // 許可された場合は、続く onResume でセッションを開始する。
            if (!granted) mapper.setBlockingMessage("カメラの使用を許可してください")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        perception = PerceptionPipeline(
            MediaPipePersonDetector(this),
            MediaPipeHandDetector(this),
            getExternalFilesDir(null)?.resolve("perception_log.csv"),
        )
        renderer = MappingRenderer(mapper, perception, ::currentDisplayRotation)
        surfaceView = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(renderer)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            // 映像自体は読み上げ対象にせず、状況は Compose 側のテキストで伝える。
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }

        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                val state by mapper.state.collectAsStateWithLifecycle()
                val perceptionState by perception.state.collectAsStateWithLifecycle()
                MappingScreen(
                    state = state,
                    perception = perceptionState,
                    surfaceView = surfaceView,
                    orbit = mapper.orbit,
                    onToggleMap = { mapper.setShowMap(!state.showMap) },
                    onToggleCamera = { mapper.setShowCamera(!state.showCamera) },
                    onToggleRecording = { mapper.setRecording(!state.isRecording) },
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (!hasCameraPermission()) {
            if (!permissionDenied) requestCamera.launch(Manifest.permission.CAMERA)
            return
        }
        mapper.setBlockingMessage(null)
        val current = session ?: createSession() ?: return
        try {
            current.resume()
        } catch (e: CameraNotAvailableException) {
            Log.e(TAG, "Camera not available", e)
            mapper.setBlockingMessage("カメラを使用できません")
            return
        }
        if (!current.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
            mapper.setBlockingMessage("この端末は深度の取得に対応していません")
        }
        perception.uprightRotationDegrees = uprightRotationDegrees(current)
        renderer.attachSession(current)
        surfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        // 描画スレッドを止めてからセッションを停止する。
        surfaceView.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        renderer.attachSession(null)
        session?.close()
        session = null
        mapper.close()
        perception.close()
        super.onDestroy()
    }

    private fun createSession(): Session? {
        try {
            val install = ArCoreApk.getInstance().requestInstall(this, !installRequested)
            if (install == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                installRequested = true
                return null
            }
            val created = Session(this)
            created.configure(
                Config(created).apply {
                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    planeFindingMode = Config.PlaneFindingMode.DISABLED
                    lightEstimationMode = Config.LightEstimationMode.DISABLED
                    // 前方の判定に密な深度を、地図に生深度を使う。AUTOMATIC なら両方取得できる。
                    if (created.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                        depthMode = Config.DepthMode.AUTOMATIC
                    }
                },
            )
            session = created
            return created
        } catch (e: UnavailableException) {
            Log.e(TAG, "ARCore unavailable", e)
            mapper.setBlockingMessage("この端末では ARCore を利用できません")
            return null
        } catch (e: FatalException) {
            // カメラが使えない場合など。アプリを落とさず、読み上げられる形で伝える。
            Log.e(TAG, "Failed to create ARCore session", e)
            mapper.setBlockingMessage("AR を開始できませんでした")
            return null
        }
    }

    private fun uprightRotationDegrees(session: Session): Int {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val sensor = manager.getCameraCharacteristics(session.cameraConfig.cameraId)
            .get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        val display = when (currentDisplayRotation()) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (sensor - display + 360) % 360
    }

    private fun currentDisplayRotation(): Int =
        (getSystemService(Context.DISPLAY_SERVICE) as DisplayManager)
            .getDisplay(Display.DEFAULT_DISPLAY)
            .rotation

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private companion object {
        const val TAG = "MainActivity"
    }
}
