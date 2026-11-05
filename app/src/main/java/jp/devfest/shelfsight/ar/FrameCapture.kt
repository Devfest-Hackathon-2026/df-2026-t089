package jp.devfest.shelfsight.ar

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.google.ar.core.Camera
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.ResourceExhaustedException
import java.io.ByteArrayOutputStream
import java.nio.ByteOrder
import kotlin.math.max

/** GL スレッドから取り出した撮影データ。ARCore の Image や Frame への参照は持たない。 */
class CapturedFrame(
    val nv21: ByteArray,
    val width: Int,
    val height: Int,
    val geometry: CaptureGeometry,
)

internal object FrameCapture {
    /** GL スレッドで呼ぶ。カメラ画像がまだ用意できていなければ null を返す。 */
    fun capture(frame: Frame, camera: Camera, uprightRotationDegrees: Int, depthEnabled: Boolean): CapturedFrame? {
        val image = try {
            frame.acquireCameraImage()
        } catch (_: NotYetAvailableException) {
            return null
        } catch (_: ResourceExhaustedException) {
            return null
        }
        return image.use {
            val width = it.width
            val height = it.height
            val nv21 = toNv21(it)

            val intrinsics = camera.imageIntrinsics
            val focal = intrinsics.focalLength
            val principal = intrinsics.principalPoint
            val cameraToWorld = FloatArray(16).also { m -> camera.pose.toMatrix(m, 0) }

            val corners = floatArrayOf(0f, 0f, width.toFloat(), height.toFloat())
            val texture = FloatArray(4)
            frame.transformCoordinates2d(
                Coordinates2d.IMAGE_PIXELS, corners,
                Coordinates2d.TEXTURE_NORMALIZED, texture,
            )

            CapturedFrame(
                nv21, width, height,
                CaptureGeometry(
                    imageWidth = width,
                    imageHeight = height,
                    focalX = focal[0],
                    focalY = focal[1],
                    principalX = principal[0],
                    principalY = principal[1],
                    cameraToWorld = cameraToWorld,
                    uprightRotationDegrees = uprightRotationDegrees,
                    depth = if (depthEnabled) captureDepth(frame) else null,
                    depthU0 = texture[0],
                    depthUPerPixel = (texture[2] - texture[0]) / width,
                    depthV0 = texture[1],
                    depthVPerPixel = (texture[3] - texture[1]) / height,
                ),
            )
        }
    }

    /** 深度は端末を動かし始めてしばらくは取得できないため、その間は null を返す。 */
    private fun captureDepth(frame: Frame): DepthSnapshot? {
        val image = try {
            frame.acquireDepthImage16Bits()
        } catch (_: NotYetAvailableException) {
            return null
        }
        return image.use {
            val plane = it.planes[0]
            val shorts = plane.buffer.order(ByteOrder.nativeOrder()).asShortBuffer()
            val rowStride = plane.rowStride / Short.SIZE_BYTES
            val data = ShortArray(it.width * it.height)
            for (row in 0 until it.height) {
                shorts.position(row * rowStride)
                shorts.get(data, row * it.width, it.width)
            }
            DepthSnapshot(it.width, it.height, data)
        }
    }

    private fun toNv21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val nv21 = ByteArray(width * height * 3 / 2)

        val yPlane = image.planes[0]
        val yBuffer = yPlane.buffer
        for (row in 0 until height) {
            yBuffer.position(row * yPlane.rowStride)
            yBuffer.get(nv21, row * width, width)
        }

        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        var out = width * height
        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                nv21[out++] = vBuffer.get(row * vPlane.rowStride + col * vPlane.pixelStride)
                nv21[out++] = uBuffer.get(row * uPlane.rowStride + col * uPlane.pixelStride)
            }
        }
        return nv21
    }
}

internal object JpegEncoder {
    private const val MAX_LONG_SIDE = 1024
    private const val QUALITY = 85

    /** 重い処理なので GL スレッドでは呼ばない。 */
    fun toUprightJpeg(frame: CapturedFrame): ByteArray {
        val raw = ByteArrayOutputStream().use { out ->
            YuvImage(frame.nv21, ImageFormat.NV21, frame.width, frame.height, null)
                .compressToJpeg(Rect(0, 0, frame.width, frame.height), QUALITY, out)
            out.toByteArray()
        }
        val decoded = BitmapFactory.decodeByteArray(raw, 0, raw.size)
        val scale = minOf(1f, MAX_LONG_SIDE.toFloat() / max(frame.width, frame.height))
        val matrix = Matrix().apply {
            postScale(scale, scale)
            postRotate(frame.geometry.uprightRotationDegrees.toFloat())
        }
        val upright = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
        return ByteArrayOutputStream().use { out ->
            upright.compress(Bitmap.CompressFormat.JPEG, QUALITY, out)
            if (upright !== decoded) upright.recycle()
            decoded.recycle()
            out.toByteArray()
        }
    }
}
