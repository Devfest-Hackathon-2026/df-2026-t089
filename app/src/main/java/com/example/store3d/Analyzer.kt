package com.example.store3d

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.Image
import com.google.ar.core.Pose
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/** Upright (portrait) normalized coords -> CPU-image pixel coords, for an image that must be rotated `rot`° clockwise to be upright. */
fun uprightToSensor(xu: Float, yu: Float, rot: Int, w: Int, h: Int): FloatArray = when (rot) {
    90 -> floatArrayOf(yu * w, (1 - xu) * h)
    180 -> floatArrayOf((1 - xu) * w, (1 - yu) * h)
    270 -> floatArrayOf((1 - yu) * w, xu * h)
    else -> floatArrayOf(xu * w, yu * h)
}

/** Same decoding as tools/train.py decode_peaks: local maxima above thr, refined to sub-column. */
fun decodePeaks(p: FloatArray, thr: Float): List<Float> {
    val out = mutableListOf<Float>()
    for (i in p.indices) {
        val l = if (i > 0) p[i - 1] else 0f
        val r = if (i < p.size - 1) p[i + 1] else 0f
        if (p[i] >= thr && p[i] >= l && p[i] > r) {
            val d = l - 2 * p[i] + r
            val off = if (d < 0) 0.5f * (l - r) / d else 0f
            out += (i + 0.5f + off) / p.size
        }
    }
    return out
}

fun yuvToArgb(img: Image): IntArray {
    val w = img.width; val h = img.height
    val yP = img.planes[0]; val uP = img.planes[1]; val vP = img.planes[2]
    val yB = yP.buffer; val uB = uP.buffer; val vB = vP.buffer
    val out = IntArray(w * h)
    for (j in 0 until h) for (i in 0 until w) {
        val y = yB.get(j * yP.rowStride + i).toInt() and 0xff
        val uv = (j / 2) * uP.rowStride + (i / 2) * uP.pixelStride
        val u = (uB.get(uv).toInt() and 0xff) - 128
        val v = (vB.get(uv).toInt() and 0xff) - 128
        val r = (y + 1.402f * v).toInt().coerceIn(0, 255)
        val g = (y - 0.344f * u - 0.714f * v).toInt().coerceIn(0, 255)
        val b = (y + 1.772f * u).toInt().coerceIn(0, 255)
        out[j * w + i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
    }
    return out
}

/** Camera frame snapshot taken on the GL thread so the ARCore Image can be closed immediately. */
class FrameShot(val argb: IntArray, val w: Int, val h: Int, val rot: Int, val pose: Pose, val fx: Float, val fy: Float, val cx: Float, val cy: Float)

class Analyzer(ctx: Context) {
    val labels: List<String> = ctx.assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }
    private val opts = Interpreter.Options().setNumThreads(4)
    private val cls = Interpreter(map(ctx, "shelf_classifier.tflite"), opts)
    private val bnd = Interpreter(map(ctx, "shelf_boundary.tflite"), opts)

    class Result(val labelRays: List<LabelRay>, val boundaryRays: List<Ray>, val summary: String)

    fun analyze(f: FrameShot): Result {
        val src = Bitmap.createBitmap(f.argb, f.w, f.h, Bitmap.Config.ARGB_8888)
        val up = Bitmap.createBitmap(src, 0, 0, f.w, f.h, Matrix().apply { postRotate(f.rot.toFloat()) }, true)
        val bOut = Array(1) { FloatArray(BND_COLS) }
        bnd.run(input(up, BND_SIZE), bOut)
        val cuts = decodePeaks(bOut[0], BOUNDARY_THR)
        val edges = listOf(0f) + cuts + listOf(1f)
        val labelRays = mutableListOf<LabelRay>()
        val names = mutableListOf<String>()
        for (i in 0 until edges.size - 1) {
            val a = edges[i]; val b = edges[i + 1]
            if (b - a < 0.12f) continue
            val crop = Bitmap.createBitmap(up, (a * up.width).toInt(), (0.15f * up.height).toInt(),
                ((b - a) * up.width).toInt().coerceAtLeast(1), (0.7f * up.height).toInt())
            val out = Array(1) { FloatArray(labels.size) }
            cls.run(input(crop, CLS_SIZE), out)
            labelRays += LabelRay(ray(f, (a + b) / 2, 0.5f), out[0])
            names += labels[out[0].indices.maxBy { out[0][it] }]
        }
        return Result(labelRays, cuts.map { ray(f, it, 0.5f) }, names.joinToString(" | "))
    }

    private fun ray(f: FrameShot, xu: Float, yu: Float): Ray {
        val s = uprightToSensor(xu, yu, f.rot, f.w, f.h)
        // ARCore camera pose (getPose) is OpenGL-style, aligned with the CPU image readout
        val d = floatArrayOf((s[0] - f.cx) / f.fx, -(s[1] - f.cy) / f.fy, -1f)
        val l = sqrt(d[0] * d[0] + d[1] * d[1] + 1f)
        return Ray(f.pose.translation, f.pose.rotateVector(FloatArray(3) { d[it] / l }))
    }

    private fun input(bmp: Bitmap, size: Int): ByteBuffer {
        val s = Bitmap.createScaledBitmap(bmp, size, size, true)
        val px = IntArray(size * size).also { s.getPixels(it, 0, size, 0, 0, size, size) }
        val buf = ByteBuffer.allocateDirect(size * size * 12).order(ByteOrder.nativeOrder())
        for (p in px) { buf.putFloat(((p shr 16) and 0xff).toFloat()); buf.putFloat(((p shr 8) and 0xff).toFloat()); buf.putFloat((p and 0xff).toFloat()) }
        return buf.rewind() as ByteBuffer
    }

    private fun map(ctx: Context, name: String) = ctx.assets.openFd(name).use { fd ->
        FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    companion object {
        const val CLS_SIZE = 224
        const val BND_SIZE = 384
        const val BND_COLS = 24
        const val BOUNDARY_THR = 0.5f
    }
}
