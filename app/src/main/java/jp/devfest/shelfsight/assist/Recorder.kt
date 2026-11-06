package jp.devfest.shelfsight.assist

import android.graphics.Bitmap
import android.util.Log
import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.model.Vec3
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 1 秒ごとの撮影を記録する。起動ごとにフォルダを作り、画像（frame_<番号>.jpg）と、
 * 1 行 1 枚の records.jsonl（時刻、ARCore のカメラの姿勢、向き、カメラの内部パラメータ、認識した商品と地図上の位置）を書く。
 *   adb pull /sdcard/Android/data/jp.devfest.shelfsight/files/records
 */
class Recorder(root: File) {
    val dir = File(root, SimpleDateFormat("yyyyMMdd_HHmmss", Locale.JAPAN).format(Date())).apply { mkdirs() }
    private val log = File(dir, "records.jsonl")
    private var n = 0

    /** 商品 1 つの記録。[position] は地図に置いた位置（置けなければ null）。 */
    class Product(val p: RecognizedProduct, val position: Vec3?)

    fun record(time: Long, image: Bitmap, g: CaptureGeometry, headingDeg: Float?, walls: Int, products: List<Product>) {
        val name = "frame_%05d.jpg".format(n++)
        try {
            File(dir, name).outputStream().use { image.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, it) }
            val o = JSONObject()
                .put("t", time)
                .put("image", name)
                .put("width", image.width).put("height", image.height)
                // 列優先の 4x4（カメラ → 世界）。カメラ座標は +X が右、+Y が上、-Z が前（センサー画像の向き）
                .put("camera_to_world", JSONArray(g.cameraToWorld.map { it.toDouble() }))
                .put("heading_deg", headingDeg?.toDouble() ?: JSONObject.NULL)
                .put("intrinsics", JSONObject()
                    .put("fx", g.focalX.toDouble()).put("fy", g.focalY.toDouble())
                    .put("cx", g.principalX.toDouble()).put("cy", g.principalY.toDouble())
                    .put("sensor_width", g.imageWidth).put("sensor_height", g.imageHeight)
                    .put("upright_rotation", g.uprightRotationDegrees))
                .put("walls", walls)
                .put("products", JSONArray(products.map { pr ->
                    val p = pr.p
                    JSONObject().put("name", p.name).put("genre", p.genre ?: JSONObject.NULL).put("fine", p.fine ?: JSONObject.NULL)
                        .put("sim", p.sim.toDouble())
                        .put("box", JSONArray(listOf(p.left, p.top, p.right, p.bottom).map { it.toDouble() }))
                        .put("position", pr.position?.let { v -> JSONArray(listOf(v.x, v.y, v.z).map { it.toDouble() }) } ?: JSONObject.NULL)
                }))
            log.appendText(o.toString() + "\n")
        } catch (e: java.io.IOException) {
            Log.w("Recorder", "could not record $name", e)   // 記録できなくても案内は続ける
        }
    }

    private companion object {
        const val JPEG_QUALITY = 80
    }
}
