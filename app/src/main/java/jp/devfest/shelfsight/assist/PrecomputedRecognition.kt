package jp.devfest.shelfsight.assist

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * 録画のキーフレームを前もって商品認識した結果。再生中は計算せずにこの結果を引く。
 *
 * shortcut: 認識が O(N) で再生に間に合わない間の代用。学習中のモデルで認識が定数時間になったら消す。
 */
internal class PrecomputedRecognition(private val entries: List<Entry>) {
    class Entry(val timestampNs: Long, val products: List<RecognizedProduct>) {
        /** 今の枠に、重なりがいちばん大きい事前計算の名前を付ける。重なる結果がない枠には名前を付けない。 */
        fun match(boxes: List<FloatArray>): List<RecognizedProduct> = boxes.mapNotNull { b ->
            val (best, overlap) = products.map { it to ProductRecognizer.iou(b, floatArrayOf(it.left, it.top, it.right, it.bottom)) }
                .maxByOrNull { it.second } ?: return@mapNotNull null
            if (overlap < MIN_IOU) null else best.copy(left = b[0], top = b[1], right = b[2], bottom = b[3])
        }
    }

    private var next = 0

    fun rewind() { next = 0 }

    /** 再生がこの時刻まで来て、新しく使えるようになった結果（なければ null）。GL スレッドから呼ぶ。 */
    fun takeDue(timestampNs: Long): Entry? {
        var due: Entry? = null
        while (next < entries.size && entries[next].timestampNs <= timestampNs) due = entries[next++]
        return due
    }

    private class Keyframe(val timestampNs: Long, val image: File)

    companion object {
        private const val FILE = "recognition.jsonl"
        private const val MIN_IOU = 0.3f
        private const val INTERVAL_NS = 1_000_000_000L   // 再生中の認識と同じ 1 秒おき

        /** 認識するキーフレーム。ARCore が追跡できていた枚だけを 1 秒おきに選ぶ。 */
        private fun keyframes(dir: File): List<Keyframe> {
            val all = File(dir, "keyframes.jsonl").readLines().mapNotNull { line ->
                try {
                    JSONObject(line).takeIf { it.getString("state") == "TRACKING" }?.let { Keyframe(it.getLong("t"), File(dir, it.getString("image"))) }
                } catch (_: org.json.JSONException) {
                    null   // 記録が途中で切れた最後の行
                }
            }.sortedBy { it.timestampNs }
            val picked = ArrayList<Keyframe>()
            for (k in all) if (picked.isEmpty() || k.timestampNs - picked.last().timestampNs >= INTERVAL_NS) picked += k
            return picked
        }

        /** 保存済みの結果。[version] が違うファイルは読まない（途中で切れた最後の行は捨てる）。 */
        private fun read(dir: File, version: String): Map<Long, Entry> {
            val lines = File(dir, FILE).takeIf { it.isFile }?.readLines().orEmpty()
            if (lines.firstOrNull() != version) return emptyMap()
            return lines.drop(1).mapNotNull { line ->
                try { parse(JSONObject(line)) } catch (_: org.json.JSONException) { null }
            }.associateBy { it.timestampNs }
        }

        private fun parse(o: JSONObject): Entry {
            val a = o.getJSONArray("p")
            return Entry(o.getLong("t"), (0 until a.length()).map { i ->
                val p = a.getJSONArray(i)
                RecognizedProduct(
                    p.getDouble(0).toFloat(), p.getDouble(1).toFloat(), p.getDouble(2).toFloat(), p.getDouble(3).toFloat(), p.getString(4),
                    if (p.isNull(5)) null else p.getString(5), if (p.isNull(6)) null else p.getString(6), p.getDouble(7).toFloat(),
                )
            })
        }

        private fun line(timestampNs: Long, products: List<RecognizedProduct>): String = JSONObject().put("t", timestampNs).put("p", JSONArray(products.map { p ->
            JSONArray(listOf(p.left.toDouble(), p.top.toDouble(), p.right.toDouble(), p.bottom.toDouble(), p.name, p.genre ?: JSONObject.NULL, p.fine ?: JSONObject.NULL, p.sim.toDouble()))
        })).toString()

        /** すべてのキーフレームの結果がそろっていれば返す。 */
        fun load(dir: File, version: String): PrecomputedRecognition? {
            val saved = read(dir, version)
            val all = try { keyframes(dir) } catch (_: java.io.IOException) { return null }
            if (all.isEmpty() || all.any { it.timestampNs !in saved }) return null
            return PrecomputedRecognition(all.map { saved.getValue(it.timestampNs) })
        }

        /** 足りないキーフレームだけ認識して 1 枚ごとに保存する（中断しても続きから）。最後まで終わったら true。 */
        fun compute(dir: File, version: String, recognize: (Bitmap) -> List<RecognizedProduct>, cancelled: () -> Boolean, progress: (Int, Int) -> Unit): Boolean {
            val all = keyframes(dir)
            val saved = read(dir, version)
            val file = File(dir, FILE)
            if (saved.isEmpty()) file.writeText(version + "\n")   // モデルが変わった古い結果は捨てる
            var done = all.count { it.timestampNs in saved }
            progress(done, all.size)
            for (k in all) {
                if (k.timestampNs in saved) continue
                if (cancelled() || Thread.currentThread().isInterrupted) return false
                val image = BitmapFactory.decodeFile(k.image.path) ?: throw java.io.IOException("cannot read ${k.image}")
                try {
                    file.appendText(line(k.timestampNs, recognize(image)) + "\n")
                } finally {
                    image.recycle()
                }
                progress(++done, all.size)
            }
            return all.isNotEmpty()
        }
    }
}
