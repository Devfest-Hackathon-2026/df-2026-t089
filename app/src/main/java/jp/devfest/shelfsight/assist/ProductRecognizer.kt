package jp.devfest.shelfsight.assist

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 商品認識ダッシュボード（products/）で作ったモデルで、写っている商品を見つけて名前を付ける（products/infer.py の移植）。
 *
 * 1. YOLO で「商品らしいもの」の枠を全部出す（[detect]。写っているものの判定にも使う）
 * 2. 各枠を切り抜いて埋め込み（MobileCLIP-S0 256px）を出し、照合先（命名済みの切り抜き）と比べて名前の候補を出す
 * 3. 周りの商品のジャンル・細分類・カテゴリで選び直す（[ContextAdjust]）
 *
 * モデルと照合先は tools/export_products.py で assets/products/ に書き出す。
 */
class ProductRecognizer(context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val meta = JSONObject(context.assets.open("$DIR/gallery.json").bufferedReader().readText())
    private val detector = session(context, "detector.onnx")
    private val embedder = session(context, "mobileclip_s0.onnx")
    private val detInput = detector.inputNames.first()
    private val embInput = embedder.inputNames.first()
    private val detSize = meta.getInt("detector_size")
    private val embSize = meta.getInt("embed_size")
    private val dim = meta.getInt("dim")
    private val detThreshold = meta.getDouble("det_threshold").toFloat()
    private val nmsIou = meta.optDouble("nms_iou", 0.5).toFloat()
    // (ジャンル, 細分類, カテゴリそのもの)。古い gallery.json は 2 つだけなので足りない分は 0
    private val lambdas = meta.getJSONArray("context_lambda").let { a -> DoubleArray(3) { if (it < a.length()) a.getDouble(it) else 0.0 } }

    private class Product(val key: String, val name: String, val genre: String?, val fine: String?, val start: Int)

    private val products: List<Product>
    private val gallery: FloatArray
    private val byKey: Map<String, Product>

    private class Category(val name: String, val genre: String)

    private val categories: List<Category>
    private val head: CategoryHead

    init {
        val list = meta.getJSONArray("products")
        products = (0 until list.length()).map { i ->
            val o = list.getJSONObject(i)
            Product(o.getString("key"), o.getString("name"), o.text("genre"), o.text("fine"), o.getInt("start"))
        }
        byKey = products.associateBy { it.key }
        val count = meta.getInt("count")
        val raw = context.assets.open("$DIR/gallery.bin").use { it.readBytes() }
        val half = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        gallery = FloatArray(count * dim) { android.util.Half.toFloat(half.get(it)) }

        val headMeta = JSONObject(context.assets.open("$DIR/category_head.json").bufferedReader().readText())
        val names = headMeta.getJSONArray("categories")
        categories = (0 until names.length()).map { i -> names.getJSONObject(i).let { Category(it.getString("name"), it.getString("genre")) } }
        val bytes = context.assets.open("$DIR/category_head.bin").use { it.readBytes() }
        val weights = FloatArray(bytes.size / 4).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
        head = CategoryHead(headMeta.getInt("dim"), headMeta.getInt("hidden"), categories.size, weights)
    }

    /** 商品の枠（正立画像の 0..1、[left, top, right, bottom]）と確からしさ。 */
    fun detect(image: Bitmap): List<Pair<FloatArray, Float>> {
        val scale = detSize.toFloat() / maxOf(image.width, image.height)
        val nw = Math.round(image.width * scale).coerceAtLeast(1)
        val nh = Math.round(image.height * scale).coerceAtLeast(1)
        val letter = Bitmap.createBitmap(detSize, detSize, Bitmap.Config.ARGB_8888)
        Canvas(letter).apply {
            drawColor(android.graphics.Color.rgb(114, 114, 114))
            drawBitmap(image, null, Rect(0, 0, nw, nh), Paint(Paint.FILTER_BITMAP_FLAG))
        }
        val input = chw(listOf(letter), detSize)
        letter.recycle()
        val out = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, detSize.toLong(), detSize.toLong())).use { t ->
            detector.run(mapOf(detInput to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                (r[0].value as Array<Array<FloatArray>>)[0]
            }
        }
        val n = out[0].size
        val cand = (0 until n).filter { out[4][it] >= detThreshold }
            .map { i ->
                floatArrayOf(out[0][i] - out[2][i] / 2, out[1][i] - out[3][i] / 2, out[0][i] + out[2][i] / 2, out[1][i] + out[3][i] / 2) to out[4][i]
            }
        return nms(cand, nmsIou).map { (b, s) ->
            floatArrayOf((b[0] / nw).coerceIn(0f, 1f), (b[1] / nh).coerceIn(0f, 1f), (b[2] / nw).coerceIn(0f, 1f), (b[3] / nh).coerceIn(0f, 1f)) to s
        }
    }

    /** 枠のすべてに、いちばん近い商品の名前を付ける（似ていなくても外さない）。 */
    fun recognize(image: Bitmap, boxes: List<FloatArray> = detect(image).map { it.first }): List<RecognizedProduct> {
        if (boxes.isEmpty()) return emptyList()
        val emb = boxes.chunked(EMBED_BATCH).flatMap { part ->
            val crops = part.map { crop(image, it) }
            try { embed(crops) } finally { crops.forEach { it.recycle() } }
        }

        val cands = emb.map { e -> topProducts(normalize(e)) }
        val centers = boxes.map { (it[0] + it[2]) / 2 to (it[1] + it[3]) / 2 }
        val picked = ContextAdjust.adjust(
            centers, cands,
            listOf(
                ContextAdjust.Level({ byKey[it]?.genre }, lambdas[0]),
                ContextAdjust.Level({ byKey[it]?.fine }, lambdas[1]),
                ContextAdjust.Level({ it }, lambdas[2]),  // キーがカテゴリそのもの
            ),
        )
        return boxes.indices.mapNotNull { i ->
            val p = picked[i]
            val prod = p.key?.let { byKey[it] } ?: return@mapNotNull null
            val b = boxes[i]
            RecognizedProduct(b[0], b[1], b[2], b[3], prod.name, prod.genre, prod.fine, p.sim)
        }
    }

    /** 枠の切り抜き（分類層の学習と同じく広げない）。[judge] にそのまま渡せる大きさで、解放は呼ぶ側で行う。 */
    fun cropOf(image: Bitmap, box: FloatArray): Bitmap = crop(image, box, 0f)

    /** 切り抜き 1 枚を 122 カテゴリに分類する。@return 1 位の商品と、確率の高い順の上位 5 つ（[NeighborConfidence] に渡す） */
    fun judge(crop: Bitmap, box: FloatArray): Pair<RecognizedProduct, List<Pair<String, Float>>> {
        val p = head.probabilities(normalize(embed(listOf(crop)).single()))
        val order = p.indices.sortedByDescending { p[it] }.take(TOP_CATEGORIES)
        val category = categories[order[0]]
        return RecognizedProduct(box[0], box[1], box[2], box[3], category.name, category.genre, category.name, p[order[0]]) to
            order.map { categories[it].name to p[it] }
    }

    private fun embed(crops: List<Bitmap>): List<FloatArray> =
        OnnxTensor.createTensor(env, chw(crops, embSize), longArrayOf(crops.size.toLong(), 3, embSize.toLong(), embSize.toLong())).use { t ->
            embedder.run(mapOf(embInput to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                (r[0].value as Array<FloatArray>).toList()
            }
        }

    /** 商品ごとの最大類似度の上位 [TOP_K]。 */
    private fun topProducts(e: FloatArray): List<ContextAdjust.Candidate> {
        val best = FloatArray(products.size)
        val count = gallery.size / dim
        for ((pi, p) in products.withIndex()) {
            val end = if (pi + 1 < products.size) products[pi + 1].start else count
            var m = -1f
            for (row in p.start until end) {
                var s = 0f
                val o = row * dim
                for (k in 0 until dim) s += gallery[o + k] * e[k]
                if (s > m) m = s
            }
            best[pi] = m
        }
        return best.indices.sortedByDescending { best[it] }.take(TOP_K).map { ContextAdjust.Candidate(products[it].key, best[it]) }
    }

    /** 枠を少し広げて切り抜き、正方形に引き伸ばす（照合先と同じ作り方。products/s2_crops.py）。 */
    private fun crop(image: Bitmap, b: FloatArray, pad: Float = CROP_PAD): Bitmap {
        val px = (b[2] - b[0]) * pad
        val py = (b[3] - b[1]) * pad
        val l = ((b[0] - px) * image.width).toInt().coerceIn(0, image.width - 1)
        val t = ((b[1] - py) * image.height).toInt().coerceIn(0, image.height - 1)
        val r = ((b[2] + px) * image.width).toInt().coerceIn(l + 1, image.width)
        val btm = ((b[3] + py) * image.height).toInt().coerceIn(t + 1, image.height)
        val out = Bitmap.createBitmap(embSize, embSize, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(image, Rect(l, t, r, btm), Rect(0, 0, embSize, embSize), Paint(Paint.FILTER_BITMAP_FLAG))
        return out
    }

    private fun chw(images: List<Bitmap>, size: Int): FloatBuffer {
        val buf = ByteBuffer.allocateDirect(images.size * 3 * size * size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val px = IntArray(size * size)
        val plane = size * size
        for ((n, img) in images.withIndex()) {
            img.getPixels(px, 0, size, 0, 0, size, size)
            val base = n * 3 * plane
            for (i in 0 until plane) {
                val p = px[i]
                buf.put(base + i, ((p shr 16) and 0xff) / 255f)
                buf.put(base + plane + i, ((p shr 8) and 0xff) / 255f)
                buf.put(base + 2 * plane + i, (p and 0xff) / 255f)
            }
        }
        buf.rewind()
        return buf
    }

    private fun session(context: Context, name: String): OrtSession = OrtSession.SessionOptions().use { o ->
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        o.setIntraOpNumThreads(4)
        env.createSession(context.assets.open("$DIR/$name").use { it.readBytes() }, o)
    }

    override fun close() {
        detector.close()
        embedder.close()
    }

    companion object {
        private const val DIR = "products"
        private const val TOP_K = 10
        /** 枠は 1 枚で 100 個を超えることがある。256px を一度に埋め込むとメモリを食うので分ける。 */
        private const val EMBED_BATCH = 8
        private const val CROP_PAD = 0.04f
        /**
         * 地図に置く確からしさ（[NeighborConfidence] で調整したもの）。検証写真で、調整前の確率 0.7 以上は枠の 52% を残して 87% が正解、
         * 調整後 0.7 以上は 53% を残して 91% が正解だった（Gemini の確定ラベルとの一致。products/sku26n）。
         */
        const val KNOWN_PROB = 0.7f
        private const val TOP_CATEGORIES = 5

        /** 照合先やカテゴリを作り直すと変わる値。事前計算の結果が今のモデルのものかを見分ける。 */
        fun version(context: Context): String = context.assets.open("$DIR/gallery.json").bufferedReader().use { it.readText() }.hashCode().toString()

        private fun JSONObject.text(name: String): String? = if (isNull(name)) null else optString(name).ifEmpty { null }

        private fun normalize(v: FloatArray): FloatArray {
            var s = 0f
            for (x in v) s += x * x
            val n = kotlin.math.sqrt(s).coerceAtLeast(1e-6f)
            return FloatArray(v.size) { v[it] / n }
        }

        fun iou(a: FloatArray, b: FloatArray): Float {
            val w = (minOf(a[2], b[2]) - maxOf(a[0], b[0])).coerceAtLeast(0f)
            val h = (minOf(a[3], b[3]) - maxOf(a[1], b[1])).coerceAtLeast(0f)
            val inter = w * h
            val union = SceneJudge.area(a) + SceneJudge.area(b) - inter
            return if (union > 0f) inter / union else 0f
        }

        fun nms(boxes: List<Pair<FloatArray, Float>>, threshold: Float): List<Pair<FloatArray, Float>> {
            val out = mutableListOf<Pair<FloatArray, Float>>()
            for (c in boxes.sortedByDescending { it.second }) if (out.none { iou(it.first, c.first) > threshold }) out += c
            return out
        }
    }
}
