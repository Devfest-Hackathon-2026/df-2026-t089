package jp.devfest.shelfsight.ocr

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import java.util.concurrent.ExecutionException

/**
 * OCR が読み取った 1 行。[height]・[top]・[bottom] は画像の高さ、[left]・[right] は画像の幅を 1 とした割合。
 * 横の位置が分からない読み方（ML Kit）では、[left]・[right] は真ん中（0.5）にしておく。
 */
internal class OcrLine(
    val text: String,
    val height: Float,
    val top: Float,
    val bottom: Float = top + height,
    val left: Float = 0.5f,
    val right: Float = 0.5f,
) {
    val cx: Float get() = (left + right) / 2
    val cy: Float get() = (top + bottom) / 2
}

/** 手に取った商品を読み取った結果。 */
data class ProductResult(
    /** 商品名らしい文字を読めたときだけ true。 */
    val readable: Boolean,
    val name: String,
    val ocrText: List<String>,
)

/** 文字の読み取り。時間がかかるので GL スレッドでは呼ばない。読めなかったときは IllegalStateException を投げる。 */
internal fun interface TextOcr {
    fun read(bitmap: Bitmap): List<OcrLine>
}

/** ML Kit の日本語文字認識。モデルはアプリに同梱されるので、通信せずに動く。 */
internal class MlKitOcr : TextOcr, AutoCloseable {
    private val recognizer = TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build())

    override fun read(bitmap: Bitmap): List<OcrLine> {
        val result = try {
            Tasks.await(recognizer.process(InputImage.fromBitmap(bitmap, 0)))
        } catch (e: ExecutionException) {
            throw IllegalStateException("Text recognition failed", e)
        }
        val imageHeight = bitmap.height.toFloat()
        return result.textBlocks.flatMap { it.lines }.mapNotNull { line ->
            val box = line.boundingBox ?: return@mapNotNull null
            val symbols = line.elements.flatMap { it.symbols }.mapNotNull { it.boundingBox }
            // 行全体の枠は、箱が少し傾くだけで高くなるので、1 文字ずつの高さの中央値を使う。
            val heights = symbols.map { it.height() }.sorted()
            val height = heights.getOrNull(heights.size / 2) ?: box.height()
            OcrLine(line.text, height / imageHeight, box.top / imageHeight)
        }
    }

    override fun close() = recognizer.close()
}

/** 読み取った行の中から商品名を選ぶ。パッケージでは商品名がいちばん大きく印刷されている。 */
internal object ProductReading {
    /** これより小さい文字は、背景の棚や値札であることが多い。 */
    private const val MIN_NAME_HEIGHT = 0.012f

    /** 2 行に分けて印刷された商品名をつなげる範囲。 */
    private const val SAME_NAME_HEIGHT_RATIO = 0.8f
    private const val SAME_NAME_MAX_GAP_LINES = 2.5f

    fun from(lines: List<OcrLine>): ProductResult {
        val texts = lines.map { it.text.trim() }.filter { it.isNotEmpty() }
        val candidates = lines.filter { isNameLike(it.text) && it.height >= MIN_NAME_HEIGHT }
        val largest = candidates.maxByOrNull { it.height } ?: return ProductResult(false, "", texts)
        val name = candidates
            .filter {
                it.height >= largest.height * SAME_NAME_HEIGHT_RATIO &&
                    kotlin.math.abs(it.top - largest.top) <= largest.height * SAME_NAME_MAX_GAP_LINES
            }
            .sortedBy { it.top }
            .joinToString(" ") { it.text.trim() }
        return ProductResult(true, name, texts)
    }

    /** ロゴの 1 文字や、数字だけの行を商品名にしない。「お茶」のように短い日本語の名前は認める。 */
    private fun isNameLike(text: String): Boolean {
        val letters = text.filter { it.isLetter() }
        val japanese = letters.count { it.code > 0x2E7F }
        return japanese >= 2 || letters.length >= 3
    }
}
