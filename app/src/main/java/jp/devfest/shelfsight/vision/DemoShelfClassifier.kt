package jp.devfest.shelfsight.vision

import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.ProductCategory
import jp.devfest.shelfsight.model.ShelfDetection
import kotlinx.coroutines.delay

/**
 * モデルなしで AR 側の動作を確認するための固定結果。画像の中身は見ていない。
 * デバッグビルド専用で、使用中は画面に必ずデモモードと表示する。
 */
class DemoShelfClassifier : ShelfClassifier {
    override suspend fun prepare() = Unit

    override suspend fun classify(jpeg: ByteArray): List<ShelfDetection> {
        delay(500)
        return listOf(
            ShelfDetection(ProductCategory.COFFEE, NormalizedBox(0.1f, 0.15f, 0.45f, 0.45f)),
            ShelfDetection(ProductCategory.ALCOHOL, NormalizedBox(0.55f, 0.55f, 0.9f, 0.85f)),
        )
    }

    override fun close() = Unit
}
