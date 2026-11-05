package jp.devfest.shelfsight.vision

import jp.devfest.shelfsight.model.ShelfDetection

/** 棚の画像からカテゴリと位置を判別する。 */
interface ShelfClassifier : AutoCloseable {
    /** モデルの読み込みなど、時間のかかる準備を行う。 */
    suspend fun prepare()

    /** @param jpeg 正立させた JPEG 画像 */
    suspend fun classify(jpeg: ByteArray): List<ShelfDetection>
}
