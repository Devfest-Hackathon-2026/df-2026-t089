package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.perception.OwnHandFilter

/**
 * 人として検出された枠のうち、自分の手や腕（商品を持った手など）らしいものを外す。手の検出は使わず、
 * 枠の位置・形と深度だけで決める。胸の高さのカメラで見た人は、近ければ画面の上の方まで写り、
 * 全身が写るなら縦に長いことを使う。
 */
internal object PersonFilter {
    /** 全身が写った人の枠の縦横比（ピクセルで、高さ / 幅）の下限。手や持った商品はこれより横に広い。 */
    const val MIN_STANDING_ASPECT = 1.4f
    /** これより近い物は、手を伸ばして届く範囲。 */
    const val NEAR_M = 1.2f
    /** 近くの人なら、頭か胴が画面の上の方まで写る。 */
    const val NEAR_MAX_TOP = 0.25f
    const val NEAR_MIN_HEIGHT = 0.55f
    /** 写った高さから出した距離に比べて、深度がこれより小さく近いなら、人ではなく手前の小さな物。 */
    const val DEPTH_TO_HEIGHT_RATIO = 0.5f
    private const val EDGE = 0.02f

    /**
     * @param box 正立画像の枠
     * @param aspect 枠の高さ / 幅（ピクセル）
     * @param depthM 枠の中央の深度（なければ null）
     */
    fun isOwnHand(box: NormalizedBox, aspect: Float, depthM: Float?): Boolean {
        // 画面の下端から立ち上がり、上半分に届かない枠（元からある判定）
        if (OwnHandFilter.isLowerOnly(box)) return true
        val height = box.bottom - box.top
        // 手の届く距離にあるのに、画面の上の方まで写っていない
        if (depthM != null && depthM < NEAR_M && (box.top > NEAR_MAX_TOP || height < NEAR_MIN_HEIGHT)) return true
        val fullBody = box.top > EDGE && box.bottom < 1f - EDGE
        if (fullBody) {
            // 全身が写っているなら縦に長いはず
            if (aspect < MIN_STANDING_ASPECT) return true
            // 写った高さからは遠くの人に見えるのに、深度では手前にある（顔の前にかざした手など）
            if (depthM != null && depthM < NEAR_M * 2 && depthM < heightDistanceRatio(height) * DEPTH_TO_HEIGHT_RATIO) return true
        }
        return false
    }

    /** 枠の高さが画面の [height] のとき、身長 1.65m の人までの距離のおおよその値（縦の画角 65° くらいのカメラ）。 */
    private fun heightDistanceRatio(height: Float): Float = 1.65f * 0.78f / height.coerceAtLeast(0.05f)
}
