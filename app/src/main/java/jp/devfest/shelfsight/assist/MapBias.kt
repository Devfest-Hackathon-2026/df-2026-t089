package jp.devfest.shelfsight.assist

import kotlin.math.ln
import kotlin.math.min

/**
 * 棚の種類（7 分類）で、商品の名前の候補を選び直す。その棚の分類に合わない候補を少し下げる
 * （飲み物の棚でチョコと読み違えた、など）。棚の種類は、地図に貯めたその場所のラベル（[ShelfLabelMap]）か、
 * なければこの 1 枚の歩行中の判別の列から。
 *
 * 選び直しにだけ使い、名前を付けるかどうか（照合先にない商品か）は補正前の類似度で決める。
 */
object MapBias {
    /** 分類の割合で選び直す強さ（[ContextAdjust] の周りの商品による補正と同じくらい）。 */
    const val LAMBDA = 0.03f
    private const val EPS = 0.05f
    /** この票数で補正を全部かける（地図の票が少ない、または 1 枚の判別だけなら弱める）。 */
    const val FULL_VOTES = 5

    /** 補正前の類似度はそのままにして、選び直した順に並べた候補を返す。 */
    fun adjust(cands: List<ContextAdjust.Candidate>, shelf: ShelfPrior, groupOf: (String) -> String?): List<ContextAdjust.Candidate> {
        if (shelf.groups.isEmpty()) return cands
        val trust = min(1f, shelf.votes / FULL_VOTES.toFloat())
        return cands.map { c ->
            val g = groupOf(c.key) ?: return@map c
            ContextAdjust.Candidate(c.key, c.sim + trust * LAMBDA * ln(((shelf.groups[g] ?: 0f) + EPS) / (1f + EPS)))
        }.sortedByDescending { it.sim }
    }
}
