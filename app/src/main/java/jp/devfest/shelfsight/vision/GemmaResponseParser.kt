package jp.devfest.shelfsight.vision

import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.ProductCategory
import jp.devfest.shelfsight.model.ShelfDetection
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/** Gemma の応答文字列を検出結果に変換する。 */
internal object GemmaResponseParser {
    private const val BOX_SCALE = 1000f

    /**
     * コードフェンスや前後の文章が混ざっていても JSON 配列の部分を取り出す。
     * 配列自体が壊れている場合は空を返し、不正な要素は個別に読み飛ばす。
     */
    fun parse(response: String): List<ShelfDetection> {
        val start = response.indexOf('[')
        val end = response.lastIndexOf(']')
        if (start < 0 || end <= start) return emptyList()

        val array = try {
            JSONArray(response.substring(start, end + 1))
        } catch (_: JSONException) {
            return emptyList()
        }
        return (0 until array.length()).mapNotNull { index ->
            (array.opt(index) as? JSONObject)?.let(::parseItem)
        }
    }

    private fun parseItem(item: JSONObject): ShelfDetection? {
        val categoryId = item.optString("category").takeIf { it.isNotBlank() } ?: return null
        val box = item.optJSONArray("box_2d")?.let(::parseBox) ?: return null
        return ShelfDetection(ProductCategory.fromId(categoryId), box)
    }

    private fun parseBox(values: JSONArray): NormalizedBox? {
        if (values.length() != 4) return null
        val coords = FloatArray(4)
        for (i in 0 until 4) {
            val value = values.optDouble(i)
            if (value.isNaN()) return null
            coords[i] = (value.toFloat() / BOX_SCALE).coerceIn(0f, 1f)
        }
        val (top, left, bottom, right) = coords
        if (bottom <= top || right <= left) return null
        return NormalizedBox(left = left, top = top, right = right, bottom = bottom)
    }
}
