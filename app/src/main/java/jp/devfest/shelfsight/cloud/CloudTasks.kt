package jp.devfest.shelfsight.cloud

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/** クラウドへの 1 種類の依頼。文面とスキーマは assets/cloud にあり、PC のテスト（tools/cloud_test.py）と共有する。 */
internal class CloudTask(val prompt: String, val schemaJson: String, val imageResolution: String)

internal class CloudTasks(readAsset: (String) -> String, val categories: List<String>) {
    val shelfColumns = build(readAsset, "shelf_columns")

    private fun build(readAsset: (String) -> String, name: String): CloudTask {
        val list = categories.joinToString("\n") { "- $it" }
        val enum = JSONArray(categories).toString()
        return CloudTask(
            prompt = readAsset("cloud/$name.prompt.txt").replace(CATEGORIES_LIST, list),
            schemaJson = readAsset("cloud/$name.schema.json").replace(CATEGORIES_ENUM, enum),
            imageResolution = IMAGE_RESOLUTION,
        )
    }

    private companion object {
        const val CATEGORIES_LIST = "{{CATEGORIES}}"
        const val CATEGORIES_ENUM = "\"{{CATEGORIES_ENUM}}\""

        /** 値札や商品名の小さな文字を読むため、高い解像度を指定する。 */
        const val IMAGE_RESOLUTION = "high"
    }
}

/** モデルの応答を検証して結果に変換する。スキーマを指定していても、値の範囲までは信用しない。 */
internal class CloudResponseParser(private val categories: List<String>) {
    private val fallbackCategory = "その他"

    fun parseShelfColumns(json: String): ShelfColumnsResult {
        val root = parseObject(json)
        val isShelf = root.optBoolean("is_shelf_front", false)
        if (!isShelf) return ShelfColumnsResult(false, emptyList())
        val items = root.optJSONArray("columns") ?: JSONArray()
        val columns = (0 until items.length())
            .mapNotNull { items.optJSONObject(it)?.let(::parseColumn) }
            .sortedBy { it.xStart }
        return ShelfColumnsResult(true, removeOverlaps(columns))
    }

    private fun parseColumn(item: JSONObject): ShelfColumn? {
        val start = item.optDouble("x_start")
        val end = item.optDouble("x_end")
        if (start.isNaN() || end.isNaN()) return null
        val xStart = (start.toFloat() / X_SCALE).coerceIn(0f, 1f)
        val xEnd = (end.toFloat() / X_SCALE).coerceIn(0f, 1f)
        if (xEnd - xStart < MIN_COLUMN_WIDTH) return null
        return ShelfColumn(
            xStart, xEnd, normalizeCategory(item.optString("category")),
            parseConfidence(item.optString("confidence")), item.optString("evidence").trim(),
        )
    }

    /** 重なった列は、左の列の右端を次の列の左端に合わせて、同じ場所が 2 列にならないようにする。 */
    private fun removeOverlaps(columns: List<ShelfColumn>): List<ShelfColumn> =
        columns.mapIndexedNotNull { i, column ->
            val next = columns.getOrNull(i + 1)
            val end = if (next != null) minOf(column.xEnd, next.xStart) else column.xEnd
            if (end - column.xStart < MIN_COLUMN_WIDTH) null else column.copy(xEnd = end)
        }

    private fun normalizeCategory(value: String): String =
        value.trim().takeIf { it in categories } ?: fallbackCategory

    private fun parseConfidence(value: String): CloudConfidence = when (value.trim().lowercase()) {
        "high" -> CloudConfidence.HIGH
        "medium" -> CloudConfidence.MEDIUM
        else -> CloudConfidence.LOW
    }

    private fun parseObject(json: String): JSONObject = try {
        JSONObject(json)
    } catch (e: JSONException) {
        throw CloudException(CloudFailure.BAD_RESPONSE, "output is not a JSON object", e)
    }

    private companion object {
        const val X_SCALE = 1000f
        const val MIN_COLUMN_WIDTH = 0.02f
    }
}
