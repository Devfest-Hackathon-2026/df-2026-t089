package jp.devfest.shelfsight.assist

/** 棚の種類の列（画面を横に [columns] 等分）ごとに、写っている商品をまとめる。 */
object ShelfColumns {
    /** 列ごとの商品名（多く見えた順、最大 [MAX_PER_COLUMN] 個）。商品の中心の横位置で列を決める。 */
    fun productsByColumn(products: List<RecognizedProduct>, columns: Int, maxPerColumn: Int = MAX_PER_COLUMN): List<List<String>> {
        if (columns <= 0) return emptyList()
        val byColumn = products.groupBy { ((it.cx * columns).toInt()).coerceIn(0, columns - 1) }
        return List(columns) { c ->
            byColumn[c].orEmpty().groupingBy { it.name }.eachCount().entries
                .sortedByDescending { it.value }.take(maxPerColumn).map { it.key }
        }
    }

    const val MAX_PER_COLUMN = 3
}
