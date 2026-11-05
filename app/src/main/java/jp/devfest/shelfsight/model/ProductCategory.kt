package jp.devfest.shelfsight.model

/**
 * コンビニの棚を大まかに分けるカテゴリ。
 *
 * [id] は Gemma への指示と応答の両方で使う識別子。
 * 色は白文字とのコントラストが十分に取れる濃さにしている。
 */
enum class ProductCategory(val id: String, val labelJa: String, val argb: Int) {
    COFFEE("coffee", "コーヒー", 0xFF5D4037.toInt()),
    TEA("tea", "お茶", 0xFF2E7D32.toInt()),
    SOFT_DRINK("soft_drink", "ソフトドリンク", 0xFF0277BD.toInt()),
    ALCOHOL("alcohol", "お酒", 0xFF6A1B9A.toInt()),
    RICE_MEAL("rice_meal", "おにぎり・弁当", 0xFFBF360C.toInt()),
    BREAD("bread", "パン", 0xFF795548.toInt()),
    SNACK("snack", "お菓子", 0xFFC2185B.toInt()),
    INSTANT_NOODLE("instant_noodle", "カップ麺", 0xFFC62828.toInt()),
    DAIRY("dairy", "乳製品", 0xFF00695C.toInt()),
    FROZEN("frozen", "冷凍食品", 0xFF1565C0.toInt()),
    DAILY_GOODS("daily_goods", "日用品", 0xFF455A64.toInt()),
    OTHER("other", "その他", 0xFF424242.toInt());

    companion object {
        private val byId = entries.associateBy { it.id }

        /** モデルが一覧にない値を返した場合は [OTHER] として扱う。 */
        fun fromId(id: String): ProductCategory = byId[id.trim().lowercase()] ?: OTHER
    }
}
