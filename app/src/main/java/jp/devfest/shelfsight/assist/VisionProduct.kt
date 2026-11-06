package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.ocr.CloudVisionParser
import jp.devfest.shelfsight.ocr.ProductReading
import org.json.JSONArray
import org.json.JSONObject

/** 調べた商品。社名が分からないときは空。 */
data class ProductAnswer(val maker: String, val product: String) {
    fun spoken(): String = when {
        maker.isNotEmpty() && product.isNotEmpty() -> "${maker}の、${product}です。"
        product.isNotEmpty() -> "${product}です。"
        else -> "${maker}の商品です。"
    }
}

/**
 * Cloud Vision API の応答（WEB_DETECTION・LOGO_DETECTION・TEXT_DETECTION を 1 回で頼んだもの）から、社名と商品名を決める。
 * Web 検索（逆画像検索）の判定は products/s3_name.py の judge を移したもの。Web で分からなければ、ロゴと印刷の文字で決める。
 */
object VisionProductParser {
    fun request(jpegBase64: String): String = JSONObject().put(
        "requests",
        JSONArray().put(
            JSONObject()
                .put("image", JSONObject().put("content", jpegBase64))
                .put(
                    "features",
                    JSONArray()
                        .put(JSONObject().put("type", "WEB_DETECTION").put("maxResults", 10))
                        .put(JSONObject().put("type", "LOGO_DETECTION").put("maxResults", 3))
                        .put(JSONObject().put("type", "DOCUMENT_TEXT_DETECTION")),
                )
                .put(
                    "imageContext",
                    JSONObject().put("languageHints", JSONArray().put("ja"))
                        .put("webDetectionParams", JSONObject().put("includeGeoResults", false)),
                ),
        ),
    ).toString()

    /** 分からなければ null。[imageHeight] は送った画像の高さ（文字の大きさを比べるため）。 */
    fun parse(json: String, imageHeight: Int): ProductAnswer? {
        val r = JSONObject(json).getJSONArray("responses").getJSONObject(0)
        r.optJSONObject("error")?.let { error("Cloud Vision error: ${it.optString("message")}") }

        val web = r.optJSONObject("webDetection") ?: JSONObject()
        val best = web.optJSONArray("bestGuessLabels").objects().map { it.optString("label") }.firstOrNull { it.isNotBlank() }.orEmpty()
        val entities = web.optJSONArray("webEntities").objects()
            .filter { it.optString("description").isNotBlank() }
            .map { it.optString("description") to it.optDouble("score", 0.0) }.toList()
        val pages = web.optJSONArray("pagesWithMatchingImages").objects().map { it.optString("pageTitle") }.toList()
        val matches = web.optJSONArray("fullMatchingImages").count() + web.optJSONArray("partialMatchingImages").count()
        val specific = entities.filter { (d, s) -> d.lowercase() !in GENERIC && s >= 0.5 }.map { it.first }
        val webName = if (best.isNotEmpty() && best.lowercase() !in GENERIC) best else specific.firstOrNull().orEmpty()
        val webGood = webName.isNotEmpty() && (matches > 0 || pages.isNotEmpty() || specific.isNotEmpty())

        val logos = r.optJSONArray("logoAnnotations").objects().map { it.optString("description") }.toList()
        val text = (listOf(best) + entities.map { it.first } + pages + logos).joinToString(" ").lowercase()
        val maker = makerIn(best.lowercase()) ?: makerIn(text) ?: logos.firstOrNull { it.isNotBlank() }.orEmpty()

        val printed = runCatching { ProductReading.from(CloudVisionParser.lines(json, imageHeight)) }.getOrNull()
        val product = when {
            webGood -> withoutMaker(webName, maker)
            printed?.readable == true -> withoutMaker(printed.name, maker)
            else -> ""
        }
        if (product.isEmpty() && maker.isEmpty()) return null
        return ProductAnswer(maker, product)
    }

    fun makerIn(text: String): String? = MAKERS.entries.firstOrNull { (_, aliases) -> aliases.any { it in text } }?.key

    /** 商品名から社名の部分を取り除く。 */
    fun withoutMaker(name: String, maker: String): String {
        var out = name
        for (a in MAKERS[maker].orEmpty() + maker) if (a.isNotEmpty()) out = out.replace(a, "", ignoreCase = true)
        out = out.trim(' ', '-', '・', '　')
        return out.ifEmpty { name.trim() }
    }

    private fun JSONArray?.objects(): Sequence<JSONObject> =
        if (this == null) emptySequence() else (0 until length()).asSequence().mapNotNull { optJSONObject(it) }

    private fun JSONArray?.count(): Int = this?.length() ?: 0

    /** 商品名として意味のない一般語（best guess がこれだけなら Web の結果は使わない）。 */
    private val GENERIC = setOf(
        "product", "products", "packaging and labeling", "packaging", "label", "brand", "bottle", "plastic bottle",
        "glass bottle", "convenience store", "store", "shelf", "retail", "supermarket", "cosmetics", "snack", "food",
        "drink", "box", "carton", "font", "logo", "advertising", "plastic", "paper", "personal care", "skin care",
        "beauty", "health", "liquid", "water", "juice", "tea", "beer", "sock", "socks", "textile", "material property",
        "rectangle", "japan", "japanese", "graphics", "design", "pattern", "illustration", "art", "image", "photograph",
        "medicine", "pharmaceutical drug", "household supply", "hygiene", "lipstick", "toothbrush", "cosmetic",
        "makeup", "comb", "hair care", "electric blue", "magenta", "pink", "red", "white", "black", "transparency",
        "convenience food", "flavor", "ingredient", "fast food", "candy", "confectionery", "pharmacy", "drugstore",
    )

    /** 社名の辞書（表記ゆれ → 正式名）。products/s3_name.py と同じ。 */
    val MAKERS: Map<String, List<String>> = linkedMapOf(
        "花王" to listOf("花王", "kao"), "ライオン" to listOf("ライオン", "lion corporation"), "資生堂" to listOf("資生堂", "shiseido"),
        "P&G" to listOf("p&g", "procter", "ピーアンドジー"), "ユニリーバ" to listOf("ユニリーバ", "unilever"),
        "ユニ・チャーム" to listOf("ユニ・チャーム", "ユニチャーム", "unicharm"), "小林製薬" to listOf("小林製薬", "kobayashi"),
        "ロート製薬" to listOf("ロート製薬", "rohto"), "大正製薬" to listOf("大正製薬", "taisho"),
        "サントリー" to listOf("サントリー", "suntory"), "アサヒ" to listOf("アサヒ", "asahi"), "キリン" to listOf("キリン", "kirin"),
        "サッポロ" to listOf("サッポロ", "sapporo"), "伊藤園" to listOf("伊藤園", "ito en"),
        "コカ・コーラ" to listOf("コカ・コーラ", "coca-cola", "coca cola"), "ダイドー" to listOf("ダイドー", "dydo"),
        "明治" to listOf("明治", "meiji"), "森永" to listOf("森永", "morinaga"), "ロッテ" to listOf("ロッテ", "lotte"),
        "グリコ" to listOf("グリコ", "glico"), "カルビー" to listOf("カルビー", "calbee"), "湖池屋" to listOf("湖池屋", "koikeya"),
        "日清" to listOf("日清", "nissin"), "東洋水産" to listOf("東洋水産", "マルちゃん", "maruchan"),
        "ブルボン" to listOf("ブルボン", "bourbon"), "不二家" to listOf("不二家", "fujiya"),
        "ハウス食品" to listOf("ハウス食品", "house foods"), "味の素" to listOf("味の素", "ajinomoto"),
        "キッコーマン" to listOf("キッコーマン", "kikkoman"), "ヤクルト" to listOf("ヤクルト", "yakult"),
        "雪印メグミルク" to listOf("雪印", "megmilk"), "ファミリーマート" to listOf("ファミリーマート", "familymart"),
        "セブン-イレブン" to listOf("セブン-イレブン", "セブンイレブン", "7-eleven", "seven-eleven"), "ローソン" to listOf("ローソン", "lawson"),
        "ジレット" to listOf("gillette", "ジレット"),
        "ジョンソン・エンド・ジョンソン" to listOf("johnson & johnson", "ジョンソン・エンド・ジョンソン", "band-aid"),
        "アース製薬" to listOf("アース製薬", "earth corporation"), "エステー" to listOf("エステー", "st corporation"),
        "コーセー" to listOf("コーセー", "kose"), "マンダム" to listOf("マンダム", "mandom"), "ピジョン" to listOf("ピジョン", "pigeon"),
        "レッドブル" to listOf("red bull", "レッドブル"), "モンスター" to listOf("monster energy", "モンスターエナジー"),
        "ネスレ" to listOf("ネスレ", "nestle", "nestlé"), "有楽製菓" to listOf("有楽製菓", "yuraku", "ブラックサンダー"),
    )
}
