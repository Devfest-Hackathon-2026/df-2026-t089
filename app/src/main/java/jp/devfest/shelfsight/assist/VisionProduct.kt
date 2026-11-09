package jp.devfest.shelfsight.assist

import org.json.JSONArray
import org.json.JSONObject

/**
 * 調べた商品。分かれば商品名を先に言い、推定したカテゴリを続ける。社名は言わない（iOS 版の ProductAnswer と同じ）。
 *
 * @param product 商品名（分からなければ空）
 * @param category categories の中の 1 つ（分からなければ null）
 * @param kind 声で言う種類（炭酸水、ビール、グミなど）
 * @param sure false なら「〜のようです」と控えめに言う
 * @param maybeAlcohol カテゴリはお酒でないが、お酒の手がかり（「ビール」の文字、年齢確認の掲示など）もあった
 * @param source 決めたところ（「端末」「Gemini」など。画面とログ用）
 * @param detail 端末で数えた手がかり（画面とログ用）
 */
data class ProductAnswer(
    val product: String,
    val category: String?,
    val kind: String?,
    val sure: Boolean,
    val alcohol: Boolean,
    val maybeAlcohol: Boolean,
    val source: String,
    val detail: String,
) {
    /** 例:「黒ラベル。お酒、ビールです。」「ウィルキンソン。炭酸水です。」「グミのようです。」 */
    fun spoken(): String = buildString {
        if (product.isNotEmpty()) append("$product。")
        if (kind != null) {
            val k = if (alcohol) "お酒、$kind" else kind
            append(if (sure) "${k}です。" else "${k}のようです。")
        } else {
            append("種類は分かりませんでした。")
        }
        if (maybeAlcohol) append("お酒かもしれません。")
    }

    fun describe(): String =
        "種類「${kind ?: "-"}」${if (sure) "" else "（推測）"} 商品「$product」${if (alcohol) " お酒" else ""}${if (maybeAlcohol) " お酒かも" else ""}（$source、$detail）"
}

/**
 * Cloud Vision API に、印字の文字（DOCUMENT_TEXT_DETECTION）だけを頼む。
 * Web 検索（逆画像検索）とロゴは使わない（手に持った商品の写真ではネットの画像と一致せず、ビールを「Sparkling Water」と答えたこともあった）。
 */
object VisionProductParser {
    fun request(jpegBase64: String): String = JSONObject().put(
        "requests",
        JSONArray().put(
            JSONObject()
                .put("image", JSONObject().put("content", jpegBase64))
                .put("features", JSONArray().put(JSONObject().put("type", "DOCUMENT_TEXT_DETECTION")))
                .put("imageContext", JSONObject().put("languageHints", JSONArray().put("ja"))),
        ),
    ).toString()
}
