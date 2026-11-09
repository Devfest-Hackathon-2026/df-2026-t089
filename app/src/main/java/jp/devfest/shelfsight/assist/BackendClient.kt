package jp.devfest.shelfsight.assist

import android.util.Log
import jp.devfest.shelfsight.cloud.HttpTransport
import jp.devfest.shelfsight.cloud.UrlConnectionTransport
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException

/** 店のモデルのバックエンド（backend/app.py）。通信するのでメインスレッドで呼ばない。失敗したら null を返す。 */
internal class BackendClient(
    private val baseUrl: String,
    private val apiKey: String,
    private val transport: HttpTransport = UrlConnectionTransport,
    private val warn: (String) -> Unit = { Log.w(TAG, it) },
) {
    class Solved(val view: StoreView, val model: JSONObject)
    class SavedMap(val name: String, val view: StoreView?)

    val configured: Boolean get() = baseUrl.isNotEmpty() && apiKey.isNotEmpty()

    /** @param body [StoreModel.requestBody] の JSON */
    fun solve(body: String): Solved? = post("/v1/solve", body)?.let { r ->
        val model = r.getJSONObject("model")
        Solved(parseView(model), model)
    }

    fun save(lat: Double, lng: Double, model: JSONObject): Boolean =
        post("/v1/maps", location(lat, lng).put("model", model).toString())?.optBoolean("saved") == true

    fun lookup(lat: Double, lng: Double): SavedMap? = post("/v1/maps/lookup", location(lat, lng).toString())?.let { r ->
        SavedMap(r.getString("name"), if (r.isNull("model")) null else parseView(r.getJSONObject("model")))
    }

    private fun location(lat: Double, lng: Double) = JSONObject().put("lat", lat).put("lng", lng)

    private fun post(path: String, body: String): JSONObject? = try {
        val res = transport.post(baseUrl.trimEnd('/') + path, mapOf("X-Api-Key" to apiKey, "Content-Type" to "application/json"), body)
        if (res.code in 200..299) JSONObject(res.body) else {
            warn("$path failed: HTTP ${res.code}")
            null
        }
    } catch (e: IOException) {
        warn("$path unreachable: $e")
        null
    } catch (e: JSONException) {
        warn("$path bad response: $e")
        null
    }

    companion object {
        private const val TAG = "Backend"

        fun parseView(m: JSONObject): StoreView {
            val room = if (m.isNull("room")) null else m.getJSONArray("room").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
            val list = m.getJSONArray("fixtures")
            return StoreView(m.getDouble("yaw"), room, (0 until list.length()).map { parseFixture(list.getJSONObject(it)) })
        }

        private fun parseFixture(f: JSONObject): StoreFixture {
            val n = f.getInt("n")
            val w = f.getDouble("w")
            val widths = f.optJSONArray("widths")?.takeIf { it.length() == n }
            return StoreFixture(
                f.getString("kind"), f.getInt("axis"), f.getDouble("c"), f.getDouble("a"), f.getDouble("b"), f.getDouble("d"), f.getDouble("h"), n, w,
                parseSides(f.optJSONObject("faces")) { modules ->
                    (0 until modules.length()).map { i ->
                        val cats = modules.getJSONArray(i)
                        (0 until cats.length()).map { k -> cats.getJSONArray(k).let { it.getString(0) to it.getDouble(1) } }
                    }
                },
                parseSides(f.optJSONObject("themes")) { names -> (0 until names.length()).map { if (names.isNull(it)) null else names.getString(it) } },
                if (widths != null) List(n) { widths.getDouble(it) } else List(n) { w },
            )
        }

        private fun <T> parseSides(sides: JSONObject?, parse: (JSONArray) -> T): Map<String, T> =
            sides?.keys()?.asSequence()?.associateWith { parse(sides.getJSONArray(it)) }.orEmpty()
    }
}
