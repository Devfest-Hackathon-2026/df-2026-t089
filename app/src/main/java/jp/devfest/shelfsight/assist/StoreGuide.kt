package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/** 店のモデルから見つけた、近くの棚の面 1 つ。 */
class NearShelf(val main: String, val at: Vec3, val distance: Float, val azimuthDeg: Float, val facing: Boolean)

/** 商品名を調べた記録。[at] は調べたときにいた場所。 */
class LookedUp(val name: String, val at: Vec3, val time: Long)

/** タップで話す文と、質問に答える AI に渡す文脈を、店のモデルと今の位置から作る。 */
object StoreGuide {
    /** ここまでの棚を話す（商品の位置が分かる距離 [ShelfMap.MAX_DEPTH_M] と同じ）。 */
    const val RANGE_M = ShelfMap.MAX_DEPTH_M
    /** 棚に「向かっている」: この距離以内で、棚の面と向き合う角度（面の法線との角度が [FACING_DEG] 以内）。 */
    const val FACING_M = 1.8f
    const val FACING_DEG = 40f
    private const val MAX_SHELVES = 4
    /** 棚の商品として見る範囲（正面からの角度と距離）。 */
    private const val FRONT_DEG = 35f
    private const val FRONT_M = 2.2f
    /** モデルがまだないときに、商品だけで棚に向かっていると言える範囲（通路を歩いているだけの横の棚を含めない）。 */
    private const val ITEMS_ONLY_DEG = 20f
    private const val ITEMS_ONLY_M = 1.5f
    private const val KINDS_PER_BAND = 3
    /** 床からの高さで分ける（棚は 1.8 m ほど）。 */
    private const val UPPER_M = 1.25f
    private const val LOWER_M = 0.75f

    /** 右が正の方位角（度）を、話す方向にする。 */
    fun direction(azimuthDeg: Float): String {
        val a = abs(azimuthDeg)
        val side = if (azimuthDeg > 0) "右" else "左"
        return when {
            a <= 30f -> "前"
            a <= 60f -> "${side}前"
            a <= 120f -> side
            a <= 150f -> "${side}後ろ"
            else -> "後ろ"
        }
    }

    /** 近い順の、こちらを向いている棚の面。主な商品が分かっている面だけ。 */
    fun nearShelves(view: StoreView, l: Listener): List<NearShelf> {
        val cs = cos(view.yaw).toFloat()
        val sn = sin(view.yaw).toFloat()
        val u = l.position.x * cs + l.position.z * sn
        val v = -l.position.x * sn + l.position.z * cs
        val out = ArrayList<NearShelf>()
        for (f in view.fixtures) for ((sign, modules) in f.faces) {
            val s = if (sign == "+") 1f else -1f
            val face = (f.c + s * f.d / 2).toFloat()
            val cross = if (f.axis == 0) u else v
            val along = if (f.axis == 0) v else u
            if ((cross - face) * s <= 0f) continue   // 棚の裏側にいる
            val centerOf = { i: Int -> (f.a + (i + 0.5) * f.w).toFloat() }
            val i = modules.indices.filter { modules[it].isNotEmpty() }.minByOrNull { abs(centerOf(it) - along) } ?: continue
            val pu = if (f.axis == 0) face else centerOf(i)
            val pv = if (f.axis == 0) centerOf(i) else face
            val at = Vec3(pu * cs - pv * sn, l.position.y, pu * sn + pv * cs)
            val distance = hypot(at.x - l.position.x, at.z - l.position.z)
            if (distance > RANGE_M) continue
            val nu = if (f.axis == 0) s else 0f
            val nv = if (f.axis == 0) 0f else s
            val nx = nu * cs - nv * sn
            val nz = nu * sn + nv * cs
            val facing = distance <= FACING_M && -(nx * l.forwardX + nz * l.forwardZ) >= cos(Math.toRadians(FACING_DEG.toDouble())).toFloat()
            val az = Math.toDegrees(Binaural.azimuth(l, at).toDouble()).toFloat()
            out += NearShelf(modules[i][0].first, at, distance, az, facing)
        }
        return out.sortedBy { it.distance }
    }

    /** 方向ごとにいちばん近い棚を、話す分だけ。 */
    fun toSpeak(shelves: List<NearShelf>): List<NearShelf> =
        shelves.distinctBy { direction(it.azimuthDeg) }.take(MAX_SHELVES)

    /** 「右にはカップ麺の棚、」「前にはお菓子の棚があります。」のように、1 棚ごとの文。ポンの音を挟むので分ける。 */
    fun clauses(shelves: List<NearShelf>): List<String> = shelves.mapIndexed { i, s ->
        "${direction(s.azimuthDeg)}には${ShelfDescriber.spoken(s.main)}の棚" + if (i == shelves.lastIndex) "があります。" else "、"
    }

    private fun aimedAt(item: MapItem, l: Listener, maxDeg: Float, maxM: Float): Boolean {
        val d = hypot(item.position.x - l.position.x, item.position.z - l.position.z)
        return d <= maxM && abs(Math.toDegrees(Binaural.azimuth(l, item.position).toDouble())) <= maxDeg
    }

    /** 向かっている棚の商品。 */
    fun inFront(items: List<MapItem>, l: Listener): List<MapItem> = items.filter { aimedAt(it, l, FRONT_DEG, FRONT_M) }

    /** モデルがまだないときに、棚に向かっていると言える商品。 */
    fun squarelyInFront(items: List<MapItem>, l: Listener): List<MapItem> = items.filter { aimedAt(it, l, ITEMS_ONLY_DEG, ITEMS_ONLY_M) }

    /** 「カップ麺の棚で、上の方にはA、真ん中にはBやC、下の方にはDがあります。」商品がなければ null。 */
    fun facingSentence(main: String?, items: List<MapItem>, floorY: Float): String? {
        val bands = listOf(
            "上の方" to items.filter { it.position.y - floorY >= UPPER_M },
            "真ん中" to items.filter { it.position.y - floorY >= LOWER_M && it.position.y - floorY < UPPER_M },
            "下の方" to items.filter { it.position.y - floorY < LOWER_M },
        ).mapNotNull { (band, list) ->
            val kinds = list.groupBy { ShelfDescriber.spoken(it.name) }.entries
                .sortedByDescending { e -> e.value.sumOf { it.seen } }.take(KINDS_PER_BAND).map { it.key }
            if (kinds.isEmpty()) null else "${band}には${kinds.joinToString("や")}"
        }
        val top = main ?: items.groupBy { it.name }.maxByOrNull { e -> e.value.sumOf { it.seen } }?.key ?: return null
        val shelf = "${ShelfDescriber.spoken(top)}の棚"
        return if (bands.isEmpty()) "${shelf}です。" else "${shelf}で、${bands.joinToString("、")}があります。"
    }

    private fun where(at: Vec3, l: Listener): String {
        val az = Math.toDegrees(Binaural.azimuth(l, at).toDouble()).toFloat()
        val d = hypot(at.x - l.position.x, at.z - l.position.z)
        return "${direction(az)} ${"%.1f".format(java.util.Locale.ROOT, d)}m"
    }

    private val KIND_NAMES = mapOf("island" to "島の棚", "wall" to "壁ぎわの棚", "case" to "冷蔵・冷凍ケース", "end" to "島の端の棚", "register" to "レジ")

    /** 質問に答える AI に渡す、今分かっていること。方向と距離は今の位置と向きから見たもの。 */
    fun context(view: StoreView?, l: Listener?, items: List<MapItem>, looked: List<LookedUp>, now: Long, lastSpoken: String): String = buildString {
        if (l == null) {
            appendLine("- 位置はまだ分かりません。")
        } else {
            appendLine("- 方向と距離は、すべて利用者の今の位置と向きから見たものです。")
            val cs = view?.let { cos(it.yaw).toFloat() } ?: 1f
            val sn = view?.let { sin(it.yaw).toFloat() } ?: 0f
            view?.room?.let { appendLine("- 店の広さ: 約${"%.0f".format(java.util.Locale.ROOT, it[2] - it[0])}m × ${"%.0f".format(java.util.Locale.ROOT, it[3] - it[1])}m（歩いた範囲から推定）") }
            val shelves = view?.fixtures.orEmpty().map { f ->
                val cu = if (f.axis == 0) f.c else (f.a + f.b) / 2
                val cv = if (f.axis == 0) (f.a + f.b) / 2 else f.c
                val at = Vec3((cu * cs - cv * sn).toFloat(), l.position.y, (cu * sn + cv * cs).toFloat())
                val goods = f.faces.values.flatten().flatten().groupBy { it.first }.entries
                    .sortedByDescending { e -> e.value.sumOf { it.second } }.take(4).joinToString("、") { it.key }
                Triple(hypot(at.x - l.position.x, at.z - l.position.z), at, "${KIND_NAMES[f.kind] ?: f.kind}" + if (goods.isEmpty()) "" else ": $goods")
            }.sortedBy { it.first }
            if (shelves.isNotEmpty()) {
                appendLine("- 店の地図（近い順）:")
                for ((_, at, text) in shelves.take(MAX_MAP_LINES)) appendLine("  - ${where(at, l)}: $text")
            }
            if (items.isNotEmpty()) {
                appendLine("- 見つけた商品の種類（いちばん近いものの場所、近い順）:")
                val nearest = items.groupBy { it.name }.map { (name, list) -> name to list.minBy { hypot(it.position.x - l.position.x, it.position.z - l.position.z) } }
                    .sortedBy { (_, m) -> hypot(m.position.x - l.position.x, m.position.z - l.position.z) }
                for ((name, m) in nearest.take(MAX_ITEM_LINES)) appendLine("  - $name: ${where(m.position, l)}")
            }
            if (looked.isNotEmpty()) {
                appendLine("- 利用者が商品名を調べた商品（新しい順、場所は調べたときにいた所）:")
                for (x in looked.asReversed().take(MAX_LOOKED_LINES)) appendLine("  - ${x.name}: ${where(x.at, l)}、${(now - x.time) / 60000} 分前")
            }
        }
        if (lastSpoken.isNotEmpty()) appendLine("- 最後に利用者へ話したこと: $lastSpoken")
    }

    private const val MAX_MAP_LINES = 30
    private const val MAX_ITEM_LINES = 60
    private const val MAX_LOOKED_LINES = 10
}
