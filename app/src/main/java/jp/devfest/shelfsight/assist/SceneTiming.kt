package jp.devfest.shelfsight.assist

import android.os.SystemClock

/**
 * 商品の解析 1 回ごとの時間を貯め、直近 [window] 回の平均・最小・最大を出す（商品だけを計るモードの表示用）。
 * 1 回 = 撮る（GL スレッド）→ 正立画像にする → 枠を出す（YOLO）→ 名前を付ける（埋め込みと照合）。
 */
class SceneTiming(private val window: Int = 30, private val clock: () -> Long = SystemClock::elapsedRealtime) {
    private class Run(val capture: Long, val convert: Long, val detect: Long, val name: Long, val boxes: Int, val named: Int, val at: Long) {
        val total get() = capture + convert + detect + name
    }

    private val runs = ArrayDeque<Run>()
    private var count = 0

    @Synchronized
    fun add(captureMs: Long, convertMs: Long, detectMs: Long, nameMs: Long, boxes: Int, named: Int) {
        runs.addLast(Run(captureMs, convertMs, detectMs, nameMs, boxes, named, clock()))
        while (runs.size > window) runs.removeFirst()
        count++
    }

    @Synchronized
    fun lastLine(): String {
        val r = runs.lastOrNull() ?: return "-"
        return "#$count total ${r.total}ms (capture ${r.capture}, convert ${r.convert}, detect ${r.detect}, name ${r.name}), " +
            "boxes ${r.boxes}, named ${r.named}"
    }

    @Synchronized
    fun summary(): String {
        if (runs.isEmpty()) return "計測中…"
        fun stat(label: String, f: (Run) -> Long): String {
            val v = runs.map(f)
            return "$label 平均 ${v.average().toLong()} / 最小 ${v.min()} / 最大 ${v.max()}ms"
        }
        // 1 回あたりの間隔（解析が終わってから次が終わるまで）。前の解析が終わるとすぐ次を回すので、これが実際の周期
        val period = if (runs.size >= 2) (runs.last().at - runs.first().at) / (runs.size - 1) else null
        return listOf(
            "直近 ${runs.size} 回（通算 $count 回）  周期 ${period?.let { "${it}ms（%.1f 回/秒）".format(1000f / it) } ?: "-"}",
            stat("合計", Run::total),
            stat("枠", Run::detect) + "  " + stat("名前", Run::name),
            stat("撮る", Run::capture) + "  " + stat("変換", Run::convert),
            "枠 平均 %.1f 個 / 名前 平均 %.1f 個".format(runs.map { it.boxes }.average(), runs.map { it.named }.average()),
        ).joinToString("\n")
    }
}
