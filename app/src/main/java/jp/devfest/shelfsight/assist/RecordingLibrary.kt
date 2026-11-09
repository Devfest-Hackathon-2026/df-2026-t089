package jp.devfest.shelfsight.assist

import org.json.JSONObject
import java.io.File

/** App updates keep these recordings in the app-specific external files directory. */
internal class RecordingLibrary(private val root: File) {
    data class Recording(val name: String, val directory: File) {
        val dataset: File get() = File(directory, "session.mp4")
        val events: File get() = File(directory, "events.jsonl")
        val label: String get() = "${name.substring(4, 6)}/${name.substring(6, 8)} ${name.substring(9, 11)}:${name.substring(11, 13)}  (${dataset.length() / 1_000_000} MB)"
    }

    data class Event(val timestampNs: Long, val type: String, val text: String?)

    fun list(): List<Recording> = root.listFiles().orEmpty()
        .filter { it.isDirectory && NAME.matches(it.name) && File(it, "session.mp4").length() > 0L }
        .sortedByDescending { it.name }
        .map { Recording(it.name, it) }

    fun find(name: String?): Recording? = list().firstOrNull { it.name == name }

    fun events(recording: Recording): List<Event> {
        if (!recording.events.isFile) return emptyList()
        return recording.events.useLines { lines ->
            lines.mapNotNull { line ->
                try {
                    val o = JSONObject(line)
                    Event(o.getLong("t"), o.getString("type"), o.optString("text").takeIf { o.has("text") })
                } catch (_: Exception) {
                    null
                }
            }.toList()
        }.sortedBy { it.timestampNs }
    }

    companion object {
        private val NAME = Regex("\\d{8}_\\d{6}")
    }
}

/** Delivers each recorded action once, at the corresponding ARCore frame timestamp. */
internal class PlaybackActions(private val events: List<RecordingLibrary.Event>) {
    private var next = 0

    fun through(timestampNs: Long): List<RecordingLibrary.Event> {
        if (timestampNs <= 0L) return emptyList()
        val due = ArrayList<RecordingLibrary.Event>()
        while (next < events.size && events[next].timestampNs <= timestampNs) {
            due += events[next++]
        }
        return due
    }
}
