package com.example.store3d

import android.graphics.Color
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.cos
import kotlin.math.sin

object Storage {
    private fun arr(v: FloatArray) = JSONArray().apply { v.forEach { put(it.toDouble()) } }
    private fun floats(a: JSONArray) = FloatArray(a.length()) { a.getDouble(it).toFloat() }
    private fun ray(r: Ray) = JSONObject().put("o", arr(r.o)).put("d", arr(r.d))

    /** raw.json keeps every plane and ray so the model can be rebuilt later with a better reconstructor. */
    fun save(dir: File, planes: List<PlaneSnap>, labelRays: List<LabelRay>, boundaryRays: List<Ray>, labels: List<String>, boxes: List<Box>) {
        dir.mkdirs()
        File(dir, "raw.json").writeText(JSONObject()
            .put("labels", JSONArray(labels))
            .put("planes", JSONArray().apply { planes.forEach { put(JSONObject().put("vertical", it.vertical).put("poly", arr(it.poly)).put("normal", arr(it.normal))) } })
            .put("labelRays", JSONArray().apply { labelRays.forEach { put(ray(it.ray).put("probs", arr(it.probs))) } })
            .put("boundaryRays", JSONArray().apply { boundaryRays.forEach { put(ray(it)) } })
            .toString())
        File(dir, "model.json").writeText(JSONArray().apply {
            boxes.forEach { put(JSONObject().put("center", arr(it.center)).put("size", arr(it.size)).put("yaw", it.yaw.toDouble()).put("label", it.label).put("kind", it.kind.name)) }
        }.toString(1))
        writeObj(dir, boxes)
    }

    fun loadModel(file: File): List<Box> {
        val a = JSONArray(file.readText())
        return (0 until a.length()).map { a.getJSONObject(it) }.map {
            Box(floats(it.getJSONArray("center")), floats(it.getJSONArray("size")), it.getDouble("yaw").toFloat(), it.getString("label"), Kind.valueOf(it.getString("kind")))
        }
    }

    private fun writeObj(dir: File, boxes: List<Box>) {
        val labels = boxes.map { it.label }.distinct()
        File(dir, "model.mtl").writeText(labels.mapIndexed { i, l ->
            val c = colorOf(l); "# $l\nnewmtl m$i\nKd ${Color.red(c) / 255f} ${Color.green(c) / 255f} ${Color.blue(c) / 255f}\n"
        }.joinToString("\n"))
        val sb = StringBuilder("mtllib model.mtl\n")
        val faces = arrayOf(intArrayOf(0, 1, 3, 2), intArrayOf(4, 6, 7, 5), intArrayOf(0, 4, 5, 1), intArrayOf(2, 3, 7, 6), intArrayOf(0, 2, 6, 4), intArrayOf(1, 5, 7, 3))
        boxes.forEachIndexed { bi, b ->
            sb.append("o ${b.kind.name.lowercase()}_${bi}_${b.label}\nusemtl m${labels.indexOf(b.label)}\n")
            val c = cos(b.yaw); val s = sin(b.yaw)
            for (k in 0 until 8) {
                val lx = (if (k and 4 != 0) .5f else -.5f) * b.size[0]
                val ly = (if (k and 2 != 0) .5f else -.5f) * b.size[1]
                val lz = (if (k and 1 != 0) .5f else -.5f) * b.size[2]
                sb.append("v ${b.center[0] + c * lx + s * lz} ${b.center[1] + ly} ${b.center[2] - s * lx + c * lz}\n")
            }
            for (f in faces) sb.append("f ${f.joinToString(" ") { (bi * 8 + it + 1).toString() }}\n")
        }
        File(dir, "model.obj").writeText(sb.toString())
    }
}
