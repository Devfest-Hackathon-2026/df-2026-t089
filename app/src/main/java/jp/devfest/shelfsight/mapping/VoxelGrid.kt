package jp.devfest.shelfsight.mapping

import kotlin.math.floor

/**
 * 3D 空間を一定サイズの立方体に区切り、観測された位置を記録する。
 *
 * 店舗全体では数百万回の書き込みが起きるため、HashMap を使わず配列で実装している。
 * 誤った深度を地図に残さないよう、[MIN_OBSERVATIONS] 回観測された位置だけを確定点として扱う。
 */
internal class VoxelGrid(private val voxelSize: Float = DEFAULT_VOXEL_SIZE_M) {
    private var capacity = INITIAL_CAPACITY
    private var keys = LongArray(capacity) { EMPTY }
    private var colors = IntArray(capacity)
    private var counts = ByteArray(capacity)

    /** 観測されたボクセルの数。 */
    var size = 0
        private set

    /** 確定点の数。 */
    var confirmedCount = 0
        private set

    /** @param abgr 頂点データにそのまま使える 0xAABBGGRR 形式の色 */
    fun add(x: Float, y: Float, z: Float, abgr: Int) {
        if (size * 10 >= capacity * 7) grow()
        val key = keyOf(x, y, z)
        val slot = findSlot(key, keys, capacity)
        if (keys[slot] == EMPTY) {
            keys[slot] = key
            colors[slot] = abgr
            counts[slot] = 1
            size++
            return
        }
        colors[slot] = blend(colors[slot], abgr)
        val count = counts[slot].toInt()
        if (count < Byte.MAX_VALUE) {
            counts[slot] = (count + 1).toByte()
            if (count + 1 == MIN_OBSERVATIONS) confirmedCount++
        }
    }

    /** 確定点を x, y, z（Float のビット列）, 色の順に 4 要素ずつ並べて返す。 */
    fun toVertices(): IntArray {
        val out = IntArray(confirmedCount * INTS_PER_VERTEX)
        var i = 0
        for (slot in 0 until capacity) {
            val key = keys[slot]
            if (key == EMPTY || counts[slot] < MIN_OBSERVATIONS) continue
            out[i++] = (centerOf(key ushr 42)).toRawBits()
            out[i++] = (centerOf((key ushr 21) and AXIS_MASK)).toRawBits()
            out[i++] = (centerOf(key and AXIS_MASK)).toRawBits()
            out[i++] = colors[slot]
        }
        return out
    }

    private fun keyOf(x: Float, y: Float, z: Float): Long =
        (indexOf(x) shl 42) or (indexOf(y) shl 21) or indexOf(z)

    private fun indexOf(value: Float): Long =
        floor(value / voxelSize).toLong() + AXIS_OFFSET

    private fun centerOf(packed: Long): Float =
        (packed - AXIS_OFFSET + 0.5f) * voxelSize

    private fun findSlot(key: Long, table: LongArray, tableCapacity: Int): Int {
        val mask = tableCapacity - 1
        var slot = hash(key) and mask
        while (table[slot] != EMPTY && table[slot] != key) slot = (slot + 1) and mask
        return slot
    }

    private fun hash(key: Long): Int {
        val mixed = key * -7046029254386353131L // 黄金比に由来する定数でビットを混ぜる
        return (mixed xor (mixed ushr 32)).toInt()
    }

    private fun grow() {
        val newCapacity = capacity * 2
        val newKeys = LongArray(newCapacity) { EMPTY }
        val newColors = IntArray(newCapacity)
        val newCounts = ByteArray(newCapacity)
        for (slot in 0 until capacity) {
            val key = keys[slot]
            if (key == EMPTY) continue
            val target = findSlot(key, newKeys, newCapacity)
            newKeys[target] = key
            newColors[target] = colors[slot]
            newCounts[target] = counts[slot]
        }
        capacity = newCapacity
        keys = newKeys
        colors = newColors
        counts = newCounts
    }

    /** 古い色 3 に新しい色 1 の割合で混ぜる。各チャンネルを桁あふれさせずに計算する。 */
    private fun blend(old: Int, new: Int): Int {
        val rb = ((old and 0xFF00FF) * 3 + (new and 0xFF00FF)) ushr 2 and 0xFF00FF
        val g = ((old and 0x00FF00) * 3 + (new and 0x00FF00)) ushr 2 and 0x00FF00
        return OPAQUE or rb or g
    }

    companion object {
        const val DEFAULT_VOXEL_SIZE_M = 0.05f
        const val INTS_PER_VERTEX = 4
        const val MIN_OBSERVATIONS = 2

        private const val INITIAL_CAPACITY = 1 shl 12
        private const val EMPTY = -1L
        private const val AXIS_OFFSET = 1L shl 20
        private const val AXIS_MASK = (1L shl 21) - 1
        private const val OPAQUE = 0xFF shl 24
    }
}
