package jp.devfest.shelfsight.perception

import org.junit.Assert.assertEquals
import org.junit.Test

class YoloPersonDetectorTest {
    /** Columns are detections; rows are cx, cy, w, h, person score. */
    private fun output(vararg rows: FloatArray) = Array(5) { c -> FloatArray(rows.size) { rows[it][c] } }

    @Test fun `keeps confident people, merges overlaps and normalizes by the image size`() {
        val out = output(
            floatArrayOf(100f, 200f, 40f, 160f, 0.9f),
            floatArrayOf(102f, 202f, 40f, 160f, 0.6f),
            floatArrayOf(300f, 200f, 40f, 160f, 0.3f),
        )
        val boxes = YoloPersonDetector.decode(out, 234, 416)
        assertEquals(1, boxes.size)
        assertEquals(80f / 234, boxes[0].left, 1e-4f)
        assertEquals(120f / 416, boxes[0].top, 1e-4f)
        assertEquals(120f / 234, boxes[0].right, 1e-4f)
        assertEquals(280f / 416, boxes[0].bottom, 1e-4f)
    }

    @Test fun `boxes past the image edge are clamped`() {
        val boxes = YoloPersonDetector.decode(output(floatArrayOf(10f, 400f, 40f, 80f, 0.9f)), 234, 416)
        assertEquals(0f, boxes[0].left, 1e-4f)
        assertEquals(1f, boxes[0].bottom, 1e-4f)
    }
}
