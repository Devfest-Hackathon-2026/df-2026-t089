package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CropQueueTest {
    private fun entry(name: String, x: Float, area: Float = 0f, time: Long = 0L) =
        CropQueue.Entry(name, FloatArray(4), Vec3(x, 0f, 0f), Vec3(0f, 0f, 0f), 0.1f, 0.1f, area, time)

    @Test fun `nearest crop comes out first and old crops lose priority`() {
        val q = CropQueue<String>(10)
        q.add(entry("far", 4f, time = 10_000L))
        q.add(entry("near", 1f, time = 10_000L))
        assertEquals("near", q.take(Vec3(0f, 0f, 0f), 10_000L)!!.crop)
        assertEquals("far", q.take(Vec3(0f, 0f, 0f), 10_000L)!!.crop)
        assertNull(q.take(null, 0L))
        q.add(entry("stale", 1f, time = 0L))
        q.add(entry("fresh", 2f, time = 20_000L))
        assertEquals("fresh", q.take(Vec3(0f, 0f, 0f), 20_000L)!!.crop)
    }

    @Test fun `the same spot keeps only the crop seen largest`() {
        val q = CropQueue<String>(10)
        assertTrue(q.add(entry("small", 1f, area = 0.01f)).isEmpty())
        assertEquals(listOf("small"), q.add(entry("big", 1.05f, area = 0.2f)))
        assertEquals(listOf("tiny"), q.add(entry("tiny", 1.02f, area = 0.001f)))
        assertEquals(1, q.size)
        assertEquals("big", q.take(null, 0L)!!.crop)
    }

    @Test fun `over capacity the oldest crop is dropped`() {
        val q = CropQueue<String>(2)
        q.add(entry("a", 1f, time = 1L))
        q.add(entry("b", 2f, time = 2L))
        assertEquals(listOf("a"), q.add(entry("c", 3f, time = 3L)))
    }

    @Test fun `a judged spot is touched instead of judged again`() {
        val map = ShelfMap()
        assertEquals(false, map.touch(Vec3(1f, 0f, 0f), 0.1f, 0.1f, 1L, null))
        map.add("tea", null, null, Vec3(1f, 0f, 0f), 0.1f, 0.1f, 1L)
        assertEquals(true, map.touch(Vec3(1.05f, 0f, 0f), 0.1f, 0.1f, 2L, null))
        assertEquals(2, map.snapshot().single().seen)
    }
}
