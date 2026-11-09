package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.ShelfChange
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** 説明した棚から、別の棚に移ったかの判定（iOS 版の ShelfChange と同じ）。 */
class ShelfChangeTest {
    private val tea = setOf("緑茶", "紅茶")

    @Test
    fun `nothing described yet is never another shelf`() {
        val c = ShelfChange()
        assertFalse(c.update(setOf("グミ"), null, 0))
        assertFalse(c.update(setOf("グミ"), null, 5000))
    }

    @Test
    fun `different kinds count as another shelf only after they last a second`() {
        val c = ShelfChange()
        assertFalse(c.update(setOf("グミ", "チョコレート"), tea, 1000))
        assertFalse(c.update(setOf("グミ", "チョコレート"), tea, 1900))
        assertTrue(c.update(setOf("グミ", "チョコレート"), tea, 2000))
    }

    @Test
    fun `one frame that looks like the described shelf starts the wait again`() {
        val c = ShelfChange()
        c.update(setOf("グミ"), tea, 1000)
        assertFalse(c.update(setOf("緑茶"), tea, 1500))   // 認識のぶれで元の棚に見えた
        assertFalse(c.update(setOf("グミ"), tea, 2100))
        assertTrue(c.update(setOf("グミ"), tea, 3100))
    }

    @Test
    fun `half or more of the kinds in common is the same shelf`() {
        val c = ShelfChange()
        assertEquals(0.5f, ShelfChange.overlap(setOf("緑茶", "麦茶"), tea), 1e-6f)
        assertFalse(c.update(setOf("緑茶", "麦茶"), tea, 0))
        assertFalse(c.update(setOf("緑茶", "麦茶"), tea, 5000))
    }

    @Test
    fun `no kinds seen resets the wait`() {
        val c = ShelfChange()
        c.update(setOf("グミ"), tea, 0)
        c.update(emptySet(), tea, 500)
        assertFalse(c.update(setOf("グミ"), tea, 1200))
        assertTrue(c.describe(1200).startsWith("前の説明と重なり 0%"))
    }
}
