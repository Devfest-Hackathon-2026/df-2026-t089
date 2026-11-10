package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class BeaconTest {
    private val here = Listener(Vec3(0f, 1.3f, 0f), 0f, -1f)
    private fun item(name: String, genre: String?, x: Float, z: Float) = MapItem(name, genre, null, Vec3(x, 1f, z), 0.1f, 0.2f, 3, 0L)

    @Test
    fun `where questions give the thing asked for`() {
        assertEquals("お茶", Beacon.whereQuery("お茶はどこ？"))
        assertEquals("チョコ", Beacon.whereQuery("チョコの場所はどこ"))
        assertEquals("カップ麺", Beacon.whereQuery("カップ麺 どこにありますか"))
        assertEquals("おにぎり", Beacon.whereQuery("おにぎり売り場はどっち"))
        assertNull(Beacon.whereQuery("これは何？"))
        assertNull(Beacon.whereQuery("右には何がある"))
    }

    @Test
    fun `the nearest matching product, else the shelf whose main goods match`() {
        val items = listOf(item("緑茶", "ソフトドリンク", 3f, 0f), item("緑茶", "ソフトドリンク", -1.5f, 0f), item("コーラ", "ソフトドリンク", 1f, 0f))
        assertEquals("緑茶" to Vec3(-1.5f, 1f, 0f), Beacon.find("緑茶", items, null, here))
        assertNull(Beacon.find("おにぎり", items, null, here))
        val view = StoreView(0.0, null, listOf(StoreFixture("wall", 0, 2.0, -0.6, 0.6, 0.0, 1.8, 1, 1.2, mapOf("-" to listOf(listOf("おにぎり" to 0.9))))))
        assertEquals("おにぎり" to Vec3(2f, 1.3f, 0f), Beacon.find("おにぎり", items, view, here))
    }

    @Test
    fun `plays every 3 seconds until the user is within a meter`() {
        val b = Beacon("お茶", Vec3(0f, 1f, -4f))
        assertEquals(Beacon.Event.PLAY, b.update(here, 0L))
        assertNull(b.update(here, 2_999L))
        assertEquals(Beacon.Event.PLAY, b.update(here, 3_000L))
        assertEquals(Beacon.Event.ARRIVED, b.update(Listener(Vec3(0f, 1.3f, -3.2f), 0f, -1f), 3_500L))
    }
}
