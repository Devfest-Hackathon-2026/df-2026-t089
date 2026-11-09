package jp.devfest.shelfsight.assist

import jp.devfest.shelfsight.cloud.HttpResult
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BackendClientTest {
    private val model = """{"yaw":0.5,"room":[0,0,4,3],"fixtures":[{"kind":"wall","axis":1,"c":1.5,"a":0.0,"b":2.1,"d":0.45,"h":1.8,"n":2,"w":1.05,
        "widths":[0.9,1.2],"faces":{"+":[[["お茶",0.8],["水",0.2]],[]]},"themes":{"+":["お茶・水",null]},"obs":40.0}]}"""

    private fun client(code: Int, body: String, sent: MutableList<String> = ArrayList()) =
        BackendClient("http://server/", "k", { url, headers, b ->
            sent += url
            assertEquals("k", headers["X-Api-Key"])
            sent += b
            HttpResult(code, body)
        }, {})

    @Test fun `a saved model becomes a store view`() {
        val view = client(200, """{"name":"店","model":$model}""").lookup(35.0, 139.0)!!.view!!
        val f = view.fixtures.single()
        assertEquals(0.5, view.yaw, 1e-9)
        assertEquals(listOf(0.9, 1.2), f.moduleWidths)
        assertEquals("お茶" to 0.8, f.faces.getValue("+")[0][0])
        assertEquals(listOf("お茶・水", null), f.themes["+"])
    }

    @Test fun `uploading sends the location and the observations to the right address`() {
        val sent = ArrayList<String>()
        assertTrue(client(200, """{"saved":true}""", sent).upload(35.0, 139.0, """{"path":[[0,1]],"observations":[]}"""))
        assertEquals("http://server/v1/maps", sent[0])
        val body = JSONObject(sent[1])
        assertEquals(35.0, body.getDouble("lat"), 1e-9)
        assertEquals(1, body.getJSONArray("path").length())
    }

    @Test fun `a store without a saved map has a name and no view`() {
        val found = client(200, """{"place_id":"p","name":"店","model":null}""").lookup(35.0, 139.0)!!
        assertEquals("店", found.name)
        assertNull(found.view)
    }

    @Test fun `server errors give null`() {
        assertNull(client(500, "oops").lookup(35.0, 139.0))
    }
}
