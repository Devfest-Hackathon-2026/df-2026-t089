package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.StoreModel
import org.junit.Assert.assertEquals
import org.junit.Test

/** 同じ観測を流して、products/store_model.py と同じ記述になることを確かめる（データは python products/store_sim.py --fixture）。 */
class StoreModelTest {
    @Test
    fun matchesThePythonEstimator() {
        val all = javaClass.classLoader!!.getResource("store_model_trace.tsv")!!.readText(Charsets.UTF_8).replace("\r\n", "\n")
        val (input, expected) = all.split("\n===\n")
        val m = StoreModel()
        for (line in input.lines()) {
            val p = line.split("\t")
            when (p[0]) {
                "P" -> m.addProduct(p[1].toDouble(), p[2].toDouble(), p[3].toDouble(), p[4].toDouble(), p[5], p[6])
                "C" -> m.addCounter(p[1].toDouble(), p[2].toDouble())
                "T" -> m.addCamera(p[1].toDouble(), p[2].toDouble())
            }
        }
        assertEquals(expected.trim(), StoreModel.text(m.estimate()))
    }
}
