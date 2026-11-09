package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.StoreModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** 同じ観測を流して、products/store_model.py と同じ記述になることを確かめる（データは python products/store_sim.py --fixture）。 */
class StoreModelTest {
    @Test
    fun measuredIslandKeepsItsDepthAndFitsMixedModuleWidths() {
        val model = StoreModel(emptyList())
        for (i in 0..100) {
            val z = 0.05 + i * 2.9 / 100
            model.addProduct(0.4, z, 1.56, z, "A", "")
            model.addProduct(3.6, z, 2.44, z, "B", "")
        }
        val island = model.estimate().fixtures.single { it.kind == "island" }
        assertEquals(0.88, island.d, 0.05)
        assertEquals(3, island.n)
        assertEquals(1, island.moduleWidths.count { it > 1.05 })
        assertEquals(2.9, island.b - island.a, 0.05)

        // A later batch is included by the next estimate, without restarting the model.
        val before = island.b
        for (i in 101..130) {
            val z = 0.05 + i * 2.9 / 100
            model.addProduct(0.4, z, 1.56, z, "A", "")
            model.addProduct(3.6, z, 2.44, z, "B", "")
        }
        assertTrue(model.estimate().fixtures.single { it.kind == "island" }.b > before)
    }

    @Test
    fun matchesThePythonEstimator() {
        val all = javaClass.classLoader!!.getResource("store_model_trace.tsv")!!.readText(Charsets.UTF_8).replace("\r\n", "\n")
        val (input, expected) = all.split("\n===\n")
        val root = generateSequence(java.io.File("").absoluteFile) { it.parentFile }.first { java.io.File(it, "settings.gradle.kts").exists() }
        val m = StoreModel(StoreModel.parseThemes(java.io.File(root, "app/src/main/assets/store_themes.json").readText()))
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
