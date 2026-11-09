package jp.devfest.shelfsight

import jp.devfest.shelfsight.assist.GlassProjection
import jp.devfest.shelfsight.assist.MapItem
import jp.devfest.shelfsight.assist.StoreModel
import jp.devfest.shelfsight.model.Vec3
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
    fun scatterAlongTheViewRaysBecomesAGlassCaseAndIsRestoredOntoIt() {
        val model = StoreModel(emptyList())
        val rng = java.util.Random(1)
        for (i in 0 until 500) {
            val camZ = 0.2 + rng.nextDouble() * 2.6
            val depth = 1.6 + rng.nextDouble() * 2.6
            model.addProduct(0.0, camZ, depth, camZ + (rng.nextDouble() * 1.8 - 0.9) * depth / 3, "tea", "")
            model.addProduct(0.0, camZ, -1.5, rng.nextDouble() * 3, "battery", "")
        }
        for (v in 0 until 10) model.addCamera(0.0, v * 0.3)
        val view = model.estimate()
        assertTrue(StoreModel.text(view), view.fixtures.any { it.kind == "glass" })
        val item = { x: Float, z: Float -> MapItem("tea", null, null, Vec3(x, 1.3f, z), 0.1f, 0.1f, 1, 0L, Vec3(0f, 1.3f, z)) }
        assertTrue((0..6).any { i -> GlassProjection.corrected(item(3.5f, i * 0.5f), view)?.let { it.x < 3.5f } == true })
        assertEquals(null, GlassProjection.corrected(item(-1.5f, 1.5f), view))
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
