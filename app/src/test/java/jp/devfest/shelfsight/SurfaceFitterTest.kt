package jp.devfest.shelfsight

import java.util.Random
import jp.devfest.shelfsight.mapping.FitResult
import jp.devfest.shelfsight.mapping.PlaneSpec
import jp.devfest.shelfsight.mapping.SurfaceFitter
import jp.devfest.shelfsight.mapping.SurfaceKind
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SurfaceFitterTest {
    private val floorY = -1.3f
    private val ceiling = 1f

    private class Points {
        val values = ArrayList<Float>()
        fun add(x: Float, y: Float, z: Float) {
            values += x; values += y; values += z
        }
        fun lattice(
            x0: Float, x1: Float, y0: Float, y1: Float, z0: Float, z1: Float,
            keep: (Float, Float, Float) -> Boolean = { _, _, _ -> true },
        ) {
            var x = x0
            while (x <= x1) {
                var y = y0
                while (y <= y1) {
                    var z = z0
                    while (z <= z1) {
                        if (keep(x, y, z)) add(x, y, z)
                        z += 0.05f
                    }
                    y += 0.05f
                }
                x += 0.05f
            }
        }
        fun array() = values.toFloatArray()
        val count get() = values.size / 3
    }

    private fun floorSeed(x0: Float, x1: Float, z0: Float, z1: Float, y: Float = floorY) = PlaneSpec(
        origin = floatArrayOf((x0 + x1) / 2, y, (z0 + z1) / 2),
        normal = floatArrayOf(0f, 1f, 0f),
        kind = SurfaceKind.FLOOR,
        polygon = floatArrayOf(x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1),
        area = (x1 - x0) * (z1 - z0),
    )

    private fun fit(points: Points, specs: List<PlaneSpec>): FitResult =
        SurfaceFitter().fit(points.array(), points.count, specs, ceiling)

    @Test
    fun `a detected floor grows out to the connected points`() {
        val points = Points().apply { lattice(0f, 8f, floorY, floorY, 0f, 5f) }
        val surface = fit(points, listOf(floorSeed(3f, 4f, 2f, 3f))).surfaces.single()
        assertEquals(40f, surface.area, 4f)
    }

    @Test
    fun `the detected range stays even without points`() {
        val surface = fit(Points(), listOf(floorSeed(0f, 2f, 0f, 2f))).surfaces.single()
        assertEquals(4f, surface.area, 1f)
    }

    @Test
    fun `a floor can take an L shape, which a convex polygon cannot`() {
        val points = Points().apply {
            lattice(0f, 6f, floorY, floorY, 0f, 6f) { x, _, z -> x < 2f || z < 2f }
        }
        val surface = fit(points, listOf(floorSeed(0.5f, 1.5f, 0.5f, 1.5f))).surfaces.single()
        assertEquals(20f, surface.area, 2.5f) // 凸多角形で包むと 36 平方メートルになる
    }

    @Test
    fun `small gaps are bridged but a distant island is not joined`() {
        val points = Points().apply {
            lattice(0f, 2f, floorY, floorY, 0f, 2f)
            lattice(2.25f, 4f, floorY, floorY, 0f, 2f) // 25cm の隙間
            lattice(8f, 10f, floorY, floorY, 0f, 2f) // 4m 離れた点の塊
        }
        val surface = fit(points, listOf(floorSeed(0.5f, 1.5f, 0.5f, 1.5f))).surfaces.single()
        assertEquals(8f, surface.area, 1.5f)
    }

    @Test
    fun `coplanar detected planes become a single surface`() {
        val points = Points().apply { lattice(0f, 6f, floorY, floorY, 0f, 2f) }
        val result = fit(points, listOf(floorSeed(0f, 1f, 0f, 1f), floorSeed(5f, 6f, 1f, 2f, floorY + 0.02f)))
        assertEquals(1, result.surfaces.size)
        assertEquals(2, result.surfaces.single().members.size)
    }

    @Test
    fun `a wall ARCore missed is found from points and fills in on the next fit`() {
        val fitter = SurfaceFitter()
        val points = Points().apply {
            lattice(0f, 8f, floorY, floorY, 0f, 5f)
            lattice(1f, 5f, floorY + 0.15f, floorY + 2f, -1f, -1f)
        }
        val floor = floorSeed(3f, 4f, 2f, 3f)
        val first = fitter.fit(points.array(), points.count, listOf(floor), ceiling)
        val wall = first.discovered.single { it.kind == SurfaceKind.WALL }
        assertEquals(1f, abs(wall.normal[2]), 0.02f)
        assertEquals(-1f, wall.origin[2], 0.03f)

        val second = fitter.fit(points.array(), points.count, listOf(floor, wall), ceiling)
        val surface = second.surfaces.single { it.kind == SurfaceKind.WALL }
        assertEquals(7.4f, surface.area, 1.5f)
    }

    @Test
    fun `a counter top is found as a raised surface`() {
        val points = Points().apply {
            lattice(0f, 8f, floorY, floorY, 0f, 5f)
            lattice(2f, 4f, floorY + 0.9f, floorY + 0.9f, 1f, 2f)
        }
        val raised = fit(points, listOf(floorSeed(0f, 1f, 3f, 4f))).discovered.single()
        assertEquals(SurfaceKind.RAISED, raised.kind)
        assertEquals(floorY + 0.9f, raised.origin[1], 0.03f)
    }

    @Test
    fun `scattered noise does not produce surfaces`() {
        val random = Random(1)
        val points = Points().apply {
            lattice(0f, 8f, floorY, floorY, 0f, 5f)
            repeat(20_000) {
                add(random.nextFloat() * 4f, floorY + 0.2f + random.nextFloat() * 2f, random.nextFloat() * 4f)
            }
        }
        val result = fit(points, listOf(floorSeed(3f, 4f, 2f, 3f)))
        assertEquals(1, result.surfaces.size)
        assertTrue("found ${result.discovered.size}", result.discovered.isEmpty())
    }

    @Test
    fun `a narrow post is not reported as a wall`() {
        val points = Points().apply {
            lattice(0f, 8f, floorY, floorY, 0f, 5f)
            lattice(2f, 2.2f, floorY + 0.15f, floorY + 2f, 1f, 1f)
        }
        val result = fit(points, listOf(floorSeed(3f, 4f, 2f, 3f)))
        assertTrue("found ${result.discovered.size}", result.discovered.isEmpty())
    }

    @Test
    fun `nothing is discovered before the floor height is known`() {
        val points = Points().apply { lattice(1f, 5f, -1f, 1f, -1f, -1f) }
        assertTrue(fit(points, emptyList()).discovered.isEmpty())
    }

    /** 遠くの壁の手がかり。10cm 単位で、奥行きに揺れた厚みを持つ。 */
    private fun farWall(x0: Float, x1: Float, z: Float, y1: Float = floorY + 2f) = Points().apply {
        var x = x0
        while (x <= x1) {
            var y = floorY + 0.35f
            while (y <= y1) {
                add(x, y, z - 0.1f); add(x, y, z); add(x, y, z + 0.1f)
                y += 0.1f
            }
            x += 0.1f
        }
    }

    private val floorOnly = Points().apply { lattice(0f, 3f, floorY, floorY, 0f, 3f) }

    @Test
    fun `a distant wall is found from the far evidence alone`() {
        val fitter = SurfaceFitter()
        val far = farWall(0f, 6f, -5f)
        val floor = floorSeed(1f, 2f, 1f, 2f)
        val first = fitter.fit(floorOnly.array(), floorOnly.count, listOf(floor), ceiling, far.array(), far.count)
        val wall = first.discovered.single()
        assertTrue(wall.fromFarPoints)
        assertEquals(SurfaceKind.WALL, wall.kind)
        assertEquals(-5f, wall.origin[2], 0.1f)

        val second = fitter.fit(floorOnly.array(), floorOnly.count, listOf(floor, wall), ceiling, far.array(), far.count)
        val surface = second.surfaces.single { it.kind == SurfaceKind.WALL }
        assertEquals(9.9f, surface.area, 2f)
    }

    @Test
    fun `far evidence extends a wall ARCore detected`() {
        val wall = PlaneSpec(
            origin = floatArrayOf(1f, -0.5f, -5f),
            normal = floatArrayOf(0f, 0f, 1f),
            kind = SurfaceKind.WALL,
            polygon = floatArrayOf(0.5f, -1f, -5f, 1.5f, -1f, -5f, 1.5f, 0f, -5f, 0.5f, 0f, -5f),
            area = 1f,
        )
        val far = farWall(0f, 6f, -5f)
        val result = SurfaceFitter().fit(floorOnly.array(), floorOnly.count, listOf(floorSeed(1f, 2f, 1f, 2f), wall), ceiling, far.array(), far.count)
        val surface = result.surfaces.single { it.kind == SurfaceKind.WALL }
        assertEquals(9.9f, surface.area, 2f)
    }

    @Test
    fun `far evidence does not widen the floor`() {
        val far = Points().apply {
            var x = 3.1f
            while (x < 8f) {
                var z = 0f
                while (z < 3f) {
                    add(x, floorY, z)
                    z += 0.1f
                }
                x += 0.1f
            }
        }
        val result = SurfaceFitter().fit(floorOnly.array(), floorOnly.count, listOf(floorSeed(1f, 2f, 1f, 2f)), ceiling, far.array(), far.count)
        assertEquals(9f, result.surfaces.single().area, 1.5f)
        assertTrue(result.discovered.isEmpty())
    }

    @Test
    fun `scattered far evidence does not become a wall`() {
        val random = Random(2)
        val far = Points().apply {
            repeat(8_000) {
                add(random.nextFloat() * 8f, floorY + 0.35f + random.nextFloat() * 2f, random.nextFloat() * 8f)
            }
        }
        val result = SurfaceFitter().fit(floorOnly.array(), floorOnly.count, listOf(floorSeed(1f, 2f, 1f, 2f)), ceiling, far.array(), far.count)
        assertTrue("found ${result.discovered.size}", result.discovered.isEmpty())
    }

    @Test
    fun `a far wall merges into a wall found up close`() {
        val near = PlaneSpec(floatArrayOf(1f, 0f, -5f), floatArrayOf(0f, 0f, 1f), SurfaceKind.WALL, null, 2f)
        val far = PlaneSpec(floatArrayOf(4f, 0f, -5.1f), floatArrayOf(0f, 0f, 1f), SurfaceKind.WALL, null, 8f, fromFarPoints = true)
        val points = Points().apply { lattice(0f, 6f, -1f, 1f, -5f, -5f) }
        val surface = SurfaceFitter().fit(points.array(), points.count, listOf(far, near), ceiling).surfaces.single()
        assertTrue(surface.reference === near)
        assertEquals(2, surface.members.size)
    }

    @Test
    fun `points above the ceiling limit are ignored`() {
        val points = Points().apply {
            lattice(0f, 8f, floorY, floorY, 0f, 5f)
            lattice(0f, 8f, 1.2f, 1.2f, 0f, 5f)
        }
        val result = fit(points, listOf(floorSeed(3f, 4f, 2f, 3f)))
        assertTrue(result.discovered.isEmpty())
    }
}
