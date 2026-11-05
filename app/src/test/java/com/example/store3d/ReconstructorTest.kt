package com.example.store3d

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class ReconstructorTest {
    private val labels = listOf("A", "B", "アイス")

    /** Vertical rect on z=z0 facing +z, spanning x0..x1, slightly tilted by yawDeg. */
    private fun wall(x0: Float, x1: Float, z0: Float, top: Float, yawDeg: Float = 0f): PlaneSnap {
        val a = Math.toRadians(yawDeg.toDouble())
        val dz = (x1 - x0) * sin(a).toFloat()
        return PlaneSnap(true, floatArrayOf(x0, 0.2f, z0, x1, 0.2f, z0 + dz, x1, top, z0 + dz, x0, top, z0),
            floatArrayOf(-sin(a).toFloat(), 0f, cos(a).toFloat()))
    }

    private fun flat(x0: Float, x1: Float, z0: Float, z1: Float, y: Float) =
        PlaneSnap(false, floatArrayOf(x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1), floatArrayOf(0f, 1f, 0f))

    private fun rayTo(from: FloatArray, to: FloatArray): Ray {
        val d = FloatArray(3) { to[it] - from[it] }
        val l = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        return Ray(from, FloatArray(3) { d[it] / l })
    }

    @Test
    fun mergesNoisyPlanesSplitsAtBoundaryAndLabels() {
        val planes = listOf(
            wall(0f, 1.3f, 0f, 1.6f, 2f),
            wall(1.1f, 2.4f, 0.06f, 1.8f, -3f), // same face, a bit off: must merge
            flat(-3f, 5f, -2f, 6f, 0f),           // floor
            flat(3.5f, 5f, 2f, 3f, 0.85f),        // freezer top
        )
        val cam = floatArrayOf(1.2f, 1.4f, 2f)
        val bnd = List(3) { rayTo(cam, floatArrayOf(1.2f + it * 0.02f, 1.0f, 0f)) } +
            rayTo(cam, floatArrayOf(2.0f, 1.0f, 0f)) // single vote = noise
        val lab = listOf(
            LabelRay(rayTo(cam, floatArrayOf(0.5f, 1f, 0f)), floatArrayOf(0.9f, 0.05f, 0.05f)),
            LabelRay(rayTo(cam, floatArrayOf(1.8f, 1f, 0f)), floatArrayOf(0.1f, 0.8f, 0.1f)),
            LabelRay(rayTo(floatArrayOf(4f, 1.6f, 3.6f), floatArrayOf(4.2f, 0.85f, 2.5f)), floatArrayOf(0f, 0.1f, 0.9f)),
        )
        val boxes = Reconstructor.build(planes, lab, bnd, labels)
        val shelves = boxes.filter { it.kind == Kind.SHELF }.sortedBy { it.center[0] }
        assertEquals(2, shelves.size)
        assertEquals(listOf("A", "B"), shelves.map { it.label })
        assertEquals(1.21f, shelves[0].center[0] + shelves[0].size[0] / 2 + 0.02f, 0.08f)
        assertEquals(1.8f, shelves[1].size[1], 0.01f)
        val fr = boxes.single { it.kind == Kind.FREEZER }
        assertEquals("アイス", fr.label)
        assertEquals(0.85f, fr.size[1], 0.01f)
        assertEquals(1, boxes.count { it.kind == Kind.FLOOR })
    }

    @Test
    fun longFaceWithoutBoundariesIsSplitIntoShelfSizedUnits() {
        val boxes = Reconstructor.build(listOf(wall(0f, 5f, 0f, 1.8f), flat(-1f, 6f, -1f, 3f, 0f)), emptyList(), emptyList(), labels)
        val shelves = boxes.filter { it.kind == Kind.SHELF }
        assertEquals(3, shelves.size)
        shelves.forEach { assertEquals(UNKNOWN, it.label) }
    }

    @Test
    fun clusterDropsSingleVotesAndTooCloseCuts() {
        val c = Reconstructor.clusterCuts(listOf(1.0f, 1.02f, 1.05f, 1.25f, 1.3f, 2.0f, 3.0f, 3.01f), 0f, 4f)
        assertEquals(listOf(1.02f, 3.01f), c)
    }

    @Test
    fun uprightToSensorRotation() {
        // portrait upright (0.25, 0.1) on a 640x480 sensor image rotated 90°
        val p = uprightToSensor(0.25f, 0.1f, 90, 640, 480)
        assertEquals(64f, p[0], 1e-3f)
        assertEquals(360f, p[1], 1e-3f)
        val q = uprightToSensor(0.25f, 0.1f, 0, 640, 480)
        assertEquals(160f, q[0], 1e-3f)
    }

    @Test
    fun decodePeaksFindsRefinedLocalMaxima() {
        val p = FloatArray(10); p[2] = 0.9f; p[3] = 0.9f * 0.5f; p[1] = 0.9f * 0.5f; p[7] = 0.3f
        assertEquals(listOf(0.25f), decodePeaks(p, 0.5f))
    }
}
