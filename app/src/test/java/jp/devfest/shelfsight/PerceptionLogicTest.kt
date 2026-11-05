package jp.devfest.shelfsight

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.ar.DepthSnapshot
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.perception.AheadKind
import jp.devfest.shelfsight.perception.ObstacleClassifier
import jp.devfest.shelfsight.perception.PersonDistance
import jp.devfest.shelfsight.perception.PersonMotion
import jp.devfest.shelfsight.perception.PersonObservation
import jp.devfest.shelfsight.perception.PersonTracker
import jp.devfest.shelfsight.perception.SurfaceSample
import jp.devfest.shelfsight.perception.TrackedPerson
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PerceptionLogicTest {
    private val box = NormalizedBox(0.4f, 0.1f, 0.6f, 0.9f)
    private val camera = Vec3(0f, 1.3f, 0f)

    /** 人が [start] から毎秒 [velocity] で動く様子を 2 秒分（10Hz）与え、最後の判定を返す。 */
    private fun motionOf(start: Vec3, velocity: Vec3, cameraAt: (Double) -> Vec3 = { camera }): PersonMotion {
        val tracker = PersonTracker()
        var last = PersonMotion.UNKNOWN
        for (i in 0 until 20) {
            val t = i * 0.1
            val p = Vec3(
                start.x + velocity.x * t.toFloat(),
                start.y,
                start.z + velocity.z * t.toFloat(),
            )
            last = tracker.update(t, listOf(PersonObservation(box, 3f, p)), cameraAt(t)).single().motion
        }
        return last
    }

    @Test
    fun `motion is unknown until enough time has passed`() {
        val tracker = PersonTracker()
        val motion = tracker.update(0.0, listOf(PersonObservation(box, 3f, Vec3(0f, 1f, -3f))), camera)
        assertEquals(PersonMotion.UNKNOWN, motion.single().motion)
    }

    @Test
    fun `standing person is stationary`() {
        assertEquals(PersonMotion.STATIONARY, motionOf(Vec3(0f, 1f, -3f), Vec3(0f, 0f, 0f)))
    }

    @Test
    fun `standing person stays stationary while the wearer walks toward them`() {
        val walking = motionOf(Vec3(0f, 1f, -5f), Vec3(0f, 0f, 0f)) { t -> Vec3(0f, 1.3f, -t.toFloat()) }
        assertEquals(PersonMotion.STATIONARY, walking)
    }

    @Test
    fun `person walking at the wearer is approaching`() {
        assertEquals(PersonMotion.APPROACHING, motionOf(Vec3(0f, 1f, -6f), Vec3(0f, 0f, 1.2f)))
    }

    @Test
    fun `person walking away is leaving`() {
        assertEquals(PersonMotion.LEAVING, motionOf(Vec3(0f, 1f, -2f), Vec3(0f, 0f, -1.2f)))
    }

    @Test
    fun `person walking sideways is crossing`() {
        assertEquals(PersonMotion.CROSSING, motionOf(Vec3(-2f, 1f, -3f), Vec3(1.2f, 0f, 0f)))
    }

    @Test
    fun `two people are tracked separately`() {
        val tracker = PersonTracker()
        var result = emptyList<TrackedPerson>()
        for (i in 0 until 20) {
            val t = i * 0.1
            result = tracker.update(
                t,
                listOf(
                    PersonObservation(box, 3f, Vec3(-3f, 1f, -3f)),
                    PersonObservation(box, 4f, Vec3(3f, 1f, -6f + 1.2f * t.toFloat())),
                ),
                camera,
            )
        }
        assertEquals(PersonMotion.STATIONARY, result[0].motion)
        assertEquals(PersonMotion.APPROACHING, result[1].motion)
    }

    @Test
    fun `person without a position stays unknown`() {
        val tracked = PersonTracker().update(0.0, listOf(PersonObservation(box, null, null)), camera)
        assertEquals(PersonMotion.UNKNOWN, tracked.single().motion)
    }

    /** 正面 [distance] メートルに、カメラと向かい合う縦の平面があるときの点。 */
    private fun wall(distance: Float) = (0 until 15).flatMap { row ->
        (0 until 15).map { col ->
            SurfaceSample(
                Vec3(-0.3f + 0.6f * col / 14f, 1.0f + 0.6f * row / 14f, -distance),
                distance,
            )
        }
    }

    @Test
    fun `flat vertical surface is a wall`() {
        val result = ObstacleClassifier.classify(wall(2f), camera.y, emptyList())
        assertEquals(AheadKind.WALL, result.kind)
        assertEquals(2f, result.distanceM!!, 0.01f)
    }

    @Test
    fun `a wall seen at an angle is still a wall`() {
        val angled = (0 until 15).flatMap { row ->
            (0 until 15).map { col ->
                val x = -0.4f + 0.8f * col / 14f
                val z = -2f + x
                SurfaceSample(Vec3(x, 1.0f + 0.6f * row / 14f, z), -z)
            }
        }
        assertEquals(AheadKind.WALL, ObstacleClassifier.classify(angled, camera.y, emptyList()).kind)
    }

    @Test
    fun `uneven surface is an object`() {
        // 手前と奥に凹凸のある、棚に置かれた物のような点
        val uneven = (0 until 15).flatMap { row ->
            (0 until 15).map { col ->
                val depth = 1.5f + (if ((row + col) % 2 == 0) 0f else 0.4f)
                SurfaceSample(Vec3(-0.3f + 0.6f * col / 14f, 1.0f + 0.6f * row / 14f, -depth), depth)
            }
        }
        assertEquals(AheadKind.OBJECT, ObstacleClassifier.classify(uneven, camera.y, emptyList()).kind)
    }

    @Test
    fun `floor alone is not an obstacle`() {
        val floor = (0 until 15).flatMap { row ->
            (0 until 15).map { col ->
                val z = -2f - row * 0.1f
                SurfaceSample(Vec3(-0.3f + 0.6f * col / 14f, 0f, z), -z)
            }
        }
        val result = ObstacleClassifier.classify(floor, camera.y, emptyList())
        assertEquals(AheadKind.UNKNOWN, result.kind)
        assertNull(result.distanceM)
    }

    @Test
    fun `missing depth is unknown rather than clear`() {
        assertEquals(AheadKind.UNKNOWN, ObstacleClassifier.classify(emptyList(), camera.y, emptyList()).kind)
    }

    @Test
    fun `person in front of a wall is reported as the person`() {
        val person = TrackedPerson(box, 1.5f, PersonMotion.APPROACHING)
        val result = ObstacleClassifier.classify(wall(3f), camera.y, listOf(person))
        assertEquals(AheadKind.PERSON, result.kind)
        assertEquals(1.5f, result.distanceM!!, 0.01f)
        assertEquals(PersonMotion.APPROACHING, result.motion)
    }

    @Test
    fun `wall in front of a person is reported as the wall`() {
        val person = TrackedPerson(box, 4f, PersonMotion.STATIONARY)
        assertEquals(AheadKind.WALL, ObstacleClassifier.classify(wall(2f), camera.y, listOf(person)).kind)
    }

    @Test
    fun `person off to the side does not count as ahead`() {
        val side = TrackedPerson(NormalizedBox(0f, 0.1f, 0.2f, 0.9f), 1f, PersonMotion.STATIONARY)
        assertEquals(AheadKind.WALL, ObstacleClassifier.classify(wall(2f), camera.y, listOf(side)).kind)
    }

    @Test
    fun `person with unknown distance takes priority over a wall`() {
        val person = TrackedPerson(box, null, PersonMotion.UNKNOWN)
        val result = ObstacleClassifier.classify(wall(2f), camera.y, listOf(person))
        assertEquals(AheadKind.PERSON, result.kind)
        assertNull(result.distanceM)
    }

    private val identity = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)

    private fun geometry(depthMm: Int?, rotation: Int = 90) = CaptureGeometry(
        imageWidth = 640, imageHeight = 480,
        focalX = 500f, focalY = 500f, principalX = 320f, principalY = 240f,
        cameraToWorld = identity,
        uprightRotationDegrees = rotation,
        depth = depthMm?.let { mm -> DepthSnapshot(160, 120, ShortArray(160 * 120) { mm.toShort() }) },
        depthU0 = 0f, depthUPerPixel = 1f / 640f, depthV0 = 0f, depthVPerPixel = 1f / 480f,
    )

    @Test
    fun `full body distance uses the upright height of the image`() {
        // 縦向きでは正立画像の高さは 640px。人の高さ 50% = 320px なので 500 * 1.65 / 320
        val distance = PersonDistance.estimate(NormalizedBox(0.4f, 0.25f, 0.6f, 0.75f), geometry(depthMm = 9000))
        assertEquals(2.578f, distance!!, 0.01f)
    }

    @Test
    fun `cut off body uses depth`() {
        val distance = PersonDistance.estimate(NormalizedBox(0.3f, 0f, 0.7f, 1f), geometry(depthMm = 1200))
        assertEquals(1.2f, distance!!, 0.01f)
    }

    @Test
    fun `cut off body without depth has no distance`() {
        assertNull(PersonDistance.estimate(NormalizedBox(0.3f, 0f, 0.7f, 1f), geometry(depthMm = null)))
    }

    @Test
    fun `surface samples skip people and missing depth`() {
        val all = ObstacleClassifier.collectSamples(geometry(depthMm = 2000), emptyList())
        assertEquals(225, all.size)
        val withPerson = ObstacleClassifier.collectSamples(geometry(depthMm = 2000), listOf(box))
        assertEquals(0, withPerson.size)
        assertEquals(0, ObstacleClassifier.collectSamples(geometry(depthMm = null), emptyList()).size)
    }
}
