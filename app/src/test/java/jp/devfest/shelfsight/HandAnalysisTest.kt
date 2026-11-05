package jp.devfest.shelfsight

import jp.devfest.shelfsight.ar.CaptureGeometry
import jp.devfest.shelfsight.model.NormalizedBox
import jp.devfest.shelfsight.model.Vec3
import jp.devfest.shelfsight.perception.HandAnalysis
import jp.devfest.shelfsight.perception.HandBoxMemory
import jp.devfest.shelfsight.perception.HandGesture
import jp.devfest.shelfsight.perception.HandLandmarks
import jp.devfest.shelfsight.perception.OwnHandFilter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HandAnalysisTest {
    private val identity = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)

    private fun geometry(rotation: Int = 0, cameraToWorld: FloatArray = identity) = CaptureGeometry(
        imageWidth = 640, imageHeight = 640,
        focalX = 500f, focalY = 500f, principalX = 320f, principalY = 320f,
        cameraToWorld = cameraToWorld,
        uprightRotationDegrees = rotation,
        depth = null,
        depthU0 = 0f, depthUPerPixel = 1f / 640f, depthV0 = 0f, depthVPerPixel = 1f / 640f,
    )

    /**
     * 指を上（-Y）へ伸ばした手。[extended] は人差し指、中指、薬指、小指が伸びているか。
     * 手のひらをカメラに向けた右手の配置で、人差し指が画像の右、小指が左にある。
     */
    private fun worldHand(extended: List<Boolean>): List<Vec3> {
        val points = ArrayList<Vec3>()
        points += Vec3(0f, 0f, 0f) // 手首
        repeat(4) { i -> points += Vec3(0.04f - 0.01f * i, -0.02f * i, 0f) } // 親指
        val baseX = listOf(0.03f, 0.01f, -0.01f, -0.03f)
        for (finger in 0 until 4) {
            val x = baseX[finger]
            points += Vec3(x, -0.09f, 0f) // 付け根
            if (extended[finger]) {
                points += Vec3(x, -0.12f, 0f)
                points += Vec3(x, -0.15f, 0f)
                points += Vec3(x, -0.18f, 0f)
            } else {
                points += Vec3(x, -0.11f, 0.02f)
                points += Vec3(x, -0.09f, 0.03f)
                points += Vec3(x, -0.07f, 0.02f)
            }
        }
        return points
    }

    /** 実寸を、カメラから [distance] メートルにある画像座標へ投影する。 */
    private fun imageOf(world: List<Vec3>, distance: Float) = world.map {
        val scale = 500f / distance / 640f
        Vec3(0.5f + it.x * scale, 0.5f + it.y * scale, 0f)
    }

    private fun hand(extended: List<Boolean>, distance: Float = 0.4f, isRight: Boolean = true): HandLandmarks {
        val world = worldHand(extended)
        return HandLandmarks(imageOf(world, distance), world, isRight)
    }

    private val pointing = listOf(true, false, false, false)
    private val open = listOf(true, true, true, true)
    private val fist = listOf(false, false, false, false)

    @Test
    fun `gestures are recognized from finger extension`() {
        assertEquals(HandGesture.POINTING, HandAnalysis.gestureOf(worldHand(pointing)))
        assertEquals(HandGesture.OPEN, HandAnalysis.gestureOf(worldHand(open)))
        assertEquals(HandGesture.FIST, HandAnalysis.gestureOf(worldHand(fist)))
        assertEquals(HandGesture.OTHER, HandAnalysis.gestureOf(worldHand(listOf(true, true, false, false))))
    }

    @Test
    fun `index direction points from the knuckle to the fingertip`() {
        val info = HandAnalysis.analyze(hand(pointing), geometry())
        assertEquals(0f, info.indexDirection.x, 1e-3f)
        assertEquals(-1f, info.indexDirection.y, 1e-3f)
    }

    @Test
    fun `distance is recovered from hand size`() {
        val info = HandAnalysis.analyze(hand(open, distance = 0.4f), geometry())
        assertEquals(0.4f, info.distanceM!!, 0.01f)
    }

    @Test
    fun `implausible hand size gives no distance or fingertip position`() {
        val info = HandAnalysis.analyze(hand(open, distance = 3f), geometry())
        assertNull(info.distanceM)
        assertNull(info.fingertipWorld)
    }

    @Test
    fun `fingertip world position is placed in front of the camera`() {
        val info = HandAnalysis.analyze(hand(pointing, distance = 0.4f), geometry())
        val tip = info.fingertipWorld!!
        assertEquals(-0.4f, tip.z, 0.02f)
        assertTrue("fingertip above image center: $tip", tip.y > 0f)
    }

    @Test
    fun `palm side uses the handedness`() {
        assertTrue(HandAnalysis.analyze(hand(open, isRight = true), geometry()).palmFacesCamera)
        assertFalse(HandAnalysis.analyze(hand(open, isRight = false), geometry()).palmFacesCamera)
    }

    private fun assertVec(expected: Vec3, actual: Vec3) {
        assertTrue("expected $expected but was $actual", expected.distanceTo(actual) < 1e-3f)
    }

    @Test
    fun `direction into the scene is forward in world space`() {
        assertVec(Vec3(0f, 0f, -1f), HandAnalysis.toWorldDirection(Vec3(0f, 0f, 1f), geometry()))
    }

    @Test
    fun `upright up maps through each sensor rotation`() {
        val up = Vec3(0f, -1f, 0f)
        assertVec(Vec3(0f, 1f, 0f), HandAnalysis.toWorldDirection(up, geometry(rotation = 0)))
        // 時計回り 90 度で正立させるとき、正立画像の上はセンサー画像の左
        assertVec(Vec3(-1f, 0f, 0f), HandAnalysis.toWorldDirection(up, geometry(rotation = 90)))
        assertVec(Vec3(0f, -1f, 0f), HandAnalysis.toWorldDirection(up, geometry(rotation = 180)))
        assertVec(Vec3(1f, 0f, 0f), HandAnalysis.toWorldDirection(up, geometry(rotation = 270)))
    }

    @Test
    fun `camera rotation is applied to the pointing direction`() {
        // カメラ座標の +X をワールドの -Z へ、-Z をワールドの -X へ向ける回転（Y 軸まわり 90 度）
        val turned = floatArrayOf(0f, 0f, -1f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 1f)
        val world = HandAnalysis.toWorldDirection(Vec3(0f, 0f, 1f), geometry(cameraToWorld = turned))
        assertVec(Vec3(-1f, 0f, 0f), world)
    }

    @Test
    fun `direction labels cover forward and image directions`() {
        assertEquals("前方", HandAnalysis.directionLabel(Vec3(0f, 0f, 1f)))
        assertEquals("手前", HandAnalysis.directionLabel(Vec3(0f, 0f, -1f)))
        assertEquals("右", HandAnalysis.directionLabel(Vec3(1f, 0f, 0f)))
        assertEquals("上", HandAnalysis.directionLabel(Vec3(0f, -1f, 0f)))
        assertEquals("左下", HandAnalysis.directionLabel(Vec3(-0.7f, 0.7f, 0f)))
    }

    private val handBox = NormalizedBox(0.4f, 0.5f, 0.6f, 0.7f) // 面積 0.04

    @Test
    fun `person box that is mostly a hand is removed`() {
        val arm = NormalizedBox(0.35f, 0.45f, 0.65f, 1f) // 面積 0.165、手の割合 約 24%
        assertEquals(emptyList<NormalizedBox>(), OwnHandFilter.withoutOwnHands(listOf(arm), listOf(handBox)))
    }

    @Test
    fun `real person behind a hand is kept`() {
        val person = NormalizedBox(0.1f, 0f, 0.9f, 1f) // 面積 0.8、手の割合 5%
        assertEquals(listOf(person), OwnHandFilter.withoutOwnHands(listOf(person), listOf(handBox)))
    }

    @Test
    fun `person elsewhere in the image is kept when a hand is visible`() {
        val person = NormalizedBox(0.7f, 0.1f, 0.8f, 0.3f)
        assertEquals(listOf(person), OwnHandFilter.withoutOwnHands(listOf(person), listOf(handBox)))
    }

    @Test
    fun `box rising from the bottom edge without reaching the upper half is lower only`() {
        assertTrue(OwnHandFilter.isLowerOnly(NormalizedBox(0.3f, 0.6f, 0.7f, 1f)))
    }

    @Test
    fun `a standing person is never lower only`() {
        // 全身が写った人、下端に接していても頭が上半分にある人、離れて小さく写った人
        assertFalse(OwnHandFilter.isLowerOnly(NormalizedBox(0.4f, 0.1f, 0.6f, 0.8f)))
        assertFalse(OwnHandFilter.isLowerOnly(NormalizedBox(0.2f, 0.2f, 0.8f, 1f)))
        assertFalse(OwnHandFilter.isLowerOnly(NormalizedBox(0.45f, 0.6f, 0.5f, 0.8f)))
    }

    @Test
    fun `hand memory keeps a box briefly after the hand is lost`() {
        val memory = HandBoxMemory(holdSeconds = 0.7)
        assertEquals(listOf(handBox), memory.update(1.0, listOf(handBox)))
        assertEquals(listOf(handBox), memory.update(1.5, emptyList()))
        assertEquals(emptyList<NormalizedBox>(), memory.update(1.8, emptyList()))
    }

    @Test
    fun `people are kept when no hand is visible`() {
        val person = NormalizedBox(0.4f, 0.5f, 0.6f, 0.7f)
        assertEquals(listOf(person), OwnHandFilter.withoutOwnHands(listOf(person), emptyList()))
    }
}
