package jp.devfest.shelfsight.assist

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.exp

class CategoryHeadTest {
    @Test fun `applies both layers with relu and softmax`() {
        val head = CategoryHead(2, 2, 2, floatArrayOf(
            1f, 0f,    // 中間 0 = x0
            0f, -1f,   // 中間 1 = -x1（x1 が正なら ReLU で 0）
            0f, 0f,    // 1 層目のバイアス
            2f, 0f,    // 出力 0 = 2 × 中間 0 + 0.5
            0f, 3f,    // 出力 1 = 3 × 中間 1
            0.5f, 0f,  // 2 層目のバイアス
        ))
        val (index, prob) = head.predict(floatArrayOf(1f, 1f))
        assertEquals(0, index)
        assertEquals(1f / (1f + exp(-2.5f)), prob, 1e-5f)
        assertEquals(1, head.predict(floatArrayOf(0f, -1f)).first)
    }
}
