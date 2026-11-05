package jp.devfest.shelfsight.ui

import android.opengl.GLSurfaceView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import java.util.Locale
import jp.devfest.shelfsight.labeling.LabelState
import jp.devfest.shelfsight.mapping.MapUiState
import jp.devfest.shelfsight.mapping.OrbitCamera
import jp.devfest.shelfsight.ocr.ProductResult
import jp.devfest.shelfsight.perception.HandView
import jp.devfest.shelfsight.perception.PerceptionState

@Composable
fun MappingScreen(
    state: MapUiState,
    perception: PerceptionState,
    labels: LabelState,
    surfaceView: GLSurfaceView,
    orbit: OrbitCamera,
    onToggleMap: () -> Unit,
    onToggleCamera: () -> Unit,
    onTogglePoints: () -> Unit,
    onToggleCloud: () -> Unit,
    onToggleRecording: () -> Unit,
) {
    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { surfaceView }, modifier = Modifier.fillMaxSize())

        if (state.showMap) {
            Box(Modifier.fillMaxSize().pointerInput(orbit) { orbitGestures(orbit) })
        }

        if (!state.showMap) {
            Canvas(Modifier.fillMaxSize()) {
                perception.personRects.forEach { rect ->
                    drawRect(
                        color = Color(0xFFFFEB3B),
                        topLeft = Offset(rect.left * size.width, rect.top * size.height),
                        size = Size(rect.width * size.width, rect.height * size.height),
                        style = Stroke(width = 6.dp.toPx()),
                    )
                }
                perception.handViews.forEach { hand -> drawHand(hand) }
            }
        }

        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color(0xCC000000))
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(16.dp),
        ) {
            Text(
                text = perception.ahead,
                color = Color(0xFFFFEB3B),
                fontSize = 26.sp,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
            )
            if (perception.people.isNotEmpty()) {
                Text(text = perception.people, color = Color.White, fontSize = 20.sp)
            }
            if (perception.hands.isNotEmpty()) {
                Text(
                    text = perception.hands,
                    color = Color(0xFF80DEEA),
                    fontSize = 20.sp,
                    modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                )
            }
            LabelPanel(labels)
            Text(
                text = state.status,
                color = Color.White,
                fontSize = 22.sp,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
            )
            Text(
                text = String.format(
                    Locale.JAPAN,
                    "面 %d 枚（点から %d 枚）・ %.1f 万点 ・ %.0f m 歩行",
                    state.surfaceCount,
                    state.surfacesFromPoints,
                    state.pointCount / 10_000f,
                    state.walkedMeters,
                ),
                color = Color.White,
                fontSize = 18.sp,
            )
            if (state.showMap || !state.showCamera) {
                val unplaced = if (perception.unplacedPeople > 0) "（位置不明 ${perception.unplacedPeople}人）" else ""
                Text(
                    text = "地図上の人 ${perception.placedPeople}人$unplaced\n" +
                        "赤：向かってくる　橙：止まっている　緑：その他　白：確認中",
                    color = Color.White,
                    fontSize = 16.sp,
                )
            }
            if (state.showMap) {
                Text(
                    text = "1本指で回転、2本指で移動と拡大",
                    color = Color(0xFFBDBDBD),
                    fontSize = 16.sp,
                )
            }
            if (!state.showMap) {
                Button(
                    onClick = onToggleCamera,
                    modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
                ) {
                    Text(if (state.showCamera) "地図の表示に切り替え" else "カメラ表示に切り替え", fontSize = 18.sp)
                }
            }
            if (labels.cloudAvailable && !state.showMap) {
                Button(
                    onClick = onToggleCloud,
                    modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
                ) {
                    Text(if (labels.cloudEnabled) "棚のクラウド判別をオフ" else "棚のクラウド判別をオン", fontSize = 18.sp)
                }
            }
            if (state.showMap || !state.showCamera) {
                Button(
                    onClick = onTogglePoints,
                    modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
                ) {
                    Text(if (state.showPoints) "面だけを表示" else "点も重ねて表示", fontSize = 18.sp)
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = onToggleMap,
                    modifier = Modifier.weight(1f).heightIn(min = 72.dp),
                ) {
                    Text(if (state.showMap) "今の視点に戻る" else "地図を見る", fontSize = 18.sp)
                }
                Button(
                    onClick = onToggleRecording,
                    modifier = Modifier.weight(1f).heightIn(min = 72.dp),
                ) {
                    Text(if (state.isRecording) "記録を一時停止" else "記録を再開", fontSize = 18.sp)
                }
            }
        }
    }
}

@Composable
private fun LabelPanel(labels: LabelState) {
    if (labels.status.isNotEmpty()) {
        Text(text = labels.status, color = Color(0xFFBDBDBD), fontSize = 16.sp)
    }
    labels.product?.let { product ->
        Text(
            text = productText(product),
            color = Color(0xFFA5D6A7),
            fontSize = 22.sp,
            modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
        )
    }
    if (labels.columns.isNotEmpty()) {
        Text(
            text = "棚：" + labels.columns.joinToString(" ／ ") { it.column.category },
            color = Color(0xFFA5D6A7),
            fontSize = 20.sp,
            modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
        )
    }
}

private fun productText(product: ProductResult): String {
    if (!product.readable) return "商品：ラベルを読めません。文字をカメラに向けてください"
    return "商品：${product.name}"
}

/** MediaPipe の 21 点のうち、骨としてつなぐ組み合わせ。 */
private val HAND_BONES = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4,
    0 to 5, 5 to 6, 6 to 7, 7 to 8,
    5 to 9, 9 to 10, 10 to 11, 11 to 12,
    9 to 13, 13 to 14, 14 to 15, 15 to 16,
    13 to 17, 17 to 18, 18 to 19, 19 to 20,
    0 to 17,
)

private fun DrawScope.drawHand(hand: HandView) {
    fun offset(x: Float, y: Float) = Offset(x * size.width, y * size.height)
    HAND_BONES.forEach { (a, b) ->
        val from = hand.points[a]
        val to = hand.points[b]
        drawLine(
            color = Color(0xFF80DEEA),
            start = offset(from.x, from.y),
            end = offset(to.x, to.y),
            strokeWidth = 4.dp.toPx(),
        )
    }
    val from = hand.pointingFrom
    val to = hand.pointingTo
    if (from != null && to != null) {
        drawLine(
            color = Color(0xFFFF5252),
            start = offset(from.x, from.y),
            end = offset(to.x, to.y),
            strokeWidth = 8.dp.toPx(),
        )
    }
    hand.pointingTarget?.let { target ->
        drawCircle(
            color = Color(0xFFFF5252),
            radius = 28.dp.toPx(),
            center = offset(target.x, target.y),
            style = Stroke(width = 8.dp.toPx()),
        )
    }
}

private suspend fun androidx.compose.ui.input.pointer.PointerInputScope.orbitGestures(orbit: OrbitCamera) {
    awaitEachGesture {
        awaitFirstDown(requireUnconsumed = false)
        do {
            val event = awaitPointerEvent()
            val pressed = event.changes.count { it.pressed }
            val pan = event.calculatePan()
            if (pressed == 1) {
                orbit.rotate(pan.x, pan.y)
            } else if (pressed >= 2) {
                orbit.zoom(event.calculateZoom())
                orbit.pan(pan.x, pan.y)
            }
            event.changes.forEach { if (it.positionChanged()) it.consume() }
        } while (event.changes.any { it.pressed })
    }
}
