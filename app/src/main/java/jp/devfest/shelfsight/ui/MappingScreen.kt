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
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.height
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
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
import jp.devfest.shelfsight.walking.WalkingState

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun MappingScreen(
    state: MapUiState,
    perception: PerceptionState,
    labels: LabelState,
    walking: WalkingState,
    onToggleWalking: () -> Unit,
    /** 学習データの記録枚数。-1 なら記録していない。null ならボタンを出さない。 */
    recordedFrames: Int?,
    onToggleRecordTraining: () -> Unit,
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

        WalkingStrip(
            walking,
            Modifier.align(Alignment.TopCenter).fillMaxWidth().windowInsetsPadding(WindowInsets.safeDrawing),
        )

        Column(
            verticalArrangement = Arrangement.spacedBy(4.dp),
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color(0x99000000))
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(horizontal = 12.dp, vertical = 8.dp),
        ) {
            Text(
                text = perception.ahead,
                color = Color(0xFFFFEB3B),
                fontSize = 20.sp,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
            )
            if (perception.people.isNotEmpty()) {
                Text(text = perception.people, color = Color.White, fontSize = 15.sp)
            }
            if (perception.hands.isNotEmpty()) {
                Text(
                    text = perception.hands,
                    color = Color(0xFF80DEEA),
                    fontSize = 15.sp,
                    modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                )
            }
            WalkingText(walking)
            LabelPanel(labels)
            if (state.status.isNotEmpty()) {
                Text(
                    text = state.status,
                    color = Color.White,
                    fontSize = 16.sp,
                    modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                )
            }
            Text(
                text = String.format(
                    Locale.JAPAN,
                    "面 %d 枚（点から %d 枚）・ %.1f 万点 ・ %.0f m 歩行",
                    state.surfaceCount,
                    state.surfacesFromPoints,
                    state.pointCount / 10_000f,
                    state.walkedMeters,
                ),
                color = Color(0xFFBDBDBD),
                fontSize = 13.sp,
            )
            if (state.showMap || !state.showCamera) {
                val unplaced = if (perception.unplacedPeople > 0) "（位置不明 ${perception.unplacedPeople}人）" else ""
                Text(
                    text = "地図上の人 ${perception.placedPeople}人$unplaced　" +
                        "赤：向かってくる　橙：止まっている　緑：その他　白：確認中",
                    color = Color(0xFFBDBDBD),
                    fontSize = 13.sp,
                )
            }
            if (state.showMap) {
                Text(text = "1本指で回転、2本指で移動と拡大", color = Color(0xFFBDBDBD), fontSize = 13.sp)
            }

            // よく使う 2 つと、設定を開くボタンだけを常に出し、画面をふさがないようにする。
            var showSettings by rememberSaveable { mutableStateOf(false) }
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                SmallButton(if (state.showMap) "今の視点に戻る" else "地図を見る", onToggleMap)
                SmallButton(if (state.isRecording) "記録を一時停止" else "記録を再開", onToggleRecording)
                if (recordedFrames != null && recordedFrames >= 0) {
                    SmallButton("学習データの記録を終了（$recordedFrames 枚）", onToggleRecordTraining)
                }
                SmallButton(if (showSettings) "設定を閉じる" else "設定", { showSettings = !showSettings })
                if (showSettings) {
                    if (!state.showMap) {
                        SmallButton(if (state.showCamera) "地図の表示に切り替え" else "カメラ表示に切り替え", onToggleCamera)
                    }
                    if (recordedFrames != null && recordedFrames < 0 && !state.showMap) {
                        SmallButton("学習データを記録", onToggleRecordTraining)
                    }
                    if (walking.available) {
                        SmallButton(if (walking.enabled) "歩行中の判別をオフ" else "歩行中の判別をオン", onToggleWalking)
                    }
                    if (labels.cloudAvailable) {
                        SmallButton(if (labels.cloudEnabled) "棚のクラウド判別をオフ" else "棚のクラウド判別をオン", onToggleCloud)
                    }
                    if (state.showMap || !state.showCamera) {
                        SmallButton(if (state.showPoints) "面だけを表示" else "点も重ねて表示", onTogglePoints)
                    }
                }
            }
        }
    }
}

/** カテゴリごとの色。並びは tools/coarse_review.py と同じにしてある。 */
private val WALKING_COLORS = listOf(
    Color(0xFF1D78D2), Color(0xFFF09614), Color(0xFF00A0AA), Color(0xFF78553C),
    Color(0xFFE63946), Color(0xFF28A050), Color(0xFF828282), Color(0xFF9646C8),
)

/** 押しやすさを保つため高さは 48dp 以上にしつつ、文字と余白を小さくしたボタン。 */
@Composable
private fun SmallButton(text: String, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
        modifier = Modifier.heightIn(min = 48.dp),
    ) {
        Text(text, fontSize = 14.sp)
    }
}

/** 歩行中の判別結果を、カメラ画像の列と同じ位置に色の帯で重ねる。 */
@Composable
private fun WalkingStrip(walking: WalkingState, modifier: Modifier) {
    if (!walking.enabled || walking.columns.isEmpty()) return
    Row(modifier.height(28.dp)) {
        // 同じカテゴリが続く列はまとめて、1 つの帯に名前を書く。
        runsOf(walking.columns).forEach { (name, count) ->
            val color = WALKING_COLORS[walking.classes.indexOf(name).coerceIn(0, WALKING_COLORS.lastIndex)]
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.weight(count.toFloat()).fillMaxHeight().background(color.copy(alpha = 0.8f)),
            ) {
                Text(name, color = Color.White, fontSize = 12.sp, maxLines = 1)
            }
        }
    }
}

/** 読み上げ用。帯は画面の上にあるので、文字でも左から順に伝える。 */
@Composable
private fun WalkingText(walking: WalkingState) {
    if (!walking.enabled || walking.columns.isEmpty()) return
    Text(
        text = "歩行中：" + runsOf(walking.columns).joinToString(" → ") { it.first } + "（${walking.millis} ms）",
        color = Color.White,
        fontSize = 15.sp,
    )
}

private fun runsOf(columns: List<String>): List<Pair<String, Int>> {
    val runs = mutableListOf<Pair<String, Int>>()
    for (c in columns) {
        val last = runs.lastOrNull()
        if (last?.first == c) runs[runs.lastIndex] = c to last.second + 1 else runs.add(c to 1)
    }
    return runs
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
