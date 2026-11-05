package jp.devfest.shelfsight.ui

import android.opengl.GLSurfaceView
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import java.util.Locale
import jp.devfest.shelfsight.mapping.MapUiState
import jp.devfest.shelfsight.mapping.OrbitCamera

@Composable
fun MappingScreen(
    state: MapUiState,
    surfaceView: GLSurfaceView,
    orbit: OrbitCamera,
    onToggleMap: () -> Unit,
    onToggleRecording: () -> Unit,
) {
    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { surfaceView }, modifier = Modifier.fillMaxSize())

        if (state.showMap) {
            Box(Modifier.fillMaxSize().pointerInput(orbit) { orbitGestures(orbit) })
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
                text = state.status,
                color = Color.White,
                fontSize = 22.sp,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
            )
            Text(
                text = String.format(
                    Locale.JAPAN,
                    "%.1f 万点 ・ %.0f m 歩行",
                    state.pointCount / 10_000f,
                    state.walkedMeters,
                ),
                color = Color.White,
                fontSize = 18.sp,
            )
            if (state.showMap) {
                Text(
                    text = "1本指で回転、2本指で移動と拡大",
                    color = Color(0xFFBDBDBD),
                    fontSize = 16.sp,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = onToggleMap,
                    modifier = Modifier.weight(1f).heightIn(min = 72.dp),
                ) {
                    Text(if (state.showMap) "カメラに戻る" else "地図を見る", fontSize = 18.sp)
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
