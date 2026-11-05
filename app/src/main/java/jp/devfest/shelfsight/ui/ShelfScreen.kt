package jp.devfest.shelfsight.ui

import android.opengl.GLSurfaceView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import jp.devfest.shelfsight.scan.ScanUiState

@Composable
fun ShelfScreen(
    state: ScanUiState,
    surfaceView: GLSurfaceView,
    onTogglePaused: () -> Unit,
) {
    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { surfaceView }, modifier = Modifier.fillMaxSize())

        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color(0xCC000000))
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(16.dp),
        ) {
            if (state.isDemo) {
                Text(
                    text = "デモモード：モデル未配置のため、結果はダミーです",
                    color = Color(0xFFFFD54F),
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                )
            }
            Text(
                text = state.status,
                color = Color.White,
                fontSize = 22.sp,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
            )
            if (state.found.isNotEmpty()) {
                Text(
                    text = "見つけた売り場：" + state.found.joinToString("、") { it.labelJa },
                    color = Color.White,
                    fontSize = 20.sp,
                    modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                )
            }
            Button(
                onClick = onTogglePaused,
                modifier = Modifier.fillMaxWidth().heightIn(min = 72.dp),
            ) {
                Text(
                    text = if (state.isPaused) "判定を再開" else "判定を一時停止",
                    style = MaterialTheme.typography.titleLarge,
                )
            }
        }
    }
}
