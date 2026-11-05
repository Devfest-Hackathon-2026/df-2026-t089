package jp.devfest.shelfsight.scan

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import jp.devfest.shelfsight.BuildConfig
import jp.devfest.shelfsight.model.ProductCategory
import jp.devfest.shelfsight.vision.DemoShelfClassifier
import jp.devfest.shelfsight.vision.LiteRtGemmaClassifier
import jp.devfest.shelfsight.vision.ShelfClassifier
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ScanUiState(
    val status: String = "準備しています",
    val isDemo: Boolean = false,
    val isReady: Boolean = false,
    val isPaused: Boolean = false,
    val found: List<ProductCategory> = emptyList(),
)

class ShelfScanViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow(ScanUiState())
    val state: StateFlow<ScanUiState> = _state.asStateFlow()

    private val classifier: ShelfClassifier?

    /** モデルが使えない場合は null。 */
    val scanner: ShelfScanner?

    init {
        val modelFile = File(application.getExternalFilesDir(null), MODEL_FILE_NAME)
        classifier = when {
            modelFile.isFile -> LiteRtGemmaClassifier(modelFile.path, application.cacheDir.path)
            BuildConfig.DEBUG -> DemoShelfClassifier().also {
                _state.update { s -> s.copy(isDemo = true) }
            }
            else -> null
        }
        scanner = classifier?.let {
            ShelfScanner(
                classifier = it,
                scope = viewModelScope,
                onFailure = { setStatus("判定に失敗しました。もう一度試します") },
                onScanFinished = { setStatus("棚を映してください") },
            )
        }
        if (classifier == null) {
            setStatus("Gemma のモデルが見つかりません")
        } else {
            prepare(classifier)
        }
    }

    private fun prepare(classifier: ShelfClassifier) {
        viewModelScope.launch {
            setStatus("Gemma を読み込んでいます")
            try {
                classifier.prepare()
                scanner?.enabled = !_state.value.isPaused
                _state.update { it.copy(isReady = true, status = "棚を映してください") }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model", e)
                setStatus("Gemma を読み込めませんでした")
            }
        }
    }

    fun togglePaused() {
        val paused = !_state.value.isPaused
        scanner?.enabled = !paused && _state.value.isReady
        _state.update { it.copy(isPaused = paused) }
    }

    fun onLabelPlaced(category: ProductCategory) {
        _state.update {
            if (category in it.found) it else it.copy(found = it.found + category)
        }
    }

    fun setStatus(message: String) {
        _state.update { it.copy(status = message) }
    }

    override fun onCleared() {
        classifier?.close()
    }

    companion object {
        const val MODEL_FILE_NAME = "gemma-4-E2B-it.litertlm"
        private const val TAG = "ShelfScanViewModel"
    }
}
