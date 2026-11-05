package jp.devfest.shelfsight

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import jp.devfest.shelfsight.walking.CoarseClassifier
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 実機で歩行中の判別を動かし、PC と同じ結果になるかと、1 回にかかる時間を測る。
 * 写真には店内が写るのでリポジトリには入れず、端末に置いて使う（tools/coarse_device_check.py 参照）。
 *
 *   adb logcat -s CoarseDeviceTest   で時間を見る
 */
@RunWith(AndroidJUnit4::class)
class CoarseClassifierDeviceTest {
    @Test
    fun givesTheSameColumnsAsThePcAndReportsTheTime() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = context.getExternalFilesDir(null)
        val photo = File(dir, "coarse_test.jpg")
        val expectedFile = File(dir, "coarse_test.expected.json")
        assumeTrue("no test photo on the device", photo.exists() && expectedFile.exists())
        val expected = JSONObject(expectedFile.readText())
        val image = BitmapFactory.decodeFile(photo.path)

        val models = context.assets.list("coarse").orEmpty().filter { it.endsWith(".onnx") }.map { it.removeSuffix(".onnx") }
        assumeTrue("no model in the assets", models.isNotEmpty())
        for (name in models) {
            val loadStart = System.currentTimeMillis()
            val classifier = CoarseClassifier.fromAssets(context, name)!!
            val loadMillis = System.currentTimeMillis() - loadStart
            classifier.use { model ->
                model.classify(image) // 1 回目は準備が入るので測らない
                val runs = (1..RUNS).map { model.classify(image) }
                val result = runs.last()
                val prep = runs.map { it.preprocessMillis }.sorted()[RUNS / 2]
                val infer = runs.map { it.inferenceMillis }.sorted()[RUNS / 2]
                val want = expected.getJSONArray(name).let { a -> List(a.length()) { a.getString(it) } }
                val same = result.columns.zip(want).count { (a, b) -> a == b }
                Log.i(TAG, "$name load ${loadMillis}ms, median prep ${prep}ms + model ${infer}ms; got ${result.columns} want $want")
                // 画像の縮小方法が PC と少し違うので、迷う列が 1 つ変わるくらいは許す。
                assertTrue("$name: only $same of ${want.size} columns match the PC", same >= want.size - 1)
            }
        }
        image.recycle()
    }

    private companion object {
        const val TAG = "CoarseDeviceTest"
        const val RUNS = 5
    }
}
