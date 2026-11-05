package jp.devfest.shelfsight

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import jp.devfest.shelfsight.ocr.MlKitOcr
import jp.devfest.shelfsight.ocr.ProductReading
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 実機の OCR で商品の写真を読む。写真には人や部屋が写るのでリポジトリには入れず、端末に置いて使う。
 *
 *   adb push <写真> /sdcard/Android/data/jp.devfest.shelfsight/files/ocr_test.jpg
 *   ./gradlew :app:connectedDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class ProductOcrDeviceTest {
    @Test
    fun readsTheProductNameFromAPhotoOfAHeldBox() {
        val dir = InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null)
        val photo = File(dir, "ocr_test.jpg")
        assumeTrue("no test photo on the device", photo.exists())

        val bitmap = BitmapFactory.decodeFile(photo.path)
        val start = System.currentTimeMillis()
        val lines = MlKitOcr().use { it.read(bitmap) }
        val millis = System.currentTimeMillis() - start
        val product = ProductReading.from(lines)
        Log.i("ProductOcrDeviceTest", "ms=$millis name=${product.name} lines=${lines.map { "${it.text}@${it.height}" }}")
        // 箱のロゴの L が前につくことがあるが、ロゴも印刷された文字なので誤りとはしない。
        assertTrue(product.name, product.name.contains("2種のぶどうフルーツバー"))
    }
}
