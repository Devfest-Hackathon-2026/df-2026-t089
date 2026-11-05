import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

// API キーは local.properties（コミットされない）に書く。公開リポジトリに載せないこと。
val localProperties = Properties().apply {
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
}

android {
    namespace = "jp.devfest.shelfsight"
    compileSdk = 36

    defaultConfig {
        applicationId = "jp.devfest.shelfsight"
        minSdk = 31 // Store3D の要件に合わせる
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        buildConfigField("String", "GEMINI_API_KEY", "\"${localProperties.getProperty("GEMINI_API_KEY", "")}\"")
        buildConfigField(
            "String",
            "GEMINI_MODELS",
            "\"${localProperties.getProperty("GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite")}\"",
        )
        // 商品の OCR に使う Cloud Vision API のキー（GCP_API_KEY の名前でもよい）。なければ端末内の ML Kit で読む。
        val visionKey = localProperties.getProperty("CLOUD_VISION_API_KEY") ?: localProperties.getProperty("GCP_API_KEY", "")
        buildConfigField("String", "CLOUD_VISION_API_KEY", "\"$visionKey\"")
        // 歩行中の判別に使うモデル。app/src/main/assets/coarse/<名前>.onnx と <名前>.head.json を置く（tools/coarse_experiment.py 参照）。
        buildConfigField("String", "COARSE_MODEL", "\"${localProperties.getProperty("COARSE_MODEL", "MobileCLIP2-S0")}\"")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    androidResources {
        noCompress += listOf("tflite", "task", "onnx") // モデルを圧縮せずに入れ、そのまま読めるようにする
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.arcore)
    implementation(libs.litertlm)
    implementation(libs.mediapipe.tasks.vision)
    implementation(libs.mlkit.text.japanese)
    implementation(libs.onnxruntime.android)
    implementation("org.tensorflow:tensorflow-lite:2.17.0") // Store3D

    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.junit)

    testImplementation(libs.junit)
    testImplementation(libs.org.json)
}
