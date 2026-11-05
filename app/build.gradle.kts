plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
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
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    androidResources {
        noCompress += listOf("tflite", "task") // MediaPipe がモデルを直接読み込めるようにする
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
    implementation("org.tensorflow:tensorflow-lite:2.17.0") // Store3D

    testImplementation(libs.junit)
    testImplementation(libs.org.json)
}
