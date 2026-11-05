plugins {
    id("com.android.application")
}

android {
    namespace = "com.example.store3d"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.example.store3d"
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
    }
    androidResources { noCompress += "tflite" }
}

dependencies {
    implementation("com.google.ar:core:1.50.0")
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    testImplementation("junit:junit:4.13.2")
}
