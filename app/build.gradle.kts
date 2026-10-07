plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "org.sensevoice.benchmark"
    compileSdk = 35
    defaultConfig {
        applicationId = "org.sensevoice.benchmark"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    buildFeatures { compose = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildTypes { release { isMinifyEnabled = false } }
    packaging { jniLibs { useLegacyPackaging = false } }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.04.01"))
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.10.0")
    implementation("androidx.documentfile:documentfile:1.0.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")
    implementation("com.google.ai.edge.litert:litert:2.1.6")
}

val checkCpuRuntimes by tasks.registering {
    doLast {
        listOf("benchmark_sherpa", "benchmark_mnn", "benchmark_ncnn", "benchmark_features", "c++_shared").forEach { name ->
            check(file("src/main/jniLibs/arm64-v8a/lib$name.so").isFile) {
                "Missing CPU runtime lib$name.so. Run python tools/build_native.py first."
            }
        }
    }
}
tasks.named("preBuild") { dependsOn(checkCpuRuntimes) }
