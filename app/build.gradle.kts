plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "org.sensevoice.benchmark"
    compileSdk = 35
    ndkVersion = "28.2.13676358"
    defaultConfig {
        applicationId = "org.sensevoice.benchmark.precision"
        minSdk = 26
        targetSdk = 35
        versionCode = 4
        versionName = "0.4.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    // Two native variants: baseline runs on all arm64 phones; aggressive adds i8mm/bf16 for
    // Snapdragon 8 Elite Gen 5-class CPUs and may fault (SIGILL) on older devices.
    flavorDimensions += "cpu"
    productFlavors {
        create("baseline") {
            dimension = "cpu"
            applicationIdSuffix = ".baseline"
            versionNameSuffix = "-baseline"
            resValue("string", "app_name", "SenseVoice Precision Baseline")
        }
        create("aggressive") {
            dimension = "cpu"
            applicationIdSuffix = ".aggressive"
            versionNameSuffix = "-aggressive"
            resValue("string", "app_name", "SenseVoice Precision ARMv8.6")
        }
    }
    buildFeatures { compose = true; buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildTypes { release { isMinifyEnabled = false } }
    packaging { jniLibs { useLegacyPackaging = false } }
    sourceSets {
        getByName("baseline") { jniLibs.srcDir("src/main/jniLibs") }
        getByName("aggressive") { jniLibs.srcDir("src/main/jniLibs-aggressive") }
    }
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
        listOf("baseline" to "src/main/jniLibs", "aggressive" to "src/main/jniLibs-aggressive").forEach { (flavor, dir) ->
            listOf("benchmark_sherpa", "benchmark_mnn", "benchmark_ncnn", "benchmark_features", "benchmark_gguf", "c++_shared").forEach { name ->
                check(file("$dir/arm64-v8a/lib$name.so").isFile) {
                    "Missing $flavor CPU runtime lib$name.so. Run python tools/build_native.py --variant $flavor first."
                }
            }
        }
    }
}
tasks.named("preBuild") { dependsOn(checkCpuRuntimes) }
