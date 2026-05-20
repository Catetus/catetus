// Android module that publishes the `CatetusViewer` AAR.
//
// The native library is built separately by `scripts/build-android-jniLibs.sh`
// which invokes `cargo ndk` against `packages/viewer-mobile/core/`. The output
// `.so` files land in `src/main/jniLibs/<abi>/libcatetus_viewer_mobile.so`.
// This file declares them as a normal jniLibs source set so they ship in the
// AAR with no extra plumbing.

plugins {
    id("com.android.library") version "8.4.0"
    id("org.jetbrains.kotlin.android") version "1.9.24"
}

android {
    namespace = "com.catetus.viewer"
    compileSdk = 34
    defaultConfig {
        minSdk = 24
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }
    buildFeatures {
        prefab = false
    }
    sourceSets["main"].jniLibs.srcDirs("src/main/jniLibs")
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.annotation:annotation:1.8.0")
}
