// build.gradle.kts — Android library module for the Catetus QAT-PLY
// v1 Vulkan decoder. Loads libcatetus_qat.so (built by CMake) and
// exposes the dev.catetus.qat.QATPlyDecoder Kotlin object.
//
// Requires:
//   - Android Gradle Plugin 8.4+
//   - NDK 26+ (ships glslc)
//   - target SDK 34
//
// Install: `brew install --cask android-studio`, then open the Catetus
// root in Android Studio so it can sync this module.
//
// License: MIT.

plugins {
    id("com.android.library") version "8.4.0"
    kotlin("android") version "1.9.23"
}

android {
    namespace = "dev.catetus.qat"
    compileSdk = 34
    ndkVersion = "26.2.11394342"

    defaultConfig {
        minSdk = 26
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-DCMAKE_BUILD_TYPE=Release"
                )
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    sourceSets {
        getByName("main") {
            kotlin.srcDirs("src/main/kotlin")
        }
        getByName("test") {
            kotlin.srcDirs("src/test/kotlin")
        }
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
