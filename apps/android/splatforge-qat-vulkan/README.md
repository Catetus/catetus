# splatforge-qat-vulkan

Android Vulkan compute decoder for the [SplatForge QAT-PLY v1](../../codec/qat-ply-c/)
wire format. Drop-in Gradle module for Android renderers.

- Two specialized pipelines from a single SPIR-V module: int8 +
  per-channel scale and int4-packed + per-anchor scale.
- One workgroup per (anchor × channel) tile; aligns reads to 4-byte words
  for coalesced access on Mali / Adreno / Imagination GPUs.
- JNI bridge marshals Kotlin `ByteArray` / `FloatArray` directly into
  Vulkan storage buffers.

## Prerequisites

Install before building:

```bash
brew install --cask android-studio        # Android SDK + Gradle
# Inside Android Studio: SDK Manager > NDK (Side by side) 26.2+ and CMake 3.22+
# These ship glslc, libvulkan, and the cross-toolchain.
```

Standalone glslc for host-side experimentation:

```bash
brew install glslang                      # provides glslc
```

## Build

From the SplatForge root with Android Studio attached:

```bash
./gradlew :apps:android:splatforge-qat-vulkan:assembleDebug
```

Or syntactic-only CMake validation (no compile):

```bash
cd apps/android/splatforge-qat-vulkan
cmake -B build                             # configure step alone
```

## Usage (Kotlin)

```kotlin
import dev.splatforge.qat.QATPlyDecoder

val fp32 = QATPlyDecoder.decodeInt8(
    q       = readInt8Columns(),       // ByteArray of nRows*nChannels
    scale   = readPerChannelScales(),  // FloatArray of nChannels
    nRows   = anchorCount,
    nChannels = 32
)
// fp32 is a row-major FloatArray of length nRows*nChannels.
```

For `int4` fields, call `decodeInt4Packed` and pass per-anchor scales
(one fp32 per row) instead.

## Verifying conformance

The repo's cross-target conformance suite at
`apps/codec/conformance/cross-target/` declares the Vulkan target. To
run it locally requires a Vulkan-capable device (or `lavapipe`/SwiftShader
on the host). When no Vulkan implementation is available the suite
records the row as "skipped — not validated on this host" rather than
failing.

## Library size

The compiled `libsplatforge_qat.so` (arm64-v8a, release) is ~95 KB
including the loader stub. The embedded SPIR-V is ~3 KB.

## License

MIT.
