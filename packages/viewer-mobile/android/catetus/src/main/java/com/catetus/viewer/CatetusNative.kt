// CatetusNative — thin JNI binding to the Rust C ABI.
//
// The companion Rust crate publishes a `cdylib`; `scripts/build-android-jniLibs.sh`
// invokes `cargo ndk` to drop one `.so` per ABI into
// `src/main/jniLibs/<abi>/libcatetus_viewer_mobile.so`.
//
// We do NOT use the C ABI directly from JNI — instead we hand-write a tiny
// `jni/ctmv_jni.c` shim that converts `jbyteArray` → `(const uint8_t*, size_t)`
// and back. That shim is part of the cdylib build so it ships in the same .so.

package com.catetus.viewer

import androidx.annotation.Keep

/** Static binding to the JNI shim. Loaded lazily on first viewer instance. */
@Keep
internal object CatetusNative {
    init {
        System.loadLibrary("catetus_viewer_mobile")
    }

    /** Decode a `.glb` blob. Returns an opaque buffer handle, or 0 on failure. */
    external fun decodeGlb(bytes: ByteArray): Long

    /** Number of `SplatVertex` entries in `handle`. */
    external fun bufferLen(handle: Long): Int

    /** Stride of one `SplatVertex` (always 56 bytes — we ask the Rust side to
     *  stay authoritative). */
    external fun vertexStride(): Int

    /** Copy `count` vertices starting at index 0 into `dst` (which must be at
     *  least `count * vertexStride()` bytes). */
    external fun copyVertices(handle: Long, dst: java.nio.ByteBuffer)

    /** Free a buffer returned by [decodeGlb]. Safe to call with 0. */
    external fun freeBuffer(handle: Long)
}
