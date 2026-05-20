/*
 * ctmv_jni.c — JNI shim built into the Rust cdylib.
 *
 * This file is compiled as part of the Android `catetus_viewer_mobile.so`
 * by the `cargo ndk` build (see `scripts/build-android-jniLibs.sh`). It
 * bridges Kotlin's `ByteArray` / `ByteBuffer` to the C ABI exposed by
 * `crate::ffi`.
 *
 * Status: PENDING compile until the Android build script is wired up to
 * include this in the Rust `cdylib`. The Rust side has all the FFI surface
 * it needs; this is just the JNI marshaling.
 */

#include <jni.h>
#include <stdint.h>
#include <string.h>
#include "catetus_viewer_mobile.h"

JNIEXPORT jlong JNICALL
Java_com_catetus_viewer_CatetusNative_decodeGlb(JNIEnv *env, jobject thiz, jbyteArray bytes) {
    (void)thiz;
    jsize len = (*env)->GetArrayLength(env, bytes);
    jbyte *raw = (*env)->GetByteArrayElements(env, bytes, NULL);
    SfmvBuffer *out = NULL;
    SfmvStatus status = ctmv_decode_glb((const uint8_t *)raw, (size_t)len, &out);
    (*env)->ReleaseByteArrayElements(env, bytes, raw, JNI_ABORT);
    if (status != SfmvStatusOk) return 0;
    return (jlong)(uintptr_t)out;
}

JNIEXPORT jint JNICALL
Java_com_catetus_viewer_CatetusNative_bufferLen(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env; (void)thiz;
    return (jint)ctmv_buffer_len((const SfmvBuffer *)(uintptr_t)handle);
}

JNIEXPORT jint JNICALL
Java_com_catetus_viewer_CatetusNative_vertexStride(JNIEnv *env, jobject thiz) {
    (void)env; (void)thiz;
    return (jint)ctmv_vertex_stride();
}

JNIEXPORT void JNICALL
Java_com_catetus_viewer_CatetusNative_copyVertices(JNIEnv *env, jobject thiz, jlong handle, jobject dst) {
    (void)thiz;
    void *target = (*env)->GetDirectBufferAddress(env, dst);
    if (!target) return;
    const void *src = ctmv_buffer_data((const SfmvBuffer *)(uintptr_t)handle);
    size_t bytes = ctmv_buffer_len((const SfmvBuffer *)(uintptr_t)handle) * ctmv_vertex_stride();
    memcpy(target, src, bytes);
}

JNIEXPORT void JNICALL
Java_com_catetus_viewer_CatetusNative_freeBuffer(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env; (void)thiz;
    ctmv_buffer_free((SfmvBuffer *)(uintptr_t)handle);
}
