//
// jni_bridge.cpp — JNI bridge from dev.catetus.qat.QATPlyDecoder into
// the Vulkan decoder. Marshals Java arrays into native pointers and
// calls into catetus::decode_int8 / decode_int4_packed.
//
// SPDX-License-Identifier: MIT
//

#include <jni.h>
#include <vector>
#include <string>
#include <stdexcept>

#include "qat_decoder.h"

extern "C" {

JNIEXPORT void JNICALL
Java_dev_catetus_qat_QATPlyDecoder_nativeDecodeInt8(
    JNIEnv *env, jclass /*clazz*/,
    jbyteArray jq, jfloatArray jscale,
    jint n_rows, jint n_channels,
    jfloatArray jout
) {
    jsize q_len = env->GetArrayLength(jq);
    jsize s_len = env->GetArrayLength(jscale);
    jsize o_len = env->GetArrayLength(jout);
    if (q_len < (jsize)(n_rows * n_channels) ||
        s_len < (jsize)n_channels ||
        o_len < (jsize)(n_rows * n_channels)) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "QAT int8: array size mismatch");
        return;
    }
    std::vector<int8_t> q((size_t)n_rows * n_channels);
    env->GetByteArrayRegion(jq, 0, n_rows * n_channels, (jbyte *)q.data());
    std::vector<float> s((size_t)n_channels);
    env->GetFloatArrayRegion(jscale, 0, n_channels, s.data());
    std::vector<float> out((size_t)n_rows * n_channels);
    try {
        catetus::decode_int8(q.data(), s.data(),
                                (uint32_t)n_rows, (uint32_t)n_channels,
                                out.data());
    } catch (const std::exception &e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return;
    }
    env->SetFloatArrayRegion(jout, 0, n_rows * n_channels, out.data());
}

JNIEXPORT void JNICALL
Java_dev_catetus_qat_QATPlyDecoder_nativeDecodeInt4Packed(
    JNIEnv *env, jclass /*clazz*/,
    jbyteArray jpacked, jfloatArray jscale,
    jint n_rows, jint n_channels,
    jfloatArray jout
) {
    jint B = (n_channels + 1) >> 1;
    jsize p_len = env->GetArrayLength(jpacked);
    jsize s_len = env->GetArrayLength(jscale);
    jsize o_len = env->GetArrayLength(jout);
    if (p_len < (jsize)(n_rows * B) ||
        s_len < (jsize)n_rows ||
        o_len < (jsize)(n_rows * n_channels)) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "QAT int4: array size mismatch");
        return;
    }
    std::vector<uint8_t> packed((size_t)n_rows * B);
    env->GetByteArrayRegion(jpacked, 0, n_rows * B, (jbyte *)packed.data());
    std::vector<float> scale((size_t)n_rows);
    env->GetFloatArrayRegion(jscale, 0, n_rows, scale.data());
    std::vector<float> out((size_t)n_rows * n_channels);
    try {
        catetus::decode_int4_packed(packed.data(), scale.data(),
                                       (uint32_t)n_rows, (uint32_t)n_channels,
                                       out.data());
    } catch (const std::exception &e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return;
    }
    env->SetFloatArrayRegion(jout, 0, n_rows * n_channels, out.data());
}

}  // extern "C"
