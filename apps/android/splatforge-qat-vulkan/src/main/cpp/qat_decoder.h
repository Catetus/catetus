//
// qat_decoder.h — C++ entry points for the SplatForge Vulkan QAT-PLY
// decoder. Hidden behind a JNI bridge (jni_bridge.cpp) for Kotlin/Java
// callers; exposed directly to native callers for tests.
//
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstdint>

namespace splatforge {

// Dispatch the int8 dequant kernel. q is row-major (n_rows * n_channels)
// signed int8. scale is n_channels fp32. out receives n_rows*n_channels
// row-major fp32. Throws std::runtime_error on Vulkan failure.
void decode_int8(const int8_t *q, const float *scale,
                 uint32_t n_rows, uint32_t n_channels, float *out);

// Dispatch the int4-packed dequant kernel. packed is n_rows * ceil(C/2)
// bytes. scale is n_rows fp32 (per-anchor). out receives n_rows*n_channels
// fp32.
void decode_int4_packed(const uint8_t *packed, const float *scale,
                        uint32_t n_rows, uint32_t n_channels, float *out);

}  // namespace splatforge
