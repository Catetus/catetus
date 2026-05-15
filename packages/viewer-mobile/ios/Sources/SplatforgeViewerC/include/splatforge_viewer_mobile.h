/*
 * splatforge_viewer_mobile.h
 *
 * Hand-checked-in copy of the cbindgen output. Regenerate from the Rust crate
 * with `scripts/regen-headers.sh` whenever the FFI surface changes.
 *
 * Status: PENDING — cbindgen runs locally; this file is the stable interface
 * Swift compiles against. The Rust side must keep these symbols in sync.
 */
#ifndef SPLATFORGE_VIEWER_MOBILE_H
#define SPLATFORGE_VIEWER_MOBILE_H

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SfmvStatus {
    SfmvStatusOk = 0,
    SfmvStatusDecodeFailed = -1,
    SfmvStatusEmpty = -2,
    SfmvStatusNullPointer = -3,
} SfmvStatus;

typedef struct SfmvBuffer SfmvBuffer;

SfmvStatus sfmv_decode_glb(const uint8_t *bytes, size_t len, SfmvBuffer **out);
const void *sfmv_buffer_data(const SfmvBuffer *buf);
size_t sfmv_buffer_len(const SfmvBuffer *buf);
size_t sfmv_vertex_stride(void);
void sfmv_buffer_free(SfmvBuffer *buf);
int sfmv_sort_by_depth(const SfmvBuffer *buf, const float *view_col_major, uint32_t *out_indices);

#ifdef __cplusplus
}
#endif

#endif /* SPLATFORGE_VIEWER_MOBILE_H */
