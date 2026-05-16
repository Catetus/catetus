/*
 * qat_ply_decode.h — Reference C99 decoder for the SplatForge QAT-PLY v1
 * on-disk format. See /specs/qat-ply-v1 for the bit-level specification.
 *
 * Zero dependencies beyond <stdint.h>, <stddef.h>, and <string.h>. The
 * library never calls malloc or any I/O; all output buffers are caller-
 * owned. This file is intentionally a single self-contained header +
 * one .c so renderers can vendor it directly.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef QAT_PLY_DECODE_H
#define QAT_PLY_DECODE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version of this header. Bump when wire format adds capabilities. */
#define QAT_PLY_DECODE_ABI 1

/* Maximum number of quantized_field declarations we will parse from a
 * single PLY header. Two is enough for Scaffold-GS (f_anchor_feat,
 * f_offset) and one for vanilla 3DGS (f_rest); eight leaves headroom. */
#define QAT_PLY_MAX_FIELDS 8

/* Maximum length of a quantized-field name (excluding NUL). */
#define QAT_PLY_MAX_NAME 63

typedef enum {
    QAT_PLY_OK = 0,
    QAT_PLY_ERR_BAD_HEADER = -1,
    QAT_PLY_ERR_BAD_BASE64 = -2,
    QAT_PLY_ERR_BUFFER_TOO_SMALL = -3,
    QAT_PLY_ERR_UNKNOWN_DTYPE = -4,
    QAT_PLY_ERR_CHANNEL_MISMATCH = -5,
    QAT_PLY_ERR_NULL_ARG = -6
} qat_ply_status_t;

typedef enum {
    QAT_PLY_DTYPE_INT8 = 8,
    QAT_PLY_DTYPE_INT4 = 4
} qat_ply_dtype_t;

typedef enum {
    /* scale_b64=... is present; scales are per-channel and embedded in
     * the header marker as base64-encoded little-endian fp32 array. */
    QAT_PLY_SCALE_PER_CHANNEL = 0,
    /* The per-anchor scale lives in a separate PLY column (named
     * "<field>_scale" by convention, e.g. f_offset_scale). The header
     * marker carries scale_kind=per_anchor and no scale_b64. */
    QAT_PLY_SCALE_PER_ANCHOR = 1
} qat_ply_scale_kind_t;

typedef struct {
    /* NUL-terminated field name, e.g. "f_anchor_feat" or "f_offset". */
    char name[QAT_PLY_MAX_NAME + 1];
    /* QAT_PLY_DTYPE_INT8 or QAT_PLY_DTYPE_INT4. */
    qat_ply_dtype_t dtype;
    /* Logical channel count C. For int8, C uchar columns named
     * <name>_q_<i> appear in the PLY body. For int4, ceil(C/2) packed
     * uchar columns appear (low nibble = channel 2i, high = 2i+1). */
    uint32_t channels;
    /* QAT_PLY_SCALE_PER_CHANNEL or QAT_PLY_SCALE_PER_ANCHOR. */
    qat_ply_scale_kind_t scale_kind;
    /* For int4, always 2 in v1. For int8, ignored. */
    uint8_t packed_per_byte;
    /* Offset and length of the scale_b64 substring inside the original
     * header buffer the caller passed to qat_ply_parse_header. The
     * caller can pass that slice to qat_ply_base64_decode_fp32 to
     * recover the per-channel scale array. Set to (0, 0) when
     * scale_kind == QAT_PLY_SCALE_PER_ANCHOR. */
    size_t scale_b64_offset;
    size_t scale_b64_len;
} qat_ply_field_t;

typedef struct {
    qat_ply_field_t fields[QAT_PLY_MAX_FIELDS];
    uint32_t n_fields;
} qat_ply_header_t;

/* ---------------------------------------------------------------------
 * qat_ply_parse_header
 *
 * Scan a PLY ASCII header (from the start of "ply\n" up to and including
 * "end_header\n") looking for "comment quantized_field ..." lines and
 * populate `out` with one qat_ply_field_t per match.
 *
 * `header` is a pointer to the start of the PLY header buffer. `len` is
 * its length in bytes. The function does not mutate the buffer.
 *
 * Returns QAT_PLY_OK on success (including the case where zero matching
 * lines are found — out->n_fields is set to 0).
 * ------------------------------------------------------------------- */
int qat_ply_parse_header(const char *header,
                         size_t len,
                         qat_ply_header_t *out);

/* ---------------------------------------------------------------------
 * qat_ply_dequant_int8
 *
 * Dequantize a contiguous block of int8 values using per-channel fp32
 * scales. Layout is row-major: (n_rows, n_channels). For each (r, c):
 *     out[r * n_channels + c] = (float)q[r * n_channels + c] * scale[c]
 * ------------------------------------------------------------------- */
int qat_ply_dequant_int8(const int8_t *q,
                         const float *scale,
                         size_t n_rows,
                         size_t n_channels,
                         float *out);

/* ---------------------------------------------------------------------
 * qat_ply_dequant_int4_packed
 *
 * Dequantize a contiguous block of int4 values packed two-per-byte using
 * per-anchor fp32 scales. The on-disk representation is unsigned-shifted
 * (signed_q + 8 fits in [0, 15]); this function performs the reverse
 * shift implicitly.
 *
 *   For each row r and channel c:
 *     byte_idx = c / 2
 *     nibble = (c % 2 == 0) ? (packed[r*B + byte_idx] & 0x0F)
 *                           : ((packed[r*B + byte_idx] >> 4) & 0x0F)
 *     signed_q = (int)nibble - 8                       // [-8, 7]
 *     out[r * n_channels + c] = (float)signed_q * scale[r]
 *
 *   where B = ceil(n_channels / 2).
 *
 * `packed` points to n_rows * ceil(n_channels / 2) bytes. `scale` is a
 * per-row scale array of length n_rows. `out` receives n_rows *
 * n_channels fp32 values.
 *
 * Note: each on-disk byte is the uint8 reinterpretation of an int8
 * column (PLY type 'i1' / 'uchar' / 'char' all alias the same byte).
 * Callers passing `int8_t *` should cast to `const uint8_t *`.
 * ------------------------------------------------------------------- */
int qat_ply_dequant_int4_packed(const uint8_t *packed,
                                const float *scale,
                                size_t n_rows,
                                size_t n_channels,
                                float *out);

/* ---------------------------------------------------------------------
 * qat_ply_base64_decode_fp32
 *
 * Decode an RFC 4648 standard-alphabet base64 string into a sequence of
 * little-endian fp32 values. The encoded byte count must be a multiple
 * of 4 (one fp32 per 4 bytes). Whitespace inside the input string is
 * NOT accepted — pass the trimmed substring exactly.
 *
 * `b64` points to `b64_len` ASCII characters.
 * `out` receives up to `out_cap` floats; the function will not write
 * past out + out_cap.
 *
 * On success returns the number of floats written (always >= 0).
 * On failure returns a negative qat_ply_status_t error code.
 * ------------------------------------------------------------------- */
int qat_ply_base64_decode_fp32(const char *b64,
                               size_t b64_len,
                               float *out,
                               size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* QAT_PLY_DECODE_H */
