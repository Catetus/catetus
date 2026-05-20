/*
 * test_qat_ply_decode.c — round-trip tests for the reference C decoder.
 *
 * Each test encodes a known {int8, scale} pair via the same procedure
 * the Catetus trainer uses (per-channel scales serialized as little-
 * endian fp32 then base64) and asserts that the decoder reconstructs
 * bit-exact float values.
 *
 * SPDX-License-Identifier: MIT
 */

#include "../qat_ply_decode.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failed = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); \
        g_failed = 1;                                               \
    }                                                               \
} while (0)

/* Tiny standalone base64 encoder (RFC 4648 std) for test-side encoding.
 * NOT exported; the library is decode-only on purpose so renderers can
 * vendor a minimal surface. */
static const char b64_alpha[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static size_t b64_encode(const uint8_t *in, size_t n, char *out) {
    size_t o = 0;
    size_t i = 0;
    while (i + 3 <= n) {
        uint32_t t = ((uint32_t)in[i] << 16) | ((uint32_t)in[i+1] << 8) | in[i+2];
        out[o++] = b64_alpha[(t >> 18) & 0x3F];
        out[o++] = b64_alpha[(t >> 12) & 0x3F];
        out[o++] = b64_alpha[(t >> 6) & 0x3F];
        out[o++] = b64_alpha[t & 0x3F];
        i += 3;
    }
    size_t rem = n - i;
    if (rem == 1) {
        uint32_t t = (uint32_t)in[i] << 16;
        out[o++] = b64_alpha[(t >> 18) & 0x3F];
        out[o++] = b64_alpha[(t >> 12) & 0x3F];
        out[o++] = '=';
        out[o++] = '=';
    } else if (rem == 2) {
        uint32_t t = ((uint32_t)in[i] << 16) | ((uint32_t)in[i+1] << 8);
        out[o++] = b64_alpha[(t >> 18) & 0x3F];
        out[o++] = b64_alpha[(t >> 12) & 0x3F];
        out[o++] = b64_alpha[(t >> 6) & 0x3F];
        out[o++] = '=';
    }
    out[o] = '\0';
    return o;
}

/* Pack a fp32 array into little-endian bytes (portable). */
static void fp32_to_le_bytes(const float *in, size_t n, uint8_t *out) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &in[i], 4);
        out[4*i+0] = (uint8_t)(bits & 0xFFu);
        out[4*i+1] = (uint8_t)((bits >> 8) & 0xFFu);
        out[4*i+2] = (uint8_t)((bits >> 16) & 0xFFu);
        out[4*i+3] = (uint8_t)((bits >> 24) & 0xFFu);
    }
}

/* ----- Test 1: base64 round-trip of a single fp32 ----------------- */
static void test_base64_one_float(void) {
    float f = 0.0123456f;
    uint8_t le[4];
    char b64[16];
    fp32_to_le_bytes(&f, 1, le);
    size_t n = b64_encode(le, 4, b64);
    float decoded = 0.0f;
    int rc = qat_ply_base64_decode_fp32(b64, n, &decoded, 1);
    CHECK(rc == 1, "decode count");
    CHECK(decoded == f, "round-trip bit-exact");
}

/* ----- Test 2: base64 round-trip of a 12-float array -------------- */
static void test_base64_array(void) {
    float src[12];
    for (int i = 0; i < 12; i++) src[i] = (float)i * 0.5f - 1.25f;
    uint8_t le[48];
    char b64[128];
    fp32_to_le_bytes(src, 12, le);
    size_t n = b64_encode(le, 48, b64);
    float dst[12];
    int rc = qat_ply_base64_decode_fp32(b64, n, dst, 12);
    CHECK(rc == 12, "decode count == 12");
    for (int i = 0; i < 12; i++) {
        CHECK(dst[i] == src[i], "round-trip element bit-exact");
    }
}

/* ----- Test 3: int8 dequant single channel ------------------------ */
static void test_dequant_int8_one_channel(void) {
    int8_t q[5] = {-128, -1, 0, 1, 127};
    float scale[1] = {0.1f};
    float out[5];
    int rc = qat_ply_dequant_int8(q, scale, 5, 1, out);
    CHECK(rc == QAT_PLY_OK, "dequant int8 1ch ok");
    CHECK(out[0] == (float)(-128) * 0.1f, "row 0");
    CHECK(out[1] == (float)(-1)   * 0.1f, "row 1");
    CHECK(out[2] == 0.0f,                 "row 2");
    CHECK(out[3] == (float)(1)    * 0.1f, "row 3");
    CHECK(out[4] == (float)(127)  * 0.1f, "row 4");
}

/* ----- Test 4: int8 dequant multi-channel per-channel scale ------- */
static void test_dequant_int8_multi_channel(void) {
    int8_t q[6] = {10, 20, 30, -10, -20, -30}; /* 2 rows, 3 channels */
    float scale[3] = {0.01f, 0.02f, 0.03f};
    float out[6];
    int rc = qat_ply_dequant_int8(q, scale, 2, 3, out);
    CHECK(rc == QAT_PLY_OK, "dequant int8 multi ok");
    CHECK(out[0] == 10.0f * 0.01f,  "row 0 ch 0");
    CHECK(out[1] == 20.0f * 0.02f,  "row 0 ch 1");
    CHECK(out[2] == 30.0f * 0.03f,  "row 0 ch 2");
    CHECK(out[3] == -10.0f * 0.01f, "row 1 ch 0");
    CHECK(out[4] == -20.0f * 0.02f, "row 1 ch 1");
    CHECK(out[5] == -30.0f * 0.03f, "row 1 ch 2");
}

/* ----- Test 5: int4 packed dequant -------------------------------- */
static void test_dequant_int4_packed(void) {
    /* Two rows, 4 channels each. Signed values [-8..7].
     * Row 0: signed = [-8, -1, 0, 7] -> unsigned [0, 7, 8, 15]
     *        byte 0 = (7 << 4) | 0   = 0x70
     *        byte 1 = (15 << 4) | 8  = 0xF8
     * Row 1: signed = [3, -3, 5, -5]  -> unsigned [11, 5, 13, 3]
     *        byte 0 = (5 << 4) | 11   = 0x5B
     *        byte 1 = (3 << 4) | 13   = 0x3D
     */
    uint8_t packed[4] = {0x70, 0xF8, 0x5B, 0x3D};
    float scale[2] = {0.5f, 0.25f};
    float out[8];
    int rc = qat_ply_dequant_int4_packed(packed, scale, 2, 4, out);
    CHECK(rc == QAT_PLY_OK, "dequant int4 ok");
    CHECK(out[0] == -8.0f * 0.5f, "r0 c0");
    CHECK(out[1] == -1.0f * 0.5f, "r0 c1");
    CHECK(out[2] ==  0.0f * 0.5f, "r0 c2");
    CHECK(out[3] ==  7.0f * 0.5f, "r0 c3");
    CHECK(out[4] ==  3.0f * 0.25f, "r1 c0");
    CHECK(out[5] == -3.0f * 0.25f, "r1 c1");
    CHECK(out[6] ==  5.0f * 0.25f, "r1 c2");
    CHECK(out[7] == -5.0f * 0.25f, "r1 c3");
}

/* ----- Test 6: header parser — int8 with scale_b64 ---------------- */
static void test_parse_header_int8(void) {
    /* Encode scales as the trainer does. */
    float scales[3] = {0.1f, 0.2f, 0.3f};
    uint8_t le[12];
    char b64[32];
    fp32_to_le_bytes(scales, 3, le);
    size_t n = b64_encode(le, 12, b64);

    char header[512];
    snprintf(header, sizeof(header),
             "ply\nformat binary_little_endian 1.0\n"
             "comment quantized_field f_anchor_feat int8 channels=3 scale_b64=%s\n"
             "element vertex 1\n"
             "property uchar f_anchor_feat_q_0\n"
             "end_header\n",
             b64);

    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(header, strlen(header), &hdr);
    CHECK(rc == QAT_PLY_OK, "parse header rc");
    CHECK(hdr.n_fields == 1, "n_fields == 1");
    CHECK(strcmp(hdr.fields[0].name, "f_anchor_feat") == 0, "name");
    CHECK(hdr.fields[0].dtype == QAT_PLY_DTYPE_INT8, "dtype");
    CHECK(hdr.fields[0].channels == 3, "channels");
    CHECK(hdr.fields[0].scale_kind == QAT_PLY_SCALE_PER_CHANNEL, "scale kind");
    CHECK(hdr.fields[0].scale_b64_len == n, "b64 len");

    /* Decode the embedded scale string. */
    float decoded[3];
    int dc = qat_ply_base64_decode_fp32(
        header + hdr.fields[0].scale_b64_offset,
        hdr.fields[0].scale_b64_len,
        decoded, 3);
    CHECK(dc == 3, "decoded 3 scales");
    CHECK(decoded[0] == 0.1f, "scale 0");
    CHECK(decoded[1] == 0.2f, "scale 1");
    CHECK(decoded[2] == 0.3f, "scale 2");
}

/* ----- Test 7: header parser — int4 per_anchor -------------------- */
static void test_parse_header_int4(void) {
    const char *header =
        "ply\nformat binary_little_endian 1.0\n"
        "comment quantized_field f_offset int4 channels=30 "
        "packed_per_byte=2 scale_kind=per_anchor\n"
        "element vertex 1\n"
        "end_header\n";
    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(header, strlen(header), &hdr);
    CHECK(rc == QAT_PLY_OK, "parse int4 rc");
    CHECK(hdr.n_fields == 1, "n_fields == 1");
    CHECK(strcmp(hdr.fields[0].name, "f_offset") == 0, "name");
    CHECK(hdr.fields[0].dtype == QAT_PLY_DTYPE_INT4, "dtype int4");
    CHECK(hdr.fields[0].channels == 30, "channels");
    CHECK(hdr.fields[0].scale_kind == QAT_PLY_SCALE_PER_ANCHOR, "per_anchor");
    CHECK(hdr.fields[0].packed_per_byte == 2, "ppb 2");
    CHECK(hdr.fields[0].scale_b64_len == 0, "no inline scale");
}

/* ----- Test 8: header with zero quantized_field comments ---------- */
static void test_parse_header_none(void) {
    const char *header =
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "end_header\n";
    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(header, strlen(header), &hdr);
    CHECK(rc == QAT_PLY_OK, "parse none rc");
    CHECK(hdr.n_fields == 0, "n_fields == 0");
}

/* ----- Test 9: malformed header — bad channels -------------------- */
static void test_parse_header_bad_channels(void) {
    const char *header =
        "ply\nformat binary_little_endian 1.0\n"
        "comment quantized_field f_x int8 channels=abc scale_b64=AAAA\n"
        "end_header\n";
    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(header, strlen(header), &hdr);
    CHECK(rc == QAT_PLY_ERR_BAD_HEADER, "bad channels rejected");
}

/* ----- Test 10: multiple quantized_field lines -------------------- */
static void test_parse_header_multi(void) {
    float scales[2] = {1.0f, 2.0f};
    uint8_t le[8];
    char b64[16];
    fp32_to_le_bytes(scales, 2, le);
    b64_encode(le, 8, b64);
    char header[1024];
    snprintf(header, sizeof(header),
             "ply\nformat binary_little_endian 1.0\n"
             "comment quantized_field f_anchor_feat int8 channels=2 scale_b64=%s\n"
             "comment quantized_field f_offset int4 channels=30 packed_per_byte=2 scale_kind=per_anchor\n"
             "end_header\n",
             b64);
    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(header, strlen(header), &hdr);
    CHECK(rc == QAT_PLY_OK, "multi parse rc");
    CHECK(hdr.n_fields == 2, "2 fields");
    CHECK(strcmp(hdr.fields[0].name, "f_anchor_feat") == 0, "field 0 name");
    CHECK(strcmp(hdr.fields[1].name, "f_offset") == 0, "field 1 name");
}

/* ----- Test 11: CRLF line endings tolerated ----------------------- */
static void test_parse_header_crlf(void) {
    float scales[1] = {0.5f};
    uint8_t le[4];
    char b64[16];
    fp32_to_le_bytes(scales, 1, le);
    b64_encode(le, 4, b64);
    char header[512];
    snprintf(header, sizeof(header),
             "ply\r\nformat binary_little_endian 1.0\r\n"
             "comment quantized_field f_x int8 channels=1 scale_b64=%s\r\n"
             "end_header\r\n",
             b64);
    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(header, strlen(header), &hdr);
    CHECK(rc == QAT_PLY_OK, "crlf parse rc");
    CHECK(hdr.n_fields == 1, "crlf 1 field");
    CHECK(hdr.fields[0].channels == 1, "crlf channels=1");
}

/* ----- Test 12: NULL arguments -------------------------------------*/
static void test_null_args(void) {
    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(NULL, 0, &hdr);
    CHECK(rc == QAT_PLY_ERR_NULL_ARG, "null header rejected");
    rc = qat_ply_parse_header("ply\n", 4, NULL);
    CHECK(rc == QAT_PLY_ERR_NULL_ARG, "null out rejected");
    rc = qat_ply_dequant_int8(NULL, NULL, 0, 0, NULL);
    CHECK(rc == QAT_PLY_ERR_NULL_ARG, "dequant null rejected");
}

/* ----- Test 13: end-to-end PLY-header-driven scale extraction ----- */
static void test_end_to_end(void) {
    /* Build a header with one f_anchor_feat int8 field, channels=2,
     * scale=[0.125, 0.25]. Then verify we can: parse header, decode
     * scales, dequant a tiny block of int8 values. */
    float scales[2] = {0.125f, 0.25f};
    uint8_t le[8];
    char b64[32];
    fp32_to_le_bytes(scales, 2, le);
    size_t bl = b64_encode(le, 8, b64);
    (void)bl;
    char header[512];
    snprintf(header, sizeof(header),
             "ply\nformat binary_little_endian 1.0\n"
             "comment quantized_field f_anchor_feat int8 channels=2 scale_b64=%s\n"
             "element vertex 2\n"
             "property uchar f_anchor_feat_q_0\n"
             "property uchar f_anchor_feat_q_1\n"
             "end_header\n",
             b64);
    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(header, strlen(header), &hdr);
    CHECK(rc == QAT_PLY_OK, "e2e parse");
    float decoded_scales[2];
    int dc = qat_ply_base64_decode_fp32(
        header + hdr.fields[0].scale_b64_offset,
        hdr.fields[0].scale_b64_len,
        decoded_scales, 2);
    CHECK(dc == 2, "e2e decoded 2 scales");
    int8_t q[4] = {8, -8, 4, -4}; /* 2 rows, 2 channels */
    float out[4];
    qat_ply_dequant_int8(q, decoded_scales, 2, 2, out);
    CHECK(out[0] == 8.0f * 0.125f,  "e2e r0c0");
    CHECK(out[1] == -8.0f * 0.25f,  "e2e r0c1");
    CHECK(out[2] == 4.0f * 0.125f,  "e2e r1c0");
    CHECK(out[3] == -4.0f * 0.25f,  "e2e r1c1");
}

int main(void) {
    test_base64_one_float();
    test_base64_array();
    test_dequant_int8_one_channel();
    test_dequant_int8_multi_channel();
    test_dequant_int4_packed();
    test_parse_header_int8();
    test_parse_header_int4();
    test_parse_header_none();
    test_parse_header_bad_channels();
    test_parse_header_multi();
    test_parse_header_crlf();
    test_null_args();
    test_end_to_end();
    if (g_failed) {
        fprintf(stderr, "TEST SUITE FAILED\n");
        return 1;
    }
    printf("All 13 tests passed.\n");
    return 0;
}
