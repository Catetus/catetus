/*
 * qat_ply_decode.c — C99 reference implementation of the Catetus
 * QAT-PLY v1 on-disk format. See qat_ply_decode.h for the API.
 *
 * SPDX-License-Identifier: MIT
 */

#include "qat_ply_decode.h"

#include <string.h>

/* ----- internal helpers ------------------------------------------- */

static int sf_is_space(char c) {
    return c == ' ' || c == '\t';
}

/* Compare two byte ranges (lhs ptr/len vs NUL-terminated rhs). */
static int sf_eq(const char *lhs, size_t lhs_len, const char *rhs) {
    size_t rl = strlen(rhs);
    if (lhs_len != rl) return 0;
    return memcmp(lhs, rhs, rl) == 0;
}

/* Find the next newline or end of buffer. Returns the offset of '\n'
 * (or `end - start` if there is no newline before end). */
static size_t sf_line_end(const char *p, size_t start, size_t end) {
    size_t i = start;
    while (i < end && p[i] != '\n') i++;
    return i;
}

/* Tokenize a line into pointers into the original buffer. Whitespace is
 * a single space or tab. Returns the number of tokens (>= 0). Token
 * starts and lengths are written into `starts` / `lens`. Caller must
 * size those arrays to `max_tokens`. */
static size_t sf_tokenize(const char *line, size_t len,
                          size_t max_tokens,
                          size_t *starts, size_t *lens) {
    size_t i = 0, n = 0;
    while (i < len && n < max_tokens) {
        while (i < len && sf_is_space(line[i])) i++;
        if (i >= len) break;
        size_t s = i;
        while (i < len && !sf_is_space(line[i])) i++;
        starts[n] = s;
        lens[n] = i - s;
        n++;
    }
    return n;
}

/* Try to parse a decimal uint32. Returns 1 on success, 0 on failure. */
static int sf_parse_u32(const char *s, size_t len, uint32_t *out) {
    if (len == 0) return 0;
    uint64_t v = 0;
    for (size_t i = 0; i < len; i++) {
        char c = s[i];
        if (c < '0' || c > '9') return 0;
        v = v * 10 + (uint32_t)(c - '0');
        if (v > 0xFFFFFFFFu) return 0;
    }
    *out = (uint32_t)v;
    return 1;
}

/* Match a "key=value" token: writes value pointer / length into out_v /
 * out_vlen and returns 1 on match. */
static int sf_match_kv(const char *tok, size_t len, const char *key,
                       const char **out_v, size_t *out_vlen) {
    size_t kl = strlen(key);
    if (len < kl + 1) return 0;
    if (memcmp(tok, key, kl) != 0) return 0;
    if (tok[kl] != '=') return 0;
    *out_v = tok + kl + 1;
    *out_vlen = len - kl - 1;
    return 1;
}

/* ----- public API ------------------------------------------------- */

int qat_ply_parse_header(const char *header, size_t len,
                         qat_ply_header_t *out) {
    if (!header || !out) return QAT_PLY_ERR_NULL_ARG;
    memset(out, 0, sizeof(*out));

    size_t pos = 0;
    while (pos < len) {
        size_t eol = sf_line_end(header, pos, len);
        const char *line = header + pos;
        size_t line_len = eol - pos;

        /* Strip a trailing CR (Windows line endings). */
        if (line_len > 0 && line[line_len - 1] == '\r') line_len--;

        /* end_header — done. */
        if (sf_eq(line, line_len, "end_header")) break;

        /* Need at least "comment quantized_field <name> <dtype> ..." */
        size_t starts[16], lens[16];
        size_t ntok = sf_tokenize(line, line_len, 16, starts, lens);
        if (ntok < 5 ||
            !sf_eq(line + starts[0], lens[0], "comment") ||
            !sf_eq(line + starts[1], lens[1], "quantized_field")) {
            pos = eol + 1;
            continue;
        }

        if (out->n_fields >= QAT_PLY_MAX_FIELDS) {
            return QAT_PLY_ERR_BUFFER_TOO_SMALL;
        }

        qat_ply_field_t *f = &out->fields[out->n_fields];
        memset(f, 0, sizeof(*f));

        /* Name. */
        size_t nl = lens[2];
        if (nl == 0 || nl > QAT_PLY_MAX_NAME) return QAT_PLY_ERR_BAD_HEADER;
        memcpy(f->name, line + starts[2], nl);
        f->name[nl] = '\0';

        /* dtype. */
        if (sf_eq(line + starts[3], lens[3], "int8")) {
            f->dtype = QAT_PLY_DTYPE_INT8;
            f->packed_per_byte = 0;
        } else if (sf_eq(line + starts[3], lens[3], "int4")) {
            f->dtype = QAT_PLY_DTYPE_INT4;
            f->packed_per_byte = 2;
        } else {
            return QAT_PLY_ERR_UNKNOWN_DTYPE;
        }

        /* Defaults. */
        f->scale_kind = (f->dtype == QAT_PLY_DTYPE_INT8)
                            ? QAT_PLY_SCALE_PER_CHANNEL
                            : QAT_PLY_SCALE_PER_ANCHOR;
        f->scale_b64_offset = 0;
        f->scale_b64_len = 0;

        /* Walk remaining key=value tokens. */
        int saw_channels = 0;
        for (size_t t = 4; t < ntok; t++) {
            const char *tok = line + starts[t];
            size_t tlen = lens[t];
            const char *v;
            size_t vlen;

            if (sf_match_kv(tok, tlen, "channels", &v, &vlen)) {
                if (!sf_parse_u32(v, vlen, &f->channels)) {
                    return QAT_PLY_ERR_BAD_HEADER;
                }
                saw_channels = 1;
            } else if (sf_match_kv(tok, tlen, "scale_b64", &v, &vlen)) {
                f->scale_b64_offset = (size_t)(v - header);
                f->scale_b64_len = vlen;
                f->scale_kind = QAT_PLY_SCALE_PER_CHANNEL;
            } else if (sf_match_kv(tok, tlen, "scale_kind", &v, &vlen)) {
                if (sf_eq(v, vlen, "per_anchor")) {
                    f->scale_kind = QAT_PLY_SCALE_PER_ANCHOR;
                } else if (sf_eq(v, vlen, "per_channel")) {
                    f->scale_kind = QAT_PLY_SCALE_PER_CHANNEL;
                } else {
                    return QAT_PLY_ERR_BAD_HEADER;
                }
            } else if (sf_match_kv(tok, tlen, "packed_per_byte", &v, &vlen)) {
                uint32_t ppb = 0;
                if (!sf_parse_u32(v, vlen, &ppb)) return QAT_PLY_ERR_BAD_HEADER;
                if (ppb != 2) return QAT_PLY_ERR_BAD_HEADER;
                f->packed_per_byte = 2;
            }
            /* Unknown tokens are ignored (forward compatibility). */
        }

        if (!saw_channels || f->channels == 0) return QAT_PLY_ERR_BAD_HEADER;

        out->n_fields++;
        pos = eol + 1;
    }

    return QAT_PLY_OK;
}

int qat_ply_dequant_int8(const int8_t *q, const float *scale,
                         size_t n_rows, size_t n_channels, float *out) {
    if (!q || !scale || !out) return QAT_PLY_ERR_NULL_ARG;
    for (size_t r = 0; r < n_rows; r++) {
        for (size_t c = 0; c < n_channels; c++) {
            out[r * n_channels + c] =
                (float)q[r * n_channels + c] * scale[c];
        }
    }
    return QAT_PLY_OK;
}

int qat_ply_dequant_int4_packed(const uint8_t *packed, const float *scale,
                                size_t n_rows, size_t n_channels,
                                float *out) {
    if (!packed || !scale || !out) return QAT_PLY_ERR_NULL_ARG;
    size_t bytes_per_row = (n_channels + 1) / 2;
    for (size_t r = 0; r < n_rows; r++) {
        float s = scale[r];
        const uint8_t *row = packed + r * bytes_per_row;
        float *dst = out + r * n_channels;
        for (size_t c = 0; c < n_channels; c++) {
            uint8_t b = row[c >> 1];
            uint8_t nib = ((c & 1u) == 0u) ? (b & 0x0Fu)
                                            : ((b >> 4) & 0x0Fu);
            int signed_q = (int)nib - 8;
            dst[c] = (float)signed_q * s;
        }
    }
    return QAT_PLY_OK;
}

/* RFC 4648 standard alphabet (no URL-safe variant). */
static int sf_b64_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

int qat_ply_base64_decode_fp32(const char *b64, size_t b64_len,
                               float *out, size_t out_cap) {
    if (!b64 || !out) return QAT_PLY_ERR_NULL_ARG;
    /* Length must be a multiple of 4 and represent a whole number of
     * 4-byte fp32 values. RFC 4648 allows padding '=' for non-multiple
     * input byte counts; for fp32 arrays we always have byte_count % 3
     * varying, so padding may be present. */
    if (b64_len == 0) return 0;
    if (b64_len % 4 != 0) return QAT_PLY_ERR_BAD_BASE64;

    /* Determine decoded byte count from trailing '=' padding. */
    size_t pad = 0;
    if (b64[b64_len - 1] == '=') pad++;
    if (b64_len >= 2 && b64[b64_len - 2] == '=') pad++;
    if (pad > 2) return QAT_PLY_ERR_BAD_BASE64;
    size_t out_bytes = (b64_len / 4) * 3 - pad;
    if (out_bytes % 4 != 0) return QAT_PLY_ERR_BAD_BASE64;
    size_t n_floats = out_bytes / 4;
    if (n_floats > out_cap) return QAT_PLY_ERR_BUFFER_TOO_SMALL;

    /* Decode into a temp buffer to avoid endianness assumptions on `out`.
     * fp32 is reconstructed via memcpy from little-endian 4-byte words. */
    uint8_t word[4];    /* assembling one float */
    size_t bi = 0;      /* byte index into `word` */
    size_t fi = 0;      /* float index */

    for (size_t i = 0; i < b64_len; i += 4) {
        int v0 = sf_b64_val(b64[i]);
        int v1 = sf_b64_val(b64[i + 1]);
        int v2 = (b64[i + 2] == '=') ? 0 : sf_b64_val(b64[i + 2]);
        int v3 = (b64[i + 3] == '=') ? 0 : sf_b64_val(b64[i + 3]);
        if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) return QAT_PLY_ERR_BAD_BASE64;

        uint32_t triple = ((uint32_t)v0 << 18) | ((uint32_t)v1 << 12) |
                          ((uint32_t)v2 << 6)  | (uint32_t)v3;

        size_t emit = 3;
        if (i + 4 == b64_len) emit = 3 - pad;
        for (size_t k = 0; k < emit; k++) {
            uint8_t byte = (uint8_t)((triple >> (16 - 8 * k)) & 0xFFu);
            word[bi++] = byte;
            if (bi == 4) {
                /* Reconstruct a little-endian fp32 portably. */
                uint32_t bits =
                    (uint32_t)word[0] |
                    ((uint32_t)word[1] << 8) |
                    ((uint32_t)word[2] << 16) |
                    ((uint32_t)word[3] << 24);
                float f;
                memcpy(&f, &bits, sizeof(f));
                out[fi++] = f;
                bi = 0;
            }
        }
    }

    if (bi != 0 || fi != n_floats) return QAT_PLY_ERR_BAD_BASE64;
    return (int)n_floats;
}
