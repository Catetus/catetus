/*
 * c_decode_runner.c — Conformance helper. Reads a PLY fixture from
 * argv[1], parses the header, locates the quantized_field with name
 * argv[2], reads the body columns, dequantizes, and writes the
 * row-major fp32 output to stdout as raw little-endian bytes.
 *
 * Used by verify.py to cross-check the C reference decoder against the
 * Python reference decoder. If both agree byte-for-byte with the
 * sidecar JSON, the spec has two independent implementations.
 *
 * SPDX-License-Identifier: MIT
 */

#include "../qat-ply-c/qat_ply_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_FILE  (64u * 1024u * 1024u)   /* 64 MiB cap for fixtures */
#define MAX_PROPS 1024

typedef struct {
    char name[64];
    char type[16];
    size_t size;
} prop_t;

static const struct { const char *n; size_t sz; } TYPES[] = {
    {"char", 1}, {"uchar", 1}, {"short", 2}, {"ushort", 2},
    {"int", 4}, {"uint", 4}, {"float", 4}, {"double", 8},
    {"float32", 4},
    {NULL, 0},
};

static size_t prop_size(const char *t) {
    for (int i = 0; TYPES[i].n; i++) if (!strcmp(TYPES[i].n, t)) return TYPES[i].sz;
    return 0;
}

static long find_end_header(const uint8_t *buf, long n) {
    const char *needle = "\nend_header\n";
    for (long i = 0; i + 12 <= n; i++) {
        if (memcmp(buf + i, needle, 12) == 0) return i + 12;
    }
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <ply> <field>\n", argv[0]);
        return 2;
    }
    FILE *fh = fopen(argv[1], "rb");
    if (!fh) { perror("open"); return 2; }
    fseek(fh, 0, SEEK_END);
    long fsz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    if (fsz <= 0 || (size_t)fsz > MAX_FILE) { fclose(fh); return 2; }
    uint8_t *data = (uint8_t *)malloc((size_t)fsz);
    if (!data) { fclose(fh); return 2; }
    if (fread(data, 1, (size_t)fsz, fh) != (size_t)fsz) { fclose(fh); free(data); return 2; }
    fclose(fh);

    long body_off = find_end_header(data, fsz);
    if (body_off < 0) { free(data); fprintf(stderr, "no end_header\n"); return 2; }
    char *header = (char *)data;
    size_t header_len = (size_t)body_off;
    const uint8_t *body = data + body_off;
    size_t body_len = (size_t)(fsz - body_off);

    /* Parse quantized_field comments. */
    qat_ply_header_t hdr;
    int rc = qat_ply_parse_header(header, header_len, &hdr);
    if (rc != QAT_PLY_OK) { fprintf(stderr, "parse_header rc=%d\n", rc); return 2; }

    qat_ply_field_t *field = NULL;
    for (uint32_t i = 0; i < hdr.n_fields; i++) {
        if (strcmp(hdr.fields[i].name, argv[2]) == 0) {
            field = &hdr.fields[i];
            break;
        }
    }
    if (!field) { fprintf(stderr, "field %s not found\n", argv[2]); return 2; }

    /* Lightweight ASCII parse for `element vertex N` + `property` order. */
    uint32_t n_verts = 0;
    prop_t props[MAX_PROPS];
    size_t n_props = 0;
    {
        size_t pos = 0;
        int in_vertex = 0;
        while (pos < header_len) {
            size_t eol = pos;
            while (eol < header_len && header[eol] != '\n') eol++;
            size_t llen = eol - pos;
            if (llen > 0 && header[eol - 1] == '\r') llen--;
            /* Parse simple "element vertex N" and "property T name". */
            if (llen >= 15 && memcmp(header + pos, "element vertex ", 15) == 0) {
                n_verts = (uint32_t)strtoul(header + pos + 15, NULL, 10);
                in_vertex = 1;
            } else if (llen >= 8 && memcmp(header + pos, "element ", 8) == 0) {
                in_vertex = 0;
            } else if (in_vertex && llen >= 9 && memcmp(header + pos, "property ", 9) == 0) {
                if (n_props >= MAX_PROPS) { fprintf(stderr, "too many props\n"); free(data); return 2; }
                /* property <type> <name> */
                const char *p = header + pos + 9;
                size_t plen = llen - 9;
                size_t i = 0;
                while (i < plen && p[i] != ' ' && p[i] != '\t') i++;
                size_t tlen = i;
                while (i < plen && (p[i] == ' ' || p[i] == '\t')) i++;
                size_t name_start = i;
                while (i < plen && p[i] != ' ' && p[i] != '\t') i++;
                size_t nlen = i - name_start;
                if (tlen >= sizeof(props[0].type) || nlen >= sizeof(props[0].name)) {
                    fprintf(stderr, "prop name/type too long\n"); free(data); return 2;
                }
                memcpy(props[n_props].type, p, tlen); props[n_props].type[tlen] = '\0';
                memcpy(props[n_props].name, p + name_start, nlen); props[n_props].name[nlen] = '\0';
                props[n_props].size = prop_size(props[n_props].type);
                if (props[n_props].size == 0) { fprintf(stderr, "unknown type %s\n", props[n_props].type); free(data); return 2; }
                n_props++;
            }
            pos = eol + 1;
        }
    }

    /* Compute row size + per-property offset within a row. */
    size_t row_size = 0;
    size_t prop_off[MAX_PROPS];
    for (size_t i = 0; i < n_props; i++) {
        prop_off[i] = row_size;
        row_size += props[i].size;
    }
    if ((size_t)n_verts * row_size > body_len) {
        fprintf(stderr, "body short\n"); free(data); return 2;
    }

    /* Helper: find a property index by name. */
    #define FIND_PROP(NAME, OUT) do {                       \
        ssize_t _idx = -1;                                  \
        for (size_t _i = 0; _i < n_props; _i++) {           \
            if (strcmp(props[_i].name, (NAME)) == 0) { _idx = (ssize_t)_i; break; } \
        }                                                   \
        if (_idx < 0) { fprintf(stderr, "missing prop %s\n", (NAME)); free(data); return 2; } \
        (OUT) = (size_t)_idx;                               \
    } while (0)

    if (field->dtype == QAT_PLY_DTYPE_INT8) {
        /* Pull C int8 columns row-major. */
        uint32_t C = field->channels;
        int8_t *q = (int8_t *)malloc((size_t)n_verts * C);
        if (!q) { free(data); return 2; }
        for (uint32_t c = 0; c < C; c++) {
            char colname[80];
            snprintf(colname, sizeof(colname), "%s_q_%u", field->name, c);
            size_t pi; FIND_PROP(colname, pi);
            for (uint32_t r = 0; r < n_verts; r++) {
                q[r * C + c] = (int8_t)body[r * row_size + prop_off[pi]];
            }
        }
        /* Decode scales. */
        float scales[2048];
        if (C > 2048) { free(q); free(data); return 2; }
        int dc = qat_ply_base64_decode_fp32(
            header + field->scale_b64_offset, field->scale_b64_len,
            scales, 2048);
        if (dc != (int)C) { free(q); free(data); fprintf(stderr, "scale count %d != C %u\n", dc, C); return 2; }
        float *out = (float *)malloc(sizeof(float) * n_verts * C);
        if (!out) { free(q); free(data); return 2; }
        qat_ply_dequant_int8(q, scales, n_verts, C, out);
        fwrite(out, sizeof(float), (size_t)n_verts * C, stdout);
        free(out); free(q);
    } else { /* int4 */
        uint32_t C = field->channels;
        uint32_t B = (C + 1u) / 2u;
        uint8_t *packed = (uint8_t *)malloc((size_t)n_verts * B);
        if (!packed) { free(data); return 2; }
        for (uint32_t bi = 0; bi < B; bi++) {
            char colname[80];
            snprintf(colname, sizeof(colname), "%s_q_%u", field->name, bi);
            size_t pi; FIND_PROP(colname, pi);
            for (uint32_t r = 0; r < n_verts; r++) {
                packed[r * B + bi] = body[r * row_size + prop_off[pi]];
            }
        }
        char scalecol[80];
        snprintf(scalecol, sizeof(scalecol), "%s_scale", field->name);
        size_t si; FIND_PROP(scalecol, si);
        float *scales = (float *)malloc(sizeof(float) * n_verts);
        if (!scales) { free(packed); free(data); return 2; }
        for (uint32_t r = 0; r < n_verts; r++) {
            uint32_t bits =
                (uint32_t)body[r * row_size + prop_off[si] + 0] |
                ((uint32_t)body[r * row_size + prop_off[si] + 1] << 8) |
                ((uint32_t)body[r * row_size + prop_off[si] + 2] << 16) |
                ((uint32_t)body[r * row_size + prop_off[si] + 3] << 24);
            memcpy(&scales[r], &bits, sizeof(float));
        }
        float *out = (float *)malloc(sizeof(float) * n_verts * C);
        if (!out) { free(scales); free(packed); free(data); return 2; }
        qat_ply_dequant_int4_packed(packed, scales, n_verts, C, out);
        fwrite(out, sizeof(float), (size_t)n_verts * C, stdout);
        free(out); free(scales); free(packed);
    }
    free(data);
    return 0;
}
