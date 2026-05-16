// wrapper.cc — Emscripten bindings for the SplatForge QAT-PLY v1 reference
// C decoder. Exposes a small JS-friendly facade: parseHeader, dequantInt8,
// dequantInt4Packed, base64DecodeFp32. The wrapper keeps all data
// caller-owned: TypedArrays cross the JS<->WASM boundary by value.
//
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "../../qat-ply-c/qat_ply_decode.h"
}

#include <emscripten/bind.h>
#include <emscripten/val.h>

using emscripten::val;

namespace {

struct JsField {
    std::string name;
    int dtype;
    uint32_t channels;
    int scale_kind;
    uint32_t packed_per_byte;
    uint32_t scale_b64_offset;
    uint32_t scale_b64_len;
};

struct JsHeader {
    std::vector<JsField> fields;
};

JsHeader parse_header(const std::string &hdr) {
    JsHeader out;
    qat_ply_header_t parsed{};
    int rc = qat_ply_parse_header(hdr.data(), hdr.size(), &parsed);
    if (rc != QAT_PLY_OK) {
        return out;
    }
    out.fields.reserve(parsed.n_fields);
    for (uint32_t i = 0; i < parsed.n_fields; i++) {
        JsField f;
        f.name = parsed.fields[i].name;
        f.dtype = (int)parsed.fields[i].dtype;
        f.channels = parsed.fields[i].channels;
        f.scale_kind = (int)parsed.fields[i].scale_kind;
        f.packed_per_byte = parsed.fields[i].packed_per_byte;
        f.scale_b64_offset = (uint32_t)parsed.fields[i].scale_b64_offset;
        f.scale_b64_len = (uint32_t)parsed.fields[i].scale_b64_len;
        out.fields.push_back(f);
    }
    return out;
}

val dequant_int8(val q_bytes, val scale_floats,
                 uint32_t n_rows, uint32_t n_channels) {
    const size_t n = (size_t)n_rows * (size_t)n_channels;
    std::vector<int8_t> q(n);
    {
        val view = q_bytes;
        unsigned len = view["length"].as<unsigned>();
        if (len < n) return val::null();
        for (size_t i = 0; i < n; i++) {
            q[i] = (int8_t)view[i].as<int>();
        }
    }
    std::vector<float> scale(n_channels);
    {
        val view = scale_floats;
        unsigned len = view["length"].as<unsigned>();
        if (len < n_channels) return val::null();
        for (uint32_t c = 0; c < n_channels; c++) {
            scale[c] = view[c].as<float>();
        }
    }
    std::vector<float> out(n);
    int rc = qat_ply_dequant_int8(q.data(), scale.data(),
                                  n_rows, n_channels, out.data());
    if (rc != QAT_PLY_OK) return val::null();
    val js_out = val::global("Float32Array").new_(n);
    for (size_t i = 0; i < n; i++) {
        js_out.set(i, out[i]);
    }
    return js_out;
}

val dequant_int4_packed(val packed_bytes, val scale_floats,
                        uint32_t n_rows, uint32_t n_channels) {
    const uint32_t B = (n_channels + 1) / 2;
    const size_t n_in = (size_t)n_rows * (size_t)B;
    std::vector<uint8_t> packed(n_in);
    {
        val view = packed_bytes;
        unsigned len = view["length"].as<unsigned>();
        if (len < n_in) return val::null();
        for (size_t i = 0; i < n_in; i++) {
            packed[i] = (uint8_t)view[i].as<int>();
        }
    }
    std::vector<float> scale(n_rows);
    {
        val view = scale_floats;
        unsigned len = view["length"].as<unsigned>();
        if (len < n_rows) return val::null();
        for (uint32_t r = 0; r < n_rows; r++) {
            scale[r] = view[r].as<float>();
        }
    }
    const size_t n_out = (size_t)n_rows * (size_t)n_channels;
    std::vector<float> out(n_out);
    int rc = qat_ply_dequant_int4_packed(packed.data(), scale.data(),
                                         n_rows, n_channels, out.data());
    if (rc != QAT_PLY_OK) return val::null();
    val js_out = val::global("Float32Array").new_(n_out);
    for (size_t i = 0; i < n_out; i++) {
        js_out.set(i, out[i]);
    }
    return js_out;
}

val base64_decode_fp32(const std::string &b64, uint32_t out_cap) {
    std::vector<float> out(out_cap);
    int rc = qat_ply_base64_decode_fp32(b64.data(), b64.size(),
                                        out.data(), out_cap);
    if (rc < 0) return val::null();
    val js_out = val::global("Float32Array").new_(rc);
    for (int i = 0; i < rc; i++) {
        js_out.set(i, out[i]);
    }
    return js_out;
}

}  // namespace

EMSCRIPTEN_BINDINGS(qat_ply) {
    emscripten::value_object<JsField>("QatPlyField")
        .field("name", &JsField::name)
        .field("dtype", &JsField::dtype)
        .field("channels", &JsField::channels)
        .field("scaleKind", &JsField::scale_kind)
        .field("packedPerByte", &JsField::packed_per_byte)
        .field("scaleB64Offset", &JsField::scale_b64_offset)
        .field("scaleB64Len", &JsField::scale_b64_len);
    emscripten::register_vector<JsField>("VectorQatPlyField");
    emscripten::value_object<JsHeader>("QatPlyHeader")
        .field("fields", &JsHeader::fields);

    emscripten::function("parseHeader", &parse_header);
    emscripten::function("dequantInt8", &dequant_int8);
    emscripten::function("dequantInt4Packed", &dequant_int4_packed);
    emscripten::function("base64DecodeFp32", &base64_decode_fp32);
}
