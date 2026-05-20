# qat-ply-c — reference C99 decoder for Catetus QAT-PLY v1

Zero-dependency, single-translation-unit C99 decoder for the Catetus
QAT-PLY v1 on-disk format. Implements:

- `qat_ply_parse_header` — locate `comment quantized_field ...` markers
- `qat_ply_base64_decode_fp32` — RFC 4648 std base64 → fp32 array
- `qat_ply_dequant_int8` — per-channel int8 dequant
- `qat_ply_dequant_int4_packed` — per-anchor packed int4 dequant

No `malloc`, no I/O, no platform headers beyond `<stdint.h>`,
`<stddef.h>`, `<string.h>`. Vendor the two files (`qat_ply_decode.h` and
`qat_ply_decode.c`) directly into your renderer.

## Build + test

```bash
make test
```

Expected output: `All 13 tests passed.`

## License

MIT (see [LICENSE](./LICENSE)).
