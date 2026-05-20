# Cross-target conformance suite

This harness decodes every fixture in `../fixtures/` through each
available reference implementation of the QAT-PLY v1 wire format and
asserts they all produce byte-exact identical fp32 output.

Targets:

| # | Target          | How it's verified                                    |
|---|-----------------|------------------------------------------------------|
| 1 | Python          | Imports `verify.py` and dequantizes in-process       |
| 2 | C (reference)   | Builds and runs `c_decode_runner`                    |
| 3 | WebAssembly     | `pnpm test` inside `../qat-ply-wasm/` (Vitest)       |
| 4 | iOS + Metal     | `swift test` inside `../../../ios/CatetusQATKernel/` |
| 5 | Android Vulkan  | `cmake -B build` + `glslang -V` on the GLSL shader  |

Targets 3-5 each run their own per-fixture byte-exact check internally,
so a PASS here means that target's full conformance suite passed.
Target 5 falls back to a syntactic-only check (CMake configure + SPIR-V
compile) when no Android device is available, since libvulkan is not
present on the host.

## Usage

```bash
python3 run_cross_target.py
```

Per-row legend in the per-field matrix:

- `pass`  — bytes match expected fp32 (LE) from `conformance.json`
- `skip`  — implementation not available on this host
- `FAIL`  — bytes mismatch (this is the credibility-breaking case)
