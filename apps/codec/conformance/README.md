# QAT-PLY v1 conformance suite

Ten test fixtures + JSON sidecar assertions for the Catetus
QAT-PLY v1 spec. Each fixture is a valid PLY 1.0 binary little-endian
file with one or more `comment quantized_field ...` markers. The
sidecar JSON in `conformance.json` lists, per fixture and per field,
the expected dequantized fp32 output as a base64-encoded byte string
(little-endian, row-major).

A decoder is conformant when, for every fixture, its dequantized
output for each declared quantized field matches the corresponding
`expected_fp32_b64` byte-for-byte.

## Running the verifier

The verifier `verify.py` ships two independent reference decoders
(Python and C99) and cross-checks both against every fixture + the
JSON expectations:

```bash
# Build the C cross-check binary once:
cc -std=c99 -Wall -Wextra -O2 -I../qat-ply-c \
   ../qat-ply-c/qat_ply_decode.c c_decode_runner.c \
   -o c_decode_runner

# Then verify:
python3 verify.py
```

Expected output: `10/10 cases passed`.

To regenerate the fixtures from the canonical encoder
(`generate_fixtures.py`), pass `--rebuild`:

```bash
python3 verify.py --rebuild
```

## Adding a new fixture

1. Extend `generate_fixtures.py` with a new case.
2. Run `python3 verify.py --rebuild`.
3. Commit `fixtures/case<NN>_*.ply` + the updated `conformance.json`.

## See also

- Spec: <https://catetus.com/specs/qat-ply-v1>
- Reference C decoder: [`../qat-ply-c/`](../qat-ply-c/)
