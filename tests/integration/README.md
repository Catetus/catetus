# Integration tests

Hermetic, fixture-driven tests for the `catetus` CLI. They assume the
release binary has been built and the tiny fixture corpus has been generated.

## Prereqs

```bash
# 1. Build the CLI (release, so it matches CI timings)
cargo build --release -p catetus-cli

# 2. Generate fixtures (no-op if they already exist on disk)
python3 fixtures/build.py
```

Required tools on `PATH`: `bash`, `jq`, `diff`, `mktemp`.

## Scripts

| Script | What it does |
| ------ | ------------ |
| `cli.sh`    | Happy-path walk through `analyze -> inspect -> convert -> optimize -> inspect`. |
| `golden.sh` | Diff `analyze` JSON against `fixtures/golden/expected_reports/basic_binary.analyze.json`. The `hash` field is ignored while the golden carries the `blake3:PLACEHOLDER_REGENERATE` sentinel. |

## Run

```bash
./tests/integration/cli.sh
./tests/integration/golden.sh
```

Override the binary location with `CATETUS_BIN=/path/to/catetus`.

## Exit codes

* `0`   success
* `1`   test assertion failed (diff mismatch, missing fixture, etc.)
* `127` prerequisite missing (binary not built, `jq` not installed, ...)
