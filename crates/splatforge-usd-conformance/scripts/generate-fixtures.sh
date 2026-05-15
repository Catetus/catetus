#!/usr/bin/env bash
# Deterministically regenerate the ParticleField3DGaussianSplat conformance
# fixture corpus into `crates/splatforge-usd-conformance/fixtures/`.
#
# Re-running this script must produce byte-identical output (a Rust integration
# test enforces this — see `tests/fixtures.rs::fixtures_are_byte_deterministic`).
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
out_dir="$repo_root/crates/splatforge-usd-conformance/fixtures"

cd "$repo_root"
rm -rf "$out_dir"
mkdir -p "$out_dir"

cargo run --quiet -p splatforge-usd-conformance --bin splatforge-usd-fixtures -- "$out_dir"

echo "Regenerated fixtures at $out_dir"
ls -la "$out_dir"
