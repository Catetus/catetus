# CI Overview

Lightweight reference for what runs in GitHub Actions, when, and how to
debug failures. The CI's job is to catch broken lint / broken builds
within a few minutes of push — so a fast-moving agent can't silently
land a regression that nobody notices for hours.

## Workflows that gate `test-hero-fast`

| Workflow | File | Trigger | What runs | Budget |
| --- | --- | --- | --- | --- |
| `rust-ci` | `.github/workflows/rust-ci.yml` | push to `test-hero-fast`, all PRs, manual | `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo build --workspace --all-targets --locked` | ~10 min cold, 3–5 min warm |
| `ts-ci` | `.github/workflows/ts-ci.yml` | push to `test-hero-fast`, all PRs, manual | `pnpm install --frozen-lockfile`, build `@catetus/viewer` first, then `pnpm -r lint` and `pnpm -r build` across the workspace (excluding `catetus-visual`) | ~5 min cold, 2–3 min warm |

Both workflows use `concurrency` groups keyed on `github.ref`, so a fresh
push cancels the previous run on the same branch.

### Why no tests in `rust-ci` / `ts-ci`?

The v1 gate is **lint + build only**. Several workspace tests across
both Rust and TS are currently flaky or environment-dependent (Playwright
needs browsers, some Rust tests need fixtures not present in CI, etc.).
Gating fast iteration on those would produce more noise than signal.

Tests still run in the existing wider workflows:

- `test.yml` — runs `cargo test --workspace --all-targets` and the
  non-visual JS test suite on `main` pushes and PRs.
- `visual.yml` — Playwright visual regression.
- `benchmark.yml`, `splatbench-v3.yml`, `khr-conformance.yml`,
  `usd-conformance.yml`, `blender-addon.yml`, `release.yml` —
  product-specific pipelines, untouched by this CI refresh.

## Skipping CI

Append `[skip ci]` (or `[ci skip]`) to the **first line** of a commit
message and GitHub Actions will skip the entire workflow run. Useful
for doc-only changes that genuinely don't need a build:

```
git commit -m "docs: fix typo in README [skip ci]"
```

For force-pushes that include earlier commits, GitHub uses the **HEAD**
commit's message to decide. To skip a specific workflow but not all,
prefer `paths-ignore` in the workflow yaml instead — but for this v1
setup we run everything.

## Debugging failures

1. **Open the failing run** from the GitHub UI or `gh run list -b test-hero-fast`.
2. **Look at the failing step's log.** Steps are named for the thing they do
   (e.g. `cargo fmt --check`, `pnpm -r build`).
3. **Reproduce locally** with the exact same command from the workflow:
   ```
   # Rust
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo build --workspace --all-targets --locked

   # TypeScript
   pnpm install --frozen-lockfile
   pnpm -F @catetus/viewer run build
   pnpm --filter='!catetus-visual' -r --if-present run lint
   pnpm --filter='!catetus-visual' -r --if-present run build
   ```
4. **Caches:** `Swatinem/rust-cache@v2` keys on `Cargo.lock`; the pnpm
   cache in `actions/setup-node@v4` keys on `pnpm-lock.yaml`. After a
   lockfile change the first run will be cold (~10 min Rust / ~5 min TS).
5. **Re-run individual jobs** from the GitHub UI if the failure looks
   like a transient infrastructure issue (network blips on `cargo
   fetch`, npm registry timeouts, etc.).

## Adding a new package or crate

- New Rust crate: add it to the `[workspace] members` list in the root
  `Cargo.toml`. `rust-ci` picks it up automatically via `--workspace`.
- New pnpm package: add it under `packages/` or `apps/` so it matches
  `pnpm-workspace.yaml`'s globs. `ts-ci` picks it up via `pnpm -r`.
  Make sure it has at least `lint` and `build` scripts (or omit them
  and rely on `--if-present`).

## Action versions

All third-party actions are pinned to specific majors. When bumping:

- `actions/checkout@v4`
- `actions/setup-node@v4`
- `dtolnay/rust-toolchain@stable`
- `Swatinem/rust-cache@v2`
- `pnpm/action-setup@v3`
