# Installing SplatForge

> Build from source. Pre-built binaries are not yet published.

## Supported platforms

| Platform | Status | Notes |
| -------- | ------ | ----- |
| macOS arm64 (Apple Silicon) | tested | recommended |
| macOS x86_64 (Intel) | tested | recommended |
| Linux x86_64 | tested | recommended |
| Linux aarch64 | tested | recommended |
| Windows (WSL2) | should work | not regularly tested |
| Windows (native) | should work | not regularly tested |

## Prerequisites

| Tool | Version | Used for |
| ---- | ------- | -------- |
| **Rust** | stable, ≥ 1.74 | CLI + core crates |
| **Node.js** | ≥ 20 | viewer SDK + visual harness |
| **pnpm** | ≥ 9 | JS workspace |
| `git` | any recent | source control |
| `curl` | any recent | downloading benchmark assets |
| `make` | any recent | convenience targets |
| Python 3.10+ | optional | regenerating synthetic fixtures |
| Playwright + Chromium | optional | end-to-end visual diff |

## macOS quick start

```bash
# 1. Rust (idempotent — skip if already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Node + pnpm via Homebrew
brew install node@20
npm install -g pnpm@9

# 3. Clone
git clone https://github.com/montabano1/SplatForge.git
cd SplatForge

# 4. Build everything (CLI + viewer)
make build

# 5. Smoke test
make test
```

## Linux quick start

```bash
# 1. Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Node + pnpm (Debian/Ubuntu)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pnpm@9

# 3. Clone + build
git clone https://github.com/montabano1/SplatForge.git
cd SplatForge
make build
make test
```

## What `make build` does

```
cargo build --release -p splatforge-cli    # → target/release/splatforge
pnpm install                                # → node_modules/
pnpm -r --if-present run build              # → packages/*/dist/
```

Final layout after a successful build:

```
SplatForge/
├── target/release/splatforge      # Rust CLI binary (~2 MB)
├── packages/viewer/dist/          # @splatforge/viewer ESM build
├── packages/report-ui/dist/       # @splatforge/report-ui ESM build
└── tests/visual/node_modules/     # Playwright deps (if --F splatforge-visual install ran)
```

## Verifying the install

```bash
./target/release/splatforge --help
./target/release/splatforge analyze fixtures/tiny/basic_binary.ply --pretty
./target/release/splatforge optimize fixtures/tiny/basic_binary.ply \
    --preset web-mobile --out /tmp/out.gltf
./target/release/splatforge inspect /tmp/out.gltf
```

Expected output for the last command:
```
format=gltf splatCount=3 chunks=1 checksum=ok sf_index=true
```

## Optional: end-to-end visual diff

```bash
pnpm -F splatforge-visual install
pnpm -F splatforge-visual exec playwright install chromium
./target/release/splatforge diff before.ply after.gltf --out reports/diff/
open reports/diff/diff.html
```

If `playwright-core` is not installed, `splatforge diff` produces a valid but
degraded report (`status: "degraded"` in JSON) so downstream tooling never has
to special-case a missing report.

## Optional: run the SplatBench v0 corpus locally

```bash
# Generate synthetic scenes (~677 MB)
python3 benches/synth_scenes.py /tmp/sbench/scenes

# Download real anchors (~1.13 GB)
for SCENE in bonsai bicycle; do
  curl -L -o /tmp/sbench/scenes/$SCENE.ply \
    "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/$SCENE/point_cloud/iteration_7000/point_cloud.ply"
done

# Run the pipeline
make bench-splatbench
```

See [`benches/reports/splatbench-v0.md`](./benches/reports/splatbench-v0.md) for the published numbers.

## Troubleshooting

### `cargo: command not found` after running rustup
Restart your shell or `source "$HOME/.cargo/env"`.

### `pnpm: command not found` after `npm install -g`
Make sure the npm global `bin` directory is on your `$PATH`. `npm config get prefix` shows where global packages live.

### Cargo errors with `Operation not permitted` on writes
This happens on some FUSE-mounted directories (Dropbox, iCloud, sandboxed mounts). Build in a regular filesystem location or set `CARGO_TARGET_DIR=$HOME/.splatforge-target`.

### WebGPU not available in your browser
`@splatforge/viewer` auto-falls-back to WebGL2. Force the renderer with `new SplatForgeViewer({ renderer: 'webgl2', ... })`.

### Playwright browsers fail to install
Run `pnpm -F splatforge-visual exec playwright install --with-deps chromium` to also install OS-level prerequisites.

### `tsc` reports `cannot find name 'GPUDevice'` etc.
Make sure `@webgpu/types` is installed in the workspace: `pnpm install`. The viewer's `tsconfig.json` already references the package.

### My splat doesn't load
Run `splatforge inspect your.ply` first. Most issues are missing required fields (`rot_*`, `scale_*`, `f_dc_*`) or big-endian PLYs (not yet supported — see [SPEC-0002](./specs/0002-ply-ingest.md)).
