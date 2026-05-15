# `@splatforge/cli`

Deterministic Gaussian-splat optimizer, validator, and converter — the
[SplatForge](https://github.com/splatforge/splatforge) CLI, fetched as a
native Rust binary on `npm install`.

```bash
npm install -g @splatforge/cli
splatforge --help
splatforge-khr-validate path/to/scene.gltf
splatforge-usd-validate path/to/scene.usdc
```

## What this package does

This is a **thin npm wrapper** around the native Rust binaries published
to [SplatForge's GitHub releases](https://github.com/splatforge/splatforge/releases).
On `npm install` the `postinstall` step:

1. Detects your OS + architecture.
2. Downloads the matching pre-built archive
   (e.g. `splatforge-v0.1.0-aarch64-apple-darwin.tar.gz`).
3. Verifies the archive's SHA-256 against the release's `SHASUMS256.txt`.
4. Extracts the three binaries into the package's `native/` directory.

The three published CLIs (`splatforge`, `splatforge-khr-validate`,
`splatforge-usd-validate`) are exposed as standard npm `bin` entries, so
they appear on your `$PATH` just like a JS CLI would.

## Supported platforms

| OS      | Architectures        |
| ------- | -------------------- |
| macOS   | arm64, x64           |
| Linux   | x64 (glibc), arm64   |
| Windows | x64                  |

If your platform isn't in the matrix, fall back to building from source:

```bash
cargo install --git https://github.com/splatforge/splatforge \
  --locked splatforge-cli
```

## Offline / no-network installs

Set `SPLATFORGE_SKIP_DOWNLOAD=1` to bypass the postinstall fetch. You'll
need to drop the native binary into the package's `native/` directory
yourself.

```bash
SPLATFORGE_SKIP_DOWNLOAD=1 npm install -g @splatforge/cli
# Then manually:
curl -L https://github.com/splatforge/splatforge/releases/download/v0.1.0/<archive> \
  | tar -xz -C $(npm root -g)/@splatforge/cli/native
```

## Verifying integrity

Every release ships a `SHASUMS256.txt` manifest. The postinstall script
verifies the downloaded archive against it before extraction; if the
hash doesn't match, install fails loudly and nothing is placed on disk.

## License

Apache-2.0. See `LICENSE` in the upstream repo.
