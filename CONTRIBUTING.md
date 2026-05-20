# Contributing to Catetus

Thanks for your interest in helping make Gaussian Splats production-ready.

## Quick links

- Bug? → [open an issue](https://github.com/Catetus/catetus/issues/new?template=bug.md)
- Feature idea? → [open an issue](https://github.com/Catetus/catetus/issues/new?template=feature.md)
- Want to contribute a real splat to the benchmark corpus? → [corpus request](https://github.com/Catetus/catetus/issues/new?template=corpus_request.md)
- Pull request? → see below.

## Engineering principles

Catetus is **spec-driven** and **test-first**.

Every feature lifecycle:
1. Spec file under `specs/` (or amend an existing one)
2. Acceptance tests (Gherkin in the spec, real tests in code)
3. Fixtures if needed
4. Implementation
5. Benchmark / visual regression
6. Docs update

Rules of thumb:
- **Standards-first.** glTF KHR_gaussian_splatting is the primary delivery
  target. SPZ is a first-class compressed format. No proprietary container
  formats as the default output.
- **Determinism is non-negotiable.** Same input + same config = byte-identical
  output and stable hashes. No wall-clock, no unseeded RNG in library code.
- **Tests before refactors.** Even cosmetic refactors get a failing test first
  if behavior is changing.
- **Snapshots are sacred.** Don't update snapshot files unless the spec change
  requires it.

## Development setup

See [`INSTALL.md`](./INSTALL.md) for full toolchain bootstrap. Quick path:

```bash
# Rust (stable, 1.74+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node + pnpm (Node 20+, pnpm 9)
brew install node
npm install -g pnpm@9

# Clone + build
git clone https://github.com/Catetus/catetus.git
cd Catetus
cargo build --release -p catetus-cli
pnpm install
pnpm -r build
```

## Workflow

1. **Fork + branch.** Branch from `main` with a descriptive name:
   `feat/spz-streaming-index`, `fix/ply-ascii-quaternion-order`, etc.
2. **Make changes.** Keep PRs scoped to one spec or one concern.
3. **Run the test suite locally.**
   ```bash
   make test
   ```
   This runs `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`,
   `pnpm lint`, and `pnpm test`.
4. **Sign off your commits with DCO.** Catetus uses the
   [Developer Certificate of Origin](https://developercertificate.org/) — every
   commit must have a `Signed-off-by:` trailer:
   ```bash
   git commit -s -m "feat: ..."
   ```
5. **Open a pull request.** Fill out the template. CI must be green.
6. **Code review.** Be patient — review can take a few days. We optimize for
   honest, kind feedback.

## Commit message style

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<optional scope>): <subject>

<optional body>

Signed-off-by: Your Name <your.email@example.com>
```

Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci`.

Examples:
- `feat(gltf): emit SF_spatial_streaming_index for chunked output`
- `fix(ply): correct quaternion field order on import`
- `perf(optimize): morton-sort via prefix-sum buckets`
- `docs(readme): add SplatBench v0 link`

## Code style

- **Rust:** `cargo fmt`, `clippy -D warnings`. No `unwrap()` / `panic!()` in
  library code; use `?` and typed errors. Public items have at least a
  one-line doc comment.
- **TypeScript:** strict mode, no `any` in public API, TSDoc on exported
  functions. `tsc --noEmit` must pass.
- **Shaders:** WGSL and GLSL ES 3.00 are kept in sync; algorithm changes go
  to both.

## What kinds of contributions are welcome?

- **Real splat assets for the corpus** — see the corpus-request template.
  Especially welcome: people/characters, reflective/transparent scenes,
  mobile captures, design-partner private assets behind a license.
- **Viewer adapters** — Three.js, Babylon.js, PlayCanvas, native engines.
  See SPEC-0010 for the parity-matrix interface.
- **Optimization passes** — esp. spatial decimation, view-dependent SH
  truncation, alpha-aware floater detection.
- **Standards work** — alignment with KHR_gaussian_splatting RC, OpenUSD
  ParticleField3DGaussianSplat, conformance test assets.
- **CI / device profiles** — real-device FPS + memory matrices.
- **Bug reports with reproducible PLYs** are gold.

## What's intentionally out of scope (v1)

- Training Gaussian Splats from images/video (use Polycam, PostShot, Inria
  3DGS, etc. — Catetus consumes their output).
- A full DCC editor or 3D engine.
- Dynamic / 4D splats (reserved in IR, not implemented).
- Healthcare / surgery use cases.

## License

By contributing, you agree that your contributions are licensed under the
[Apache 2.0 License](./LICENSE).
