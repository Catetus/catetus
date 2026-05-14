# PRD + Engineering Plan: SplatForge

## Working title

**SplatForge** — the production optimization and delivery layer for Gaussian Splats.

## Committed positioning

> **SplatForge makes Gaussian Splats production-ready: optimized, standards-aligned, streamable, benchmarked, and safe to ship.**

SplatForge is **not** another capture tool, editor, or proprietary splat format. It is the standards-first infrastructure layer that sits between capture/training tools and production delivery.

## One-line pitch

**FFmpeg + Lighthouse + Cloudinary for Gaussian Splats**: ingest large `.ply`, `.spz`, and glTF Gaussian Splat assets; optimize them for real device budgets; output standards-aligned glTF/SPZ/OpenUSD-compatible assets; and generate reproducible visual/performance reports.

## Product thesis

Gaussian Splatting is crossing from research/demo workflows into production spatial media. Capture and editing tools are improving quickly, but delivery remains fragmented:

- raw PLY exports are often enormous
- SPZ is useful but not the whole pipeline
- glTF Gaussian Splatting is becoming the main runtime interchange target
- OpenUSD support makes splats relevant to professional DCC, VFX, CAD, and digital-twin pipelines
- viewer behavior varies dramatically across browsers, renderers, devices, and GPUs
- teams lack a trusted way to say, “this splat is ready to ship on web/mobile/Quest/visionOS”

The opportunity is to own the **production delivery layer**:

1. Ingest messy splat assets.
2. Normalize them into a canonical internal IR.
3. Optimize and validate against target device budgets.
4. Output standards-aligned artifacts, especially glTF KHR Gaussian Splatting and SPZ.
5. Provide progressive streaming conventions using glTF external buffers and metadata, not a competing proprietary format.
6. Render deterministic visual diffs across multiple viewers/devices.
7. Publish benchmark reports and compatibility matrices.
8. Wrap it all in CLI, CI, API, SDK, and design-partner workflows.

## Strategic decision

SplatForge should be built primarily as a **strategic infrastructure/acquisition-target product**, not only as a near-term SaaS. Revenue matters, but the deepest moat is:

- a canonical benchmark corpus
- standards-aligned conversion quality
- visual regression infrastructure
- viewer parity matrices
- device performance data
- deep glTF/OpenUSD integration
- partnerships with capture and creative tools

The business should still support SaaS/API revenue, but phase prioritization should favor strategic moat over generic billing features.

---

# Target users

## Primary persona: spatial web / 3D platform engineer

Works at a company building web-based 3D, AR, digital twin, game, robotics simulation, mapping, real estate, e-commerce, or volumetric content tools.

Pain points:

- Has giant Gaussian Splat captures that look great locally but are too large for web/mobile.
- Needs predictable load time, FPS, and memory budgets.
- Needs to ship across WebGPU/WebGL, Three.js/Babylon/PlayCanvas/custom engines.
- Needs reproducible quality checks.
- Wants glTF/OpenUSD compatibility without waiting for every tool to mature.

## Secondary persona: capture / reality-scanning tool company

Examples of target categories:

- phone scanning apps
- photogrammetry tools
- drone scanning tools
- real estate scanning products
- 3D creative/capture platforms

Pain points:

- Users create beautiful but hard-to-deliver splats.
- Export-to-web is a weak part of the workflow.
- They want an optimization API rather than building the whole pipeline in-house.
- They need viewer/embed options.

## Tertiary persona: asset pipeline / marketplace / enterprise team

Pain points:

- Needs validation for uploaded assets.
- Needs CI gates for asset changes.
- Needs before/after quality reports.
- Needs batch optimization and device budgets.
- Needs “known-good” benchmarked output for customers.

---

# Non-goals for v1

SplatForge v1 should **not** attempt to:

- train Gaussian Splats from source images/video
- replace Luma, Polycam, Scaniverse, PostShot, SuperSplat, or other capture/editor tools
- build a full DCC editor
- build a full 3D engine
- support dynamic 4D splats end-to-end in v1
- invent a new proprietary package format as the primary output
- target regulated healthcare or surgery use cases
- promise perfect compression research breakthroughs before shipping practical optimization

---

# Why now

The standards and ecosystem are finally mature enough to build around:

- **glTF KHR Gaussian Splatting** is in release-candidate status and should be treated as the primary runtime interchange target.
- **OpenUSD v26.03** introduced native 3D Gaussian Splat schema support and WebAssembly build targets.
- **SPZ** has become a serious compressed splat format and is complementary to glTF/OpenUSD, not something SplatForge should fight.
- Browser-based splat editors/viewers such as SuperSplat and related WebGPU tooling prove that web delivery is commercially relevant.
- Production teams still lack the optimizer, validator, CI, benchmark, and viewer-parity layer.

---

# Product principles

1. **Standards-first, proprietary-last**
   - glTF KHR Gaussian Splatting is the primary v1 delivery target.
   - SPZ is a first-class compressed format.
   - OpenUSD is the professional pipeline target.
   - Any SplatForge-specific metadata should be optional, transparent, and removable.

2. **No proprietary default package format**
   - Do not create `.sfz` as the main artifact.
   - Use `.gltf/.glb` with KHR Gaussian Splatting, external buffers, and optional vendor metadata for chunk/LOD/streaming behavior.
   - When advanced streaming metadata is present, the asset should degrade gracefully to baseline glTF behavior where possible.

3. **Every optimization must be measurable**
   - File size, first meaningful paint, total load time, memory, FPS, perceptual quality, and viewer compatibility must be reported.

4. **The benchmark corpus is the moat**
   - The corpus is not a marketing asset. It is strategic IP.
   - Public corpus builds trust; private corpus builds enterprise value.

5. **The CLI is the source of truth**
   - Hosted jobs and CI must call the same pinned CLI/core pipeline.
   - Every hosted result must be reproducible locally.

6. **Visual regressions are product bugs**
   - Golden visual tests are required for optimizer and renderer changes.

7. **Device budgets matter more than abstract compression ratios**
   - The product should answer: “Will this load smoothly on Safari/iPhone, Chrome/Android, Quest Browser, or desktop WebGPU?”

8. **Design partners before synthetic perfection**
   - Phase 0 must include real assets from 5 design partners.

---

# Acquirer / strategic map

## Adobe

Strategic story:

- Substance/Firefly/Photoshop 3D pipelines need production-grade splat export and optimization.
- Adobe benefits from a standards-aware asset delivery and credentialing layer.

Features that support this story:

- SPZ/glTF/OpenUSD conversion quality
- visual diff reports
- batch optimization
- creative-tool integrations
- benchmark corpus

## NVIDIA

Strategic story:

- Omniverse and Cosmos need reliable OpenUSD-native spatial asset handling.
- NVIDIA benefits from a reference-grade pipeline for splat ingestion, optimization, validation, and rendering benchmarks.

Features that support this story:

- OpenUSD `ParticleField3DGaussianSplat` support
- USD round-tripping
- GPU performance benchmarks
- device/viewer compatibility matrix
- benchmark leaderboard

## Apple

Strategic story:

- visionOS, Safari/WebGPU, and USD-centric spatial workflows need fast, mobile-friendly splat delivery.

Features that support this story:

- Safari/WebGPU performance
- mobile memory budgets
- USD/OpenUSD compatibility
- visual quality reports
- device profiles

## Meta / Niantic / XR platforms

Strategic story:

- Quest/mobile/web XR needs efficient splat delivery and SPZ compatibility.

Features that support this story:

- SPZ support
- web/mobile streaming
- Quest/browser performance profiles
- progressive loading

## Unity / Epic

Strategic story:

- Engines need import/export, optimization, and runtime integration for Gaussian Splats.

Features that support this story:

- glTF KHR support
- Unity/Unreal import examples
- SDK licensing
- runtime budgets

## Shopify / commerce platforms

Strategic story:

- Merchants need product scans that load fast and look clean on mobile web.

Features that support this story:

- object-aware cleanup/pruning
- web-mobile preset
- CDN-friendly glTF/SPZ output
- visual preview links
- batch asset catalog optimization

## Cloudflare

Strategic story:

- Edge media infrastructure could expand from images/video/AI into spatial assets.

Features that support this story:

- hosted API
- edge-friendly package layout
- CDN-aware chunking
- API/worker architecture

---

# MVP scope

## MVP name

**SplatForge Alpha**

## MVP promise

A developer can run:

```bash
splatforge analyze scene.ply
splatforge optimize scene.ply --target web-mobile --out scene.gltf
splatforge preview scene.gltf
splatforge diff scene.ply scene.gltf
```

And receive:

- standards-aligned optimized output
- glTF KHR Gaussian Splatting target, where supported
- optional SPZ output
- local preview
- before/after visual comparison
- size/load/memory/FPS estimates
- validation report
- reproducible JSON manifest/report

## v1 inputs

- Gaussian Splat PLY
- SPZ
- glTF KHR Gaussian Splatting

## v1 outputs

- glTF / GLB with KHR Gaussian Splatting
- glTF with external buffers for progressive delivery
- SPZ
- analysis and optimization reports
- preview HTML bundle

## Deferred, but designed-for

- deeper OpenUSD round-tripping
- dynamic/4D splats
- native engine plugins
- enterprise private benchmark suites

---

# Performance targets

Phase 0/1 targets:

- **Compression/size reduction:** 10x vs raw PLY on at least one real design-partner scene with visual diff under threshold.
- **Median compression target:** 8–20x vs raw PLY, depending on preset and input.
- **Perceptual degradation target:** less than 3–5% under the chosen visual metric for standard presets.
- **First meaningful paint:** under 1.5s on mid-tier mobile for target scenes under 5M splats after optimization.
- **Mobile memory target:** under 300–500 MB peak for optimized web-mobile scenes.
- **Determinism:** same input + same config = byte-stable output metadata and stable artifact hashes.

These are product targets, not guaranteed claims. They should be validated by the benchmark corpus.

---

# Product features

## 1. CLI ingest

Commands:

```bash
splatforge analyze <input>
splatforge inspect <input>
splatforge optimize <input> --preset <preset> --out <output.gltf|output.glb|output.spz>
splatforge convert <input> --to gltf|glb|spz|usd
splatforge preview <input>
splatforge diff <before> <after>
splatforge benchmark <input> --device-profile <profile>
splatforge corpus run <suite>
```

## 2. Internal representation

Define a canonical `SplatIR` that is independent of PLY/SPZ/glTF/OpenUSD.

```ts
type Splat = {
  position: Vec3Float32;
  rotation: QuaternionFloat32;
  scale: Vec3Float32;
  opacity: Float32;
  color: SHColor | RGBColor;
  semantic?: SemanticLabel;
  temporal?: TemporalInfo; // reserved for future 4D/dynamic splats
  metadata?: Record<string, unknown>;
};
```

IR requirements:

- deterministic ordering
- lossless mode where possible
- explicit attribute availability
- explicit coordinate-system metadata
- stable hashing
- streamable iteration
- reserved temporal/4D fields
- optional semantic labels

## 3. Analysis report

Reports include:

- splat count
- file size
- detected format
- bounding box
- coordinate system warnings
- attribute presence
- opacity distribution
- scale distribution
- SH degree
- estimated memory footprint
- suspicious floaters/outliers
- invalid values
- compression opportunities
- target preset recommendations

## 4. Optimization presets

Initial presets:

- `lossless-repack`
- `web-mobile`
- `web-desktop`
- `quest-browser`
- `visionos-preview`
- `thumbnail-preview`
- `quality-max`
- `size-min`

Optimization passes:

- invalid splat removal
- opacity pruning
- floater/outlier detection
- quantization of position/scale/rotation/opacity
- spherical harmonic degree reduction
- spatial sorting
- Morton/Z-order indexing
- LOD generation
- chunk layout optimization
- metadata preservation/stripping
- object-aware pruning, initially heuristic/experimental

## 5. Object-aware pruning

Use the team’s CV background as a differentiator without over-scoping v1.

V1 approach:

- heuristic subject/background separation based on spatial density, opacity, connected regions, and camera-frustum/capture metadata where available
- optional semantic masks if user supplies them
- product/person/object-preservation mode

Future approach:

- integrate open-vocabulary segmentation or image-derived masks from original capture frames where available
- preserve high-value objects while aggressively pruning background floaters

This should be a feature flag in Phase 1/2, not a blocker for core optimizer.

## 6. Standards-aligned delivery package

No `.sfz` default format.

Use:

- `.gltf` / `.glb` with KHR Gaussian Splatting
- external binary buffers for chunked/progressive delivery
- optional vendor metadata for chunk index, LOD, checksums, and streaming hints
- SPZ output for compatibility with SPZ-first workflows

Example package layout:

```text
scene/
  scene.gltf
  buffers/
    root.bin
    lod0_0001.bin
    lod0_0002.bin
    lod1_0001.bin
  previews/
    thumb.webp
    hero.webp
  reports/
    analyze.json
    optimize.json
    visual-diff.html
```

The root artifact remains glTF. The folder layout is a deployment layout, not a new file format.

## 7. Spatial streaming index

The glTF metadata should include a spatial streaming index:

- Morton/Z-order chunk ordering
- per-chunk bounding boxes
- per-chunk LOD level
- per-chunk byte ranges or external buffer URI
- checksum
- splat count
- recommended load priority

Viewer behavior:

- load root/preview chunks first
- load visible high-priority chunks next
- use camera frustum and distance for progressive loading
- expose hooks for custom loading policy

## 8. Viewer SDK

Package:

```bash
npm install @splatforge/viewer
```

Basic usage:

```ts
import { SplatForgeViewer } from '@splatforge/viewer';

const viewer = new SplatForgeViewer({
  canvas: document.getElementById('canvas'),
  src: '/assets/scene/scene.gltf',
  budget: 'web-mobile'
});

await viewer.load();
```

Renderer plan:

- WebGPU primary
- WebGL2 fallback
- deterministic camera-path mode for tests
- event lifecycle
- stats overlay
- memory/FPS metrics

Events:

- `loadStart`
- `manifestLoaded`
- `firstRender`
- `chunkLoaded`
- `qualityChanged`
- `complete`
- `error`

## 9. Viewer parity benchmarks

This is a core moat feature.

Compare the same asset across:

- SplatForge WebGPU viewer
- SplatForge WebGL2 fallback
- Three.js integration where possible
- Babylon.js integration where possible
- PlayCanvas/SuperSplat-compatible path where possible
- native reference renderer over time

Output:

- compatibility matrix
- visual similarity score
- FPS/memory/load-time comparison
- degradation warnings

Example:

```json
{
  "asset": "warehouse_scan",
  "chromeWebGPU": { "visualScore": 0.98, "fps": 61 },
  "safariWebGPU": { "visualScore": 0.94, "fps": 47 },
  "safariWebGL2": { "visualScore": 0.72, "fps": 21, "warning": "opacity_sorting_artifacts" }
}
```

## 10. Visual diff and quality reports

For every optimized output:

- render fixed camera paths before/after
- capture frames
- compute image-level and perceptual metrics
- generate HTML report
- fail CI if diff exceeds threshold

Reports should be human-readable and machine-readable.

## 11. Hosted analyzer and optimizer

Hosted product should support:

- upload
- analyze
- optimize
- preview
- download
- API keys
- shareable reports
- batch jobs
- CI integration

Every hosted job must call the pinned CLI binary.

---

# Open source strategy

## Licensing

Recommended:

- Apache 2.0 for core CLI and basic viewer SDK
- optional dual-license/commercial terms for embedded enterprise SDKs
- CLA or DCO from the start

## Repos

Possible structure:

- `splatforge/splatforge` — CLI, core, SDK
- `splatforge/corpus` — public benchmark corpus metadata, not necessarily all raw assets
- `splatforge/benchmarks` — public leaderboard and result history

## What stays open

- core IR
- PLY/SPZ/glTF import/export basics
- CLI analyzer
- basic optimizer presets
- benchmark runner
- viewer SDK baseline

## What can be paid

- hosted optimization at scale
- private benchmark corpus
- enterprise CI dashboard
- custom device profiles
- private cloud deployment
- embedded SDK licensing
- batch asset catalog processing
- support/SLA

---

# Benchmark and moat strategy

## Name

**SplatBench** — the canonical benchmark suite for production Gaussian Splat delivery.

## Corpus classes

- small product scans
- people/characters
- indoor real estate
- outdoor scenes
- reflective/transparent failure cases
- noisy captures with floaters
- dense large scenes
- mobile-friendly scenes
- glTF KHR test assets
- SPZ test assets
- OpenUSD test assets

## Public vs private corpus

Public:

- small/medium assets with permissive licensing
- reproducible benchmark results
- leaderboard
- compatibility matrix

Private:

- design-partner/customer assets
- enterprise-only regression suites
- paid private compatibility testing

## Benchmark categories

- best web-mobile output
- best high-fidelity output
- fastest first render
- lowest memory
- best visual quality under size cap
- viewer parity
- SPZ/glTF/OpenUSD round-trip fidelity

## Why this is the moat

Formats can be copied. A trusted corpus with reproducible cross-device visual regression is much harder to copy. This should become the equivalent of an MLPerf/W3C-style asset for splat delivery.

---

# Security and production readiness

## Uploaded file safety

- strict input validation
- maximum file size per tier
- parser fuzzing
- sandboxed workers
- no arbitrary code execution from asset metadata
- timeout and memory limits
- malicious archive/path traversal checks

## Package safety

- checksums for external buffers
- optional signed reports/artifacts
- deterministic builds
- dependency scanning
- SBOM for enterprise builds

## SaaS safety

- API rate limits
- quota enforcement
- private asset access controls
- encrypted storage
- short-lived signed URLs
- audit logs for enterprise

---

# Monetization

## Free

- OSS CLI
- local analyze/optimize for small assets
- public web analyzer with file-size cap
- public benchmark reports

## Pro: $49–99/month

- larger hosted jobs
- private uploads
- faster queue
- shareable previews
- batch jobs with monthly usage cap
- team workspace lite

## Team / Studio

- seats + usage caps
- CI integration
- asset catalog batch processing
- custom presets
- private reports

## Enterprise

- SDK licensing
- private cloud/VPC/on-prem
- custom device profiles
- SLA/support
- SSO/audit logs
- private benchmark corpus
- white-label viewer

## Pricing guidance

Avoid pure unpredictable per-GB pricing as the default. Use seats/plans with generous usage caps, then overage pricing for very large processing volumes.

---

# Unit economics model

Before hosted optimization launches, model:

- average input size
- average output size
- CPU/GPU processing time
- temporary storage duration
- output storage duration
- egress per preview/download
- worker retry rate
- queue idle time

Example worksheet fields:

```text
input_size_gb
output_size_gb
cpu_minutes
memory_gb
storage_days
egress_gb
cost_per_cpu_minute
cost_per_gb_storage_month
cost_per_gb_egress
gross_margin_by_plan
```

Phase 4 exit criterion should include positive gross margin under expected Pro/Team usage.

---

# Technical stack

## Core

Rust.

Recommended crates / technologies:

- `ply-rs` or custom fast parser for PLY
- SPZ C++ interop or Rust wrapper where practical
- `gltf` crate with custom extension support
- OpenUSD integration initially via CLI/FFI/subprocess, deeper binding later
- `wgpu` for Rust-side validation/rendering experiments
- `tokio` for async IO
- `rayon` for CPU parallelism
- `nalgebra` or `glam` for math
- `bitvec` for bit packing
- `rkyv` or similar for zero-copy internal cache experiments
- `serde` / `schemars` for JSON schemas
- `insta` for snapshot testing
- `proptest` for property testing

## Viewer

TypeScript.

- WebGPU primary
- WebGL2 fallback
- Playwright for browser tests
- Vitest for unit tests
- optional React wrapper after SDK stabilizes

## Hosted backend

Rust Axum preferred for consistency with core.

- Axum API
- worker queue
- object storage abstraction
- Postgres for metadata
- Redis or queue system for jobs
- workers invoke pinned CLI binary

## Frontend

Next.js or Vite.

Keep simple:

- upload
- reports
- preview
- benchmark pages
- API docs

---

# Team composition

Assumed team for 20-week plan:

1. **Rust systems / formats engineer**
   - IR, PLY, SPZ, glTF, optimization passes

2. **Graphics / WebGPU engineer**
   - viewer, renderer, progressive loading, visual tests

3. **Backend / infra engineer**
   - hosted API, workers, storage, queues, cost controls

4. **Frontend / devtools engineer**
   - web analyzer, reports, docs, dashboard, examples

5. **Optional ML/CV engineer**
   - object-aware pruning, semantic cleanup, automated quality heuristics

6. **Optional DevRel / partnerships lead**
   - design partners, capture-tool integrations, benchmark corpus licensing

---

# Parallel workstreams

## Stream A: Core + CLI + Optimizer

Owner: Rust systems engineer(s)

Responsibilities:

- IR
- parsers
- glTF/SPZ I/O
- optimization passes
- deterministic reports
- benchmarks

## Stream B: Viewer + Visual Regression

Owner: graphics/WebGPU engineer

Responsibilities:

- viewer SDK
- progressive loading
- WebGPU/WebGL2
- deterministic frame capture
- viewer parity matrix

## Stream C: Hosted + Web Demo

Owner: backend/frontend engineers

Responsibilities:

- upload/demo site
- job API
- worker orchestration
- preview links
- billing later

## Stream D: Corpus + Partnerships

Owner: founder/DevRel/engineer hybrid

Responsibilities:

- design partners
- capture-tool integrations
- public corpus
- benchmark leaderboard
- asset licensing

---

# Roadmap

## Phase 0: Technical spike + design partners, weeks 1–2

Goals:

- prove real compression/delivery value
- recruit 5 design partners with real assets
- avoid synthetic benchmark trap

Deliverables:

- parse one real PLY scene
- parse one SPZ scene
- initial glTF KHR read/write spike
- basic visual preview
- basic optimize pass
- before/after report
- design partner asset intake process

Exit criteria:

- 5 design partners identified
- 3 real assets in private corpus
- at least one real scene achieves 10x size reduction vs raw PLY with visual diff under threshold
- local CLI demo works end-to-end

## Phase 1: CLI Alpha + standards I/O, weeks 3–6

Deliverables:

- SplatIR
- PLY ingest
- SPZ ingest/export
- basic glTF KHR Gaussian Splatting ingest/export
- analyze report
- optimization presets
- deterministic output
- external-buffer glTF layout
- initial Morton spatial index
- snapshot/property/fuzz tests

Exit criteria:

- `analyze`, `optimize`, `convert`, `inspect` usable externally
- 20+ fixtures passing
- glTF KHR output viewable in supported viewers or SplatForge viewer
- public GitHub repo ready

## Phase 2: Viewer SDK + visual diff + benchmark runner, weeks 7–12

Deliverables:

- WebGPU viewer
- WebGL2 fallback
- progressive loading from glTF external buffers
- visual diff harness
- viewer parity benchmark skeleton
- SplatBench initial public suite
- local preview command
- basic docs/examples

Exit criteria:

- first meaningful render before full asset load
- deterministic camera-path tests
- public benchmark page v0
- compatibility report across at least 2 browser paths

## Phase 3: Hosted API + OpenUSD basics + partnerships, months 4–6

Deliverables:

- hosted analyzer
- hosted optimizer
- API keys
- worker architecture
- OpenUSD import/export spike
- benchmark leaderboard
- capture-tool partnership outreach
- shareable preview reports

Exit criteria:

- 3 design partners using hosted reports
- OpenUSD basic round-trip for supported subset
- first capture-tool integration conversation or prototype
- unit economics model validated

## Phase 4: Advanced compression + enterprise pipeline, months 7–9

Deliverables:

- advanced quantization and chunking
- LOD/streaming polish
- object-aware pruning v1
- GitHub Action / CI gate
- team workspaces
- private benchmark suite
- custom device profiles
- SDK licensing package

Exit criteria:

- first paid team/studio users
- private corpus used in regression testing
- one serious strategic platform conversation

## Phase 5: Standards influence + strategic depth, months 9–12

Deliverables:

- deeper OpenUSD support
- stronger glTF extension compatibility
- published SplatBench report
- viewer parity matrix across major devices
- Unity/Unreal/Three.js examples
- proposal or contribution to relevant standards discussions if appropriate

Exit criteria:

- SplatForge cited or used by ecosystem participants
- 5+ recurring teams
- recognizable corpus/benchmark moat
- acquisition/partnership conversations plausible

---

# SpecDD / TDD engineering plan

## Engineering philosophy

Use **Spec-Driven Development** for public behavior and **Test-Driven Development** for core transformations.

Every feature starts with:

1. spec file
2. fixtures
3. acceptance tests
4. implementation
5. benchmark
6. visual regression if rendering is involved

## Repo structure

```text
splatforge/
  README.md
  specs/
    0001-ir.md
    0002-ply-ingest.md
    0003-spz-io.md
    0004-gltf-khr-io.md
    0005-analyze-report.md
    0006-optimization-passes.md
    0007-spatial-streaming-index.md
    0008-viewer-sdk.md
    0009-visual-diff.md
    0010-viewer-parity.md
    0011-api.md
    0012-openusd-basics.md
  crates/
    splatforge-core/
    splatforge-cli/
    splatforge-ply/
    splatforge-spz/
    splatforge-gltf/
    splatforge-optimize/
    splatforge-bench/
  packages/
    viewer/
    report-ui/
  apps/
    web-demo/
    api/
    worker/
  fixtures/
    tiny/
    invalid/
    realworld/
    private/
    golden/
  tests/
    integration/
    e2e/
    visual/
  benches/
  docs/
  .github/
    workflows/
      test.yml
      visual.yml
      benchmark.yml
```

---

## SPEC-0001: Internal Representation

### Goal

Define a deterministic canonical splat representation used by all importers, optimizers, exporters, and tests.

### Acceptance tests

```gherkin
Feature: Canonical IR

Scenario: Load tiny synthetic scene into IR
  Given a synthetic scene with 3 splats
  When I serialize it into SplatIR
  Then the IR contains 3 splats
  And every splat has position, rotation, scale, opacity, and color
  And the scene hash equals the golden hash

Scenario: IR serialization is deterministic
  Given the same input parsed twice
  When I serialize both IR outputs
  Then the byte outputs are identical
  And the hashes are identical

Scenario: IR reserves temporal fields
  Given a static splat scene
  When I parse it into SplatIR
  Then temporal mode is "static"
  And no dynamic frame data is required
```

### Claude Code prompt

```text
Implement SPEC-0001 in crates/splatforge-core.
Write failing tests first.
Expose deterministic serialization, stable hashing, coordinate metadata, and reserved temporal fields.
Do not implement file parsing in this task.
```

---

## SPEC-0002: PLY Ingest

### Goal

Read common Gaussian Splat PLY files and convert them into SplatIR.

### Supported fields v1

- x, y, z
- scale_0, scale_1, scale_2
- rot_0, rot_1, rot_2, rot_3
- opacity
- f_dc_0, f_dc_1, f_dc_2
- optional f_rest_* SH coefficients

### Acceptance tests

```gherkin
Feature: PLY ingest

Scenario: Parse valid binary PLY
  Given fixture "tiny/basic_binary.ply"
  When I run "splatforge analyze tiny/basic_binary.ply"
  Then the command exits 0
  And the report says format is "ply"
  And splatCount is 3

Scenario: Reject PLY missing rotation
  Given fixture "invalid/missing_rotation.ply"
  When I run "splatforge analyze invalid/missing_rotation.ply"
  Then the command exits non-zero
  And stderr includes "missing required rotation fields"
```

### Claude Code prompt

```text
Implement SPEC-0002 in crates/splatforge-ply.
Use TDD with binary little-endian PLY first.
Add malformed fixtures and snapshot structured errors.
Wire into splatforge-cli analyze.
```

---

## SPEC-0003: SPZ I/O

### Goal

Support SPZ as a first-class compressed input/output target.

### Acceptance tests

```gherkin
Feature: SPZ I/O

Scenario: Decode SPZ fixture
  Given fixture "tiny/basic.spz"
  When I run "splatforge analyze tiny/basic.spz"
  Then the command exits 0
  And the report says format is "spz"

Scenario: Convert PLY to SPZ
  Given fixture "tiny/basic_binary.ply"
  When I run "splatforge convert tiny/basic_binary.ply --to spz --out out.spz"
  Then out.spz exists
  And "splatforge inspect out.spz" succeeds

Scenario: Round-trip within tolerance
  Given fixture "corpus/small_room.ply"
  When I convert it to SPZ and back to SplatIR
  Then positions and colors remain within configured tolerance
```

### Claude Code prompt

```text
Implement SPEC-0003 with the smallest robust path first.
Prefer wrapping the existing SPZ implementation if that is faster and safer than a new implementation.
Create round-trip tolerance tests.
Do not optimize SPZ internals yet.
```

---

## SPEC-0004: glTF KHR Gaussian Splatting I/O

### Goal

Make glTF KHR Gaussian Splatting the primary runtime interchange target.

### Requirements

- Read supported glTF Gaussian Splat assets into SplatIR.
- Write SplatIR to glTF/GLB using the KHR extension where supported.
- Support external buffers for progressive loading.
- Preserve metadata needed for validation and optimization reports.
- Fail gracefully if extension version is unsupported.

### Acceptance tests

```gherkin
Feature: glTF KHR Gaussian Splatting I/O

Scenario: Export PLY to glTF KHR
  Given fixture "tiny/basic_binary.ply"
  When I run "splatforge convert tiny/basic_binary.ply --to gltf --out scene.gltf"
  Then scene.gltf exists
  And the glTF declares the Gaussian Splatting extension
  And "splatforge inspect scene.gltf" succeeds

Scenario: Import glTF KHR
  Given fixture "tiny/basic_khr.gltf"
  When I run "splatforge analyze tiny/basic_khr.gltf"
  Then the command exits 0
  And the report says format is "gltf"

Scenario: Unsupported extension version fails clearly
  Given fixture "invalid/unsupported_khr_version.gltf"
  When I inspect it
  Then the command exits non-zero
  And stderr includes "unsupported Gaussian Splatting extension version"
```

### Claude Code prompt

```text
Implement SPEC-0004 in crates/splatforge-gltf.
Start with export from SplatIR to glTF KHR and inspect validation.
Use snapshot tests for generated glTF JSON.
Do not implement advanced streaming metadata until SPEC-0007.
```

---

## SPEC-0005: Analyze Report

### Goal

Generate deterministic JSON reports for any supported input.

### Acceptance tests

```gherkin
Feature: Analyze report

Scenario: Generate deterministic JSON
  Given fixture "tiny/basic_binary.ply"
  When I run analyze twice
  Then the JSON reports are byte-identical except optional timing fields

Scenario: Detect suspicious floaters
  Given fixture "invalid/floater_cluster.ply"
  When I analyze it
  Then warnings include "floater_cluster_detected"

Scenario: Recommend web-mobile optimization
  Given a large raw PLY fixture
  When I analyze it
  Then recommendations include "web-mobile"
```

### Claude Code prompt

```text
Implement SPEC-0005.
Create a JSON schema and snapshot tests.
Do not include timestamps by default.
Reports must be stable across platforms.
```

---

## SPEC-0006: Optimization Pass Framework

### Goal

Create a composable optimization pipeline where each pass is independently testable.

### Initial passes

1. `RemoveInvalidSplats`
2. `OpacityPrune`
3. `FloaterPrune`
4. `QuantizePosition`
5. `QuantizeScale`
6. `QuantizeRotation`
7. `ReduceSHDegree`
8. `MortonSort`
9. `BuildLOD`
10. `ObjectAwarePruneExperimental`

### Acceptance tests

```gherkin
Feature: Optimization passes

Scenario: Opacity pruning removes low-opacity splats
  Given a synthetic scene with 100 splats
  And 20 splats have opacity below threshold
  When I run OpacityPrune with threshold 0.01
  Then output has 80 splats
  And the pass report says removedCount is 20

Scenario: Morton sort is deterministic
  Given the same scene twice
  When I run MortonSort
  Then output ordering is identical

Scenario: Experimental object-aware pruning preserves protected labels
  Given a scene with semantic label "product" on 10 splats
  When I run ObjectAwarePruneExperimental
  Then no splats labeled "product" are removed unless explicitly allowed
```

### Claude Code prompt

```text
Implement SPEC-0006 with the pass framework and first three passes only.
Write unit tests before implementation.
Every pass must emit stats and be toggled through pipeline config.
```

---

## SPEC-0007: Spatial Streaming Index

### Goal

Define a standards-aligned progressive delivery convention using glTF external buffers and optional metadata.

### Requirements

- Chunk assets by spatial locality.
- Use Morton/Z-order for deterministic chunk ordering.
- Store per-chunk bounding boxes and LOD metadata.
- Support checksums.
- Viewer can load root/preview chunks before full asset.

### Acceptance tests

```gherkin
Feature: Spatial streaming index

Scenario: Export chunked glTF
  Given fixture "corpus/small_room.ply"
  When I run "splatforge optimize small_room.ply --preset web-mobile --chunked --out scene.gltf"
  Then scene.gltf references multiple external buffers
  And each chunk has bounding metadata
  And checksums validate

Scenario: Corrupted chunk is detected
  Given a chunked glTF output
  When I flip one byte in a chunk
  Then "splatforge inspect" reports checksum failure

Scenario: Chunk order is deterministic
  Given the same input and config
  When I optimize twice
  Then chunk order and chunk hashes are identical
```

### Claude Code prompt

```text
Implement SPEC-0007.
Do not create a proprietary package extension.
Use glTF external buffers plus optional vendor metadata for chunk index and LOD hints.
Add inspect validation for chunk checksums.
```

---

## SPEC-0008: Viewer SDK

### Goal

Render optimized glTF/SPZ splat assets in a browser with progressive loading.

### Acceptance tests

```gherkin
Feature: Viewer SDK

Scenario: Load glTF splat asset
  Given a valid optimized glTF fixture served over HTTP
  When the viewer loads it
  Then it emits loadStart
  And it requests the glTF file
  And it eventually emits firstRender

Scenario: First render before full load
  Given a multi-chunk glTF fixture
  When the viewer loads it
  Then firstRender fires before complete

Scenario: Missing chunk emits error
  Given a glTF fixture with a missing external buffer
  When the viewer loads it
  Then it emits error with code "chunk_not_found"
```

### Claude Code prompt

```text
Implement SPEC-0008 in packages/viewer.
Use WebGPU as the primary path and WebGL2 fallback as a separate renderer interface.
Start with manifest/chunk loading and lifecycle events before optimizing rendering.
Use Vitest and Playwright.
```

---

## SPEC-0009: Visual Diff Harness

### Goal

Compare source and optimized outputs using deterministic camera paths.

### Acceptance tests

```gherkin
Feature: Visual diff

Scenario: Identical scene has near-zero diff
  Given the same scene as before and after
  When I run visual diff
  Then max frame diff is below epsilon

Scenario: Aggressive pruning fails quality threshold
  Given a scene optimized with size-min preset
  When I run visual diff
  Then the report marks quality as failed if threshold is exceeded

Scenario: HTML report includes before and after frames
  Given a completed visual diff
  Then report.html includes side-by-side frames
```

### Claude Code prompt

```text
Implement SPEC-0009 with Playwright frame capture.
Start with PNG frame capture and simple pixel metrics.
Generate machine-readable JSON and human-readable HTML.
```

---

## SPEC-0010: Viewer Parity Matrix

### Goal

Measure whether the same asset renders consistently across viewer/runtime paths.

### Acceptance tests

```gherkin
Feature: Viewer parity

Scenario: Compare WebGPU and WebGL2 output
  Given an optimized glTF fixture
  When I run viewer parity benchmark
  Then the report includes WebGPU and WebGL2 visual scores

Scenario: Viewer degradation is reported
  Given a fixture known to degrade in WebGL2
  When I run viewer parity benchmark
  Then the report includes a degradation warning
```

### Claude Code prompt

```text
Implement SPEC-0010 as an extension of the visual diff harness.
Compare SplatForge WebGPU vs WebGL2 first.
Design the interface so Three.js/Babylon/PlayCanvas adapters can be added later.
```

---

## SPEC-0011: Hosted API

### Goal

Wrap CLI jobs in a hosted API for upload, analyze, optimize, preview, and download.

### Acceptance tests

```gherkin
Feature: Hosted API

Scenario: Upload and analyze asset
  Given an authenticated API client
  When it uploads a valid PLY
  And starts an analyze job
  Then job status eventually becomes "succeeded"
  And report JSON is available

Scenario: Worker records CLI version
  Given an optimize job
  When it completes
  Then report includes the CLI version used

Scenario: Reject unauthenticated upload
  Given no API key
  When client uploads an asset
  Then API returns 401
```

### Claude Code prompt

```text
Implement SPEC-0011 with Axum and local filesystem storage first.
Keep object storage behind an interface.
All worker jobs must invoke the compiled CLI binary.
```

---

## SPEC-0012: OpenUSD Basics

### Goal

Support a useful subset of OpenUSD Gaussian Splat import/export.

### Acceptance tests

```gherkin
Feature: OpenUSD basics

Scenario: Export SplatIR to USD Gaussian Splat prim
  Given fixture "tiny/basic_binary.ply"
  When I run "splatforge convert tiny/basic_binary.ply --to usd --out scene.usda"
  Then scene.usda exists
  And it declares a Gaussian Splat particle field prim

Scenario: Import supported USD splat scene
  Given fixture "tiny/basic_splat.usda"
  When I run "splatforge analyze tiny/basic_splat.usda"
  Then the report says format is "usd"
```

### Claude Code prompt

```text
Implement SPEC-0012 as a constrained OpenUSD spike.
Use subprocess/CLI integration if direct bindings are too slow to adopt.
Focus on basic round-trip, not full USD composition semantics.
```

---

# Test strategy

## Unit tests

- IR serialization
- field parsing
- glTF extension serialization
- quantization math
- Morton indexing
- pruning decisions
- checksum validation

## Integration tests

- CLI analyze on PLY/SPZ/glTF
- convert PLY → glTF
- convert PLY → SPZ
- optimize → inspect
- visual diff report generation

## Snapshot tests

- JSON reports
- glTF JSON output
- error messages
- optimization reports
- benchmark results

## Property-based tests

- quantize/dequantize error bounds
- quaternion normalization
- no NaN/Inf propagation
- deterministic sorting
- chunk checksum invariants

## Fuzz tests

- PLY headers
- corrupted binary rows
- SPZ decode paths where possible
- malformed glTF extension blocks
- external buffer path traversal
- truncated chunks

## Visual regression tests

- fixed camera paths
- fixed viewport
- WebGPU and WebGL2 capture
- CI browser matrix

## Performance benchmarks

- parse throughput
- conversion throughput
- optimize throughput
- first meaningful paint
- total load time
- FPS
- peak memory

---

# Fixtures and corpus

## Fixture classes

```text
fixtures/
  tiny/
    basic_ascii.ply
    basic_binary.ply
    basic.spz
    basic_khr.gltf
    basic_splat.usda
  invalid/
    missing_rotation.ply
    missing_scale.ply
    nan_position.ply
    extreme_outlier.ply
    floater_cluster.ply
    truncated_binary.ply
    unsupported_khr_version.gltf
  corpus/
    product_scan.ply
    indoor_room.ply
    outdoor_scene.ply
    person_scan.ply
    large_building.ply
  private/
    design_partner_001/
  golden/
    expected_reports/
    expected_gltf/
    expected_frames/
```

## User-specific fixture opportunity

If available, include real-world scans from paddle/club/court environments as private fixtures. These are useful because they contain real spatial complexity, hard edges, nets/screens, outdoor lighting, and background noise.

---

# CI gates

Every PR:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
pnpm lint
pnpm test
pnpm playwright test
splatforge corpus smoke
```

Nightly:

```bash
splatforge corpus full
splatforge benchmark --suite nightly
splatforge visual-regression --suite golden
cargo fuzz run ply_header
cargo fuzz run gltf_extension
cargo fuzz run external_buffer_paths
```

---

# First 20 GitHub issues

1. Define `SplatIR` with deterministic serialization.
2. Create tiny PLY/SPZ/glTF fixtures.
3. Implement binary PLY parser.
4. Implement `splatforge analyze`.
5. Implement SPZ wrapper/import spike.
6. Implement glTF KHR export spike.
7. Implement glTF KHR inspect/import.
8. Implement opacity pruning.
9. Implement floater pruning.
10. Implement position/scale/rotation quantization.
11. Implement Morton spatial sort.
12. Implement external-buffer chunked glTF export.
13. Implement checksum validation for chunks.
14. Implement WebGPU viewer skeleton.
15. Implement WebGL2 fallback skeleton.
16. Implement local `splatforge preview`.
17. Implement visual diff harness.
18. Implement viewer parity WebGPU vs WebGL2.
19. Build hosted analyzer API skeleton.
20. Recruit and onboard 5 design partners.

---

# Claude Code workflow

## Rules for Claude Code

1. Never implement a feature without reading the relevant spec.
2. Add failing tests first.
3. Keep changes scoped to the current spec.
4. Every CLI command needs integration tests.
5. Every JSON output needs schema/snapshot coverage.
6. Every parser change needs malformed fixture tests.
7. Every optimization pass needs before/after stats.
8. Every renderer change needs visual regression coverage.
9. Do not change snapshots unless the spec requires it.
10. Do not introduce a proprietary package format without an explicit product decision.

## Standard Claude Code task template

```text
You are working in the SplatForge repo.
Read specs/SPEC_ID first.
Implement only the acceptance criteria listed there.
Use TDD:
1. Add failing tests.
2. Implement minimum code to pass.
3. Refactor without changing behavior.
4. Run required tests.
5. Update docs if public behavior changed.
Do not add unrelated features.
Do not change snapshot outputs unless required.
```

## Example task: glTF KHR export

```text
Task: Implement basic glTF KHR Gaussian Splatting export.

Read:
- specs/0004-gltf-khr-io.md
- crates/splatforge-core/src/ir.rs

Acceptance:
- PLY fixture converts to scene.gltf
- scene.gltf declares the Gaussian Splatting extension
- inspect succeeds
- generated glTF JSON snapshot is stable

Do not implement chunked external buffers in this task.
```

## Example task: spatial streaming index

```text
Task: Implement Morton-ordered external-buffer chunking for glTF output.

Read:
- specs/0007-spatial-streaming-index.md

Acceptance:
- chunked glTF references multiple external buffers
- each chunk has bounding metadata
- checksums validate
- repeated output is deterministic

Do not create a new .sfz package format.
```

---

# Design partner plan

Design partners belong in Phase 0.

Target categories:

- capture tools
- real estate scanning companies
- 3D commerce teams
- digital twin companies
- creative tool/plugin developers
- web 3D engine teams

Outreach message:

> We are building the production optimization and delivery layer for Gaussian Splats. If you have `.ply`, `.spz`, or glTF splat assets that are too large, slow, or inconsistent across viewers, we can run them through our benchmark pipeline and return a before/after report with size, load time, memory, FPS, and visual diffs.

Design partner questions:

1. What formats do you ingest/export today?
2. What are typical file sizes and splat counts?
3. What runtime do you target?
4. Where does delivery fail today?
5. Do you care most about first render, total load, FPS, memory, or quality?
6. Do you need streaming or just smaller downloads?
7. Do you need CI validation?
8. Would an API integration fit your workflow?
9. Would you prefer glTF, SPZ, OpenUSD, or multiple outputs?
10. Would you pay for hosted optimization, SDK licensing, or private benchmarks?

---

# Capture-tool partnership strategy

This is a named GTM workstream, not a footnote.

Targets:

- Luma-like capture tools
- Polycam-like scanning tools
- Scaniverse-like mobile scanning tools
- PostShot-like training/export tools
- SuperSplat/PlayCanvas-like editing/publishing tools
- Spline-like web 3D publishing tools

Integration idea:

> “Optimize for web/mobile with SplatForge” button at export time.

Partnership value:

- Capture tools avoid building their own optimizer.
- SplatForge gets distribution at the exact moment of user pain.
- Users get measurable delivery improvements.

---

# Risk register

## Risk: Standards move faster than expected

Mitigation:

- Stay standards-first.
- Make SplatForge the best optimizer on top of the winning standards.
- Keep IR independent of input/output formats.

## Risk: Capture tools build optimization in-house

Mitigation:

- Become their API/integration before they build it.
- Offer white-label SDK and batch API.
- Maintain better benchmark corpus than any single capture tool.

## Risk: Apple ships native USDZ/splat optimization

Mitigation:

- Own cross-platform delivery.
- Support non-Apple targets.
- Publish device-budget compatibility data.

## Risk: Cloudflare or another infra company ships hosted splat optimization

Mitigation:

- Own OSS CLI, CI integration, and benchmark corpus.
- Become a standards/reference layer, not just a hosted service.

## Risk: PLY becomes obsolete

Mitigation:

- IR should not care.
- Value is in optimization, validation, benchmarks, and output quality.

## Risk: Viewer/editor companies absorb the pipeline

Mitigation:

- Focus on CI, reports, benchmarks, device budgets, and multi-format output.
- Integrate with editors rather than competing with them.

## Risk: Visual quality metrics are insufficient

Mitigation:

- Human-readable reports.
- Fixed camera paths.
- Viewer parity tests.
- Design-partner validation.

## Risk: Hosted processing unit economics are bad

Mitigation:

- Cost model before Phase 4.
- Usage caps.
- Local CLI for heavy users.
- Enterprise/private deployment option.

---

# MVP demo script

1. Upload `warehouse_raw.ply`, 620 MB.
2. SplatForge analyzes it:
   - 9.2M splats
   - high opacity redundancy
   - floater clusters detected
   - mobile memory risk
   - recommended `web-mobile`
3. Run optimize.
4. Output:
   - 620 MB PLY → 54 MB optimized glTF/SPZ output
   - first meaningful paint: 7.8s → 1.4s
   - peak memory: 1.2 GB → 420 MB
   - FPS: 22 → 58 on reference laptop
   - visual diff: pass
5. Preview side-by-side.
6. Show viewer parity matrix:
   - Chrome/WebGPU: pass
   - Safari/WebGPU: pass with warning
   - WebGL2 fallback: quality degradation warning
7. Copy React/Web viewer embed snippet.

---

# Final recommendation

Build SplatForge in this order:

1. Design partners and real corpus.
2. SplatIR.
3. PLY/SPZ/glTF KHR I/O.
4. Analyzer.
5. Basic optimizer.
6. glTF external-buffer chunking and spatial index.
7. WebGPU viewer.
8. Visual diff.
9. Viewer parity matrix.
10. Hosted analyzer/optimizer.
11. OpenUSD basics.
12. Capture-tool integrations.
13. SplatBench public leaderboard.
14. Enterprise CI/private corpus.

The most important changes from the earlier plan are:

- no proprietary `.sfz` default format
- glTF KHR Gaussian Splatting moves to Phase 1
- OpenUSD moves earlier
- WebGPU becomes primary, not experimental
- benchmark corpus becomes the core moat
- design partners start in Phase 0
- capture-tool partnerships become a dedicated GTM workstream
- object-aware pruning becomes an optional differentiator tied to CV expertise

The product should be judged by one north-star metric:

> **Optimized Gaussian Splat GB delivered per month through SplatForge-generated assets/viewer integrations.**

