# Catetus Preset Cheatsheet (LLM Quick-Pick)

**Resource URI:** `catetus://docs/preset-cheatsheet`
**Audience:** LLMs and humans picking a preset for the `optimize` / `encode` tools.
**Decision style:** "If X then Y." Designed for in-context inclusion (under ~150 lines).

---

## 30-second decision tree

```
START: what is the OUTPUT for?
├── Web (browser):
│   ├── mobile-first?          → web-mobile             (free, ~22x, SH1)
│   ├── desktop-first?         → web-desktop            (free, ~12x, SH2)
│   └── hero / marquee scene?  → v52-quality (paid)     OR quality-max (free, SH3)
├── XR:
│   ├── Meta Quest browser?    → quest-browser          (free, ~26x, aggressive prune)
│   └── Apple Vision Pro?      → visionos-preview       (free, ~16x, color-managed)
├── Catalog / thumbnail grid?  → thumbnail-preview      (free, ~50x, SH0)
├── Bandwidth-constrained?     → size-min               (free, ~32x, lossy)
├── Archive / reproducibility? → lossless-repack        (free, 1x, bit-exact)
├── Paid head-to-head with SOG?→ v52-quality            (paid, +15.56 dB vs SOG)
├── Throughput-bound batches?  → t21r-fast              (paid, +13 dB, anchor-only)
└── Mobile-budget fidelity?    → differentiable-repack  (paid, +3-5 dB at web-mobile bytes)
```

---

## By constraint type

### "I have a byte budget"

| Budget per scene | Recommended free preset | Recommended paid preset |
|---|---|---|
| ≤ 5 MB        | `size-min`           | `t21r-fast`           |
| 5–15 MB       | `web-mobile`         | `v52-balanced`        |
| 15–40 MB      | `web-desktop`        | `v52-quality`         |
| > 40 MB       | `quality-max`        | `v52-quality`         |
| no limit      | `lossless-repack`    | `v52-quality`         |

### "I have a quality floor"

| Required mean ΔE94 | Recommended preset |
|---|---|
| 0 (lossless)        | `lossless-repack`   |
| < 0.5               | `quality-max` or `v52-quality` (paid)  |
| < 1.0               | `web-desktop` / `differentiable-repack` |
| < 3.0               | `web-mobile`       |
| any (preview only)  | `thumbnail-preview` / `size-min` |

### "I have a target device"

| Device | Free | Paid |
|---|---|---|
| Mobile web (iOS Safari, Chrome Android) | `web-mobile`       | `differentiable-repack` |
| Desktop web                              | `web-desktop`      | `v52-balanced`         |
| Meta Quest browser                       | `quest-browser`    | (paid not yet tuned)   |
| Apple Vision Pro (Reality Composer Pro)  | `visionos-preview` | (paid not yet tuned)   |

---

## What each preset does (one-liner)

| Preset | Tier | Passes (high level) | Headline |
|---|---|---|---|
| `lossless-repack` | free | repack | Bit-exact PLY→glb. Reference baseline. |
| `web-mobile` | free | SH1, q8, spz | Mobile sweet spot, ~22x. |
| `web-desktop` | free | SH2, q10, spz | Desktop sweet spot, ~12x. |
| `quest-browser` | free | floater-prune, SH1, q8, spz | Quest 3 GPU-budget tuned. |
| `visionos-preview` | free | SH2, P3-color-manage, q10, gltf | Vision Pro glb. |
| `thumbnail-preview` | free | decimate, SH0, q6, spz | Tiny previews, ~50x. |
| `quality-max` | free | q12, spz | SH3, near-lossless visually. |
| `size-min` | free | prune, decimate, SH-trim, q6, spz | Smallest free output, ~32x. |
| `differentiable-repack` | paid | gsplat self-distill at q8 | +3–5 dB at web-mobile bytes. |
| `v52-quality` | paid | T2.1.R + V5.2 tail | +15.56 dB vs SOG at SOG bytes. |
| `v52-balanced` | paid | T2.1.R + light tail | +10 dB vs SOG at 0.85x SOG bytes. |
| `t21r-fast` | paid | T2.1.R anchor only | +13.10 dB vs SOG, fastest paid. |

---

## Anti-patterns (do NOT pick these)

- **Picking `size-min` for hero content.** Visible artifacts. Use `web-mobile` minimum.
- **Picking `web-mobile` for VR.** Desktop-tuned compression on a Quest causes texture pop. Use `quest-browser`.
- **Picking `lossless-repack` to "save space".** It does NOT compress. It's a format conversion.
- **Picking `v52-quality` for a thumbnail grid.** You'll pay paid-tier credits for output a user glances at for 0.4s. Use `thumbnail-preview`.
- **Picking `quality-max` AND `differentiable-repack`.** They target different operating points. `quality-max` if you want near-lossless free. `differentiable-repack` if you want web-mobile bytes with +3 dB.

---

## How to pick programmatically

1. Call `analyze` on the input scene. Note `splatCount`, `likelyClass`, `needsFloaterPrune`.
2. If `splatCount > 5M` AND target is mobile → pick `web-mobile` (or `differentiable-repack` if paid).
3. If `likelyClass == "transparent-volume"` → DO NOT use `size-min` (kills opacity gradients). Use `web-desktop` or `v52-quality`.
4. If `needsFloaterPrune == true` → choose a preset whose `passes` includes `floater-prune` (`quest-browser`, `size-min`) — or run `validate_pipeline` after.
5. If unsure → call `recommend_preset` with your constraints; that tool runs a 3-line predictor over the corpus and returns the best preset + rationale + 2 alternatives.

---

## Quick numeric anchors (from canonical-11 leaderboard)

- SOG mean: **17.81x**, ~46.5 dB PSNR (orbit vs SF-decoded GT).
- SF baseline mean: **19.8x**, **+2.14 dB over SOG**, **11/11 strict wins**.
- T2.1.R: ~1.02x SOG bytes, **+13.10 dB over SOG**.
- V5.2: ~1.02x SOG bytes, **+15.56 dB over SOG**, **11/11 strict wins**.
- Best V5.2 scene: playroom **+34.61 dB over SOG**.
- Worst V5.2 scene: stump **+1.69 dB over SOG** (still positive).

Source: `catetus://bench/canonical-11` and `catetus://corpus/3-tier-comparison`.

---

## See also

- `catetus://presets/catalog` — full preset metadata (passes, target devices).
- `catetus://corpus/competitor-codecs` — how each preset stacks vs SOG / SPZ / HAC++.
- `catetus://bench/canonical-11` — measured numbers per scene per preset.
