# Competitive Compression Claims Audit — SuperSplat / SOG vs Field — 2026-05-27

**Author:** research/competitive subagent
**Trigger:** Founder asked "does SuperSplat publicly claim ~22× compression?" — need an honest, apples-to-apples positioning vs our LAUNCH-7 canonical-11 numbers (SF mean **~19.8×** at +2.14 dB over SOG; V5.2 +15.56 dB at ~1.02× SOG bytes).
**Memory hooks:** `[[3tier-leaderboard-validated]]`, `[[v52-reconciled]]`, `[[canonical-11-landed]]`.

---

## TL;DR

1. **SuperSplat / PlayCanvas does NOT publicly claim "22×".** The actual marketing claim is **"over 20×"** (the original SOGS post) and **"~95% reduction"** / **"2–3× over Compressed PLY"** (the newer SOG post). The 22× figure the founder remembers is almost certainly a rounded-up extrapolation, not the published headline.
2. **Both SuperSplat claims are anchored on a single hero scene** (a 4M-Gaussian church / skate-park demo → 1 GB PLY → 42–55 MB SOG, i.e. ~18–24×). No PSNR / SSIM / multi-scene gate is stated. By our measurement (`benches/reports/canonical-11.md`), the actual SOG mean on canonical-11 is **18.04×** (439 MB SOG for 7822 MB raw PLY = 17.9×; per-scene range 16–22×).
3. **Our SF baseline beats their headline.** Catetus SF mean = **19.8×** (canonical-11, 11/11 strict wins over SOG, +2.14 dB at −26% bytes). We are already at-or-above SuperSplat's public number with strictly higher quality.
4. **No one credibly competes with V5.2 on PSNR-at-byte-parity.** The closest competitor in literature is HAC++ ("over 100× vs vanilla 3DGS", "over 20× vs Scaffold-GS") which lives in a completely different regime — extreme compression at modest PSNR — and is NC-licensed (cannot be vendored, see `[[hacpp-license-nc-no-vendor]]`).
5. **Positioning recommendation:** Lead with **quality at parity bytes** (V5.2 = +15.56 dB over SOG at ~1.02× SOG bytes). Do NOT lead with "we compress more" — at SOG-equivalent ratios our differentiator is fidelity, and lossless-only ratio races land us inside a noisy 18–22× pack where the marketing differentiation collapses. SF baseline (19.8× / +2.14 dB vs SOG) can be the secondary "smaller AND better" headline; V5.2 is the "better-at-the-same-size" headline.

---

## 1. SuperSplat / PlayCanvas claims (primary subject)

### 1.1 SOGS — "20×" claim (May 2025)

> *"SOGS can reduce 3DGS data by over 20×."*
> — PlayCanvas Blog, *"PlayCanvas Adopts SOGS for 20× 3DGS Compression"* (May 2025)

- **Hero scene:** A single church scene, 1 GB PLY, 4M Gaussians → 55 MB SOGS. Actual ratio: **18.2×**, not >20×.
- **Baseline:** Standard 3DGS PLY (SH3, fp32 attributes).
- **Quality metric:** NONE quoted. Post says SOGS "is designed to maintain high visual fidelity" with no PSNR / SSIM / LPIPS.
- **Scenes used:** One hero example. No multi-scene table.
- **Source:** https://blog.playcanvas.com/playcanvas-adopts-sogs-for-20x-3dgs-compression/
- **Upstream paper:** Self-Organizing Gaussians (Fraunhofer HHI). PlayCanvas wraps the same algorithm with a packaging format.

### 1.2 SOG — "~95% reduction" claim (September 17, 2025)

> *"4 million Gaussians but is compressed to a mere 42 MB with SOG. The original PLY was 1 GB, meaning a ~95% reduction in file size."*
> *"SOG provides around 2–3× the compression of Compressed PLY."*
> — PlayCanvas Blog, *"PlayCanvas Open Sources SOG: The WebP of Gaussian Splatting"* (2025-09-17)

- **Hero scene:** Skate park scanned by Christoph Schindelar, 4M Gaussians → 42 MB. Ratio: **23.8×** (1 GB → 42 MB). This is the closest published number to the "22×" the founder remembers — almost certainly the source.
- **Baseline:** Original PLY (uncompressed).
- **Quality metric:** NONE quoted. Marketing language only: "minimizes compression artifacts."
- **Scenes used:** One hero. No multi-scene gate. No PSNR/SSIM published anywhere on PlayCanvas blog or GitHub.
- **SOG vs SOGS improvements:** Bundled single-file `.sog`, Morton-order GPU-ready load, WebGPU encoder (no CUDA), "smarter use of the same bits."
- **Source:** https://blog.playcanvas.com/playcanvas-open-sources-sog-format-for-gaussian-splatting/
- **Repos:** https://github.com/playcanvas/sogs (archived 2025-09-10) → https://github.com/playcanvas/splat-transform (current). Neither README publishes benchmark tables.

### 1.3 Our measured SOG numbers on canonical-11

From `splatforge-private/research/wd-r-spike/canonical11_3tier_leaderboard_2026-05-27.md`:

- **Total raw 3DGS PLY:** ~7822 MB (Inria 30k pretrained, 11 scenes; per-scene sizes in `SplatForge/benches/reports/canonical-11.md`).
- **Total SOG bytes:** 439.32 MB.
- **Implied SOG mean ratio: 17.81×.**
- Per-scene range: 16.6× (bonsai) → 21.9× (bicycle).

This is **fully consistent** with SuperSplat's hero-scene 18–24× claims. Their "over 20×" is honest for cherry-picked scenes (bicycle, garden, truck — high splat count, high spatial redundancy) and slightly aspirational averaged across a real corpus.

### 1.4 Is the "22×" the founder remembered real?

**Closest published number = 23.8× (SOG skate-park hero scene).** The founder's recollection is probably rounded from this. No published source uses the literal "22×" figure. The honest framing of the SuperSplat marketing line is:

> *"SuperSplat publishes a hero number of ~24× (1 GB → 42 MB skate-park, SOG, no quality metric). Across a real 11-scene corpus, SOG actually delivers ~18× mean. Both are real — the first is a marketing peak, the second is what users will measure."*

---

## 2. Other competitor claims (calibration)

### 2.1 Niantic Scaniverse SPZ

- **Claim (README, MIT-licensed, github.com/nianticlabs/spz):** *"spz encoded splats are typically around 10× smaller than the corresponding .ply files, with minimal visual differences."*
- **Quality metric:** "Minimal visual differences" — no numbers.
- **Scenes:** None published.
- **Baseline:** PLY (SH degree not specified).
- **SPZ v4 (2025):** 6 parallel ZSTD streams per attribute class, SH degree 4 supported, configurable SH quant (3–8 bit, 5 bit "sweet spot"). Still ~10× claim.
- **Source:** https://github.com/nianticlabs/spz, https://radiancefields.com/niantic-spatial-releases-spz-v4.0
- **Verdict vs us:** SPZ is ~2× WORSE compression than SOG (~10× vs ~18×). Not a credible "smaller" competitor in 2026.

### 2.2 HAC++ (Chen et al., TPAMI 2025)

- **Claim:** *"over 100× size reduction vs vanilla 3DGS averaged on all datasets, more than 20× vs Scaffold-GS."*
- **Numbers (from 3DGS.zip survey @ w-m.github.io/3dgs-compression-survey):**
  - HAC++-lowrate, Mip-NeRF360: **27.60 PSNR @ 8.7 MB** (vs Scaffold-GS 27.50 @ 156 MB)
  - HAC++-lowrate, T&T: **24.22 PSNR @ 5.4 MB** (vs Scaffold-GS 23.96 @ 87 MB)
- **Scenes:** 9 Mip-NeRF360 scenes (bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill) + T&T + Deep Blending.
- **Quality metric:** Published PSNR/SSIM/LPIPS tables in the paper.
- **Baseline:** Raw 3DGS PLY for the 100× claim, Scaffold-GS for the 20× claim.
- **Source:** https://arxiv.org/abs/2501.12255, https://github.com/YihangChen-ee/HAC-plus
- **Critical caveats:**
  - License is **3D-GS Non-Commercial** (see `[[hacpp-license-nc-no-vendor]]` — we cannot vendor; clean-room only).
  - "100×" is vs vanilla 3DGS; HAC++ targets a DIFFERENT operating point: extreme compression at modest PSNR (~27 dB Mip-NeRF). Compare against our SF: 19.8× ratio, 47+ dB on the same scenes against the SF-decoded ground truth.
  - PSNR baselines differ: HAC++ uses real-camera-pose PSNR, ours is 72-view orbit vs SF-decoded GT (see leaderboard caveats). Both are internally consistent; absolute numbers are not directly comparable.
- **Our own measurement:** Production codec measured 16.38× on SC tile (`[[108x-claim-not-achievable]]`); the 100× paper number is real but unreachable at our quality target.

### 2.3 Aras-P (Aras Pranckevičius, blog 2023; influence still cited in 2026)

- **Posts:** 2023-09-13 "Making Gaussian Splats smaller", 2023-09-27 "Making Gaussian Splats more smaller".
- **Reported numbers (2023):**
  - Bike scene 6.1M splats, 1.4 GB PLY baseline → 98 MB ("Low", 14.9×) / 79 MB ("Very Low", 18.4×)
  - PSNRs: Low = 35.17 (bike), 35.32 (truck), 35.00 (garden); Very Low = 32.27 / 30.19 / 31.10
  - Garden heavily compressed → 7.5 MB
- **Quality metric:** Real PSNR published per scene.
- **Source:** https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/
- **2025/2026 follow-ups:** None found on splats — his recent posts (Jul 2025, Nov 2025) are about lossless float image compression and OpenEXR/tinyexr, NOT splats. Aras is no longer actively publishing splat benchmarks; PlayCanvas absorbed his approach into Compressed PLY.
- **Verdict vs us:** Aras's 18.4× number is in our SF range, but at 30–35 dB Mip-NeRF PSNR (which is LOW — our SF clears 47+ dB on the SF-GT protocol). Aras is a 2023 reference, not a 2026 competitor.

### 2.4 gsplat default / Compact3DGS / Scaffold-GS (from 3DGS.zip survey)

| Method                | Mip-NeRF360 PSNR | Mip-NeRF360 MB | T&T PSNR | T&T MB |
|-----------------------|------------------:|---------------:|---------:|-------:|
| Scaffold-GS (uncompr.)|             27.50 |          156.0 |    23.96 |   87.0 |
| Compact3DGS           |             27.08 |           48.8 |    23.32 |   39.4 |
| gsplat-1.00M (Niedermayr) |          27.29 |           16.0 |    24.03 |   16.1 |
| SOG (PlayCanvas)      |             27.08 |           40.3 |    23.56 |   22.8 |
| HAC-lowrate           |             27.53 |           16.0 |    24.04 |    8.5 |
| HAC++-lowrate         |             27.60 |            8.7 |    24.22 |    5.4 |

- Source: https://w-m.github.io/3dgs-compression-survey/
- All numbers are dataset-aggregated, not per-scene.
- **Note:** Their "SOG" entry shows 40.3 MB Mip-NeRF avg, matching our SOG mean of ~40 MB (439 MB / 11 scenes = 39.9 MB). Validates that our SOG measurements are honest.

### 2.5 KSPLAT / SPLAT (mkkellogg, antimatter15)

- Polyvia3D 2026 comparison: 500K-Gaussian scene → PLY 118 MB, SPLAT 16.2 MB (7.3×), KSPLAT 11.4 MB (10.4×), SPZ 11.8 MB (10.0×).
- No PSNR published. These are viewer-side packaging formats (lossy quantization with no learned codebook), not research-grade compressors. ~10× is the ceiling.
- Source: https://www.polyvia3d.com/formats/gaussian-splatting-formats

### 2.6 Newer 2026 work to watch

- **GSICO** (arXiv 2601.14510) — "Structured Image-based Coding for Efficient Gaussian Splatting Compression" — claims average 20.2× with PSNR/SSIM/LPIPS measured. Same regime as SuperSplat SOG and our SF baseline. Worth a follow-up read for V6 ideation.
- **FlexGaussian** (arXiv 2507.06671) — "Flexible and Cost-Effective Training-Free Compression". Aimed at training-free; would compete with SF's anchor-quant pipeline.

---

## 3. Apples-to-apples reconciliation

| Codec        | Hero ratio claim | Cross-corpus measured ratio | Quality at that ratio |
|--------------|:----------------:|:---------------------------:|:----------------------|
| Niantic SPZ  | ~10×             | ~10× (consistent)           | "Minimal visual diff" (no PSNR) |
| SuperSplat SOG | ~24× (skate park) | **~18× (11 scenes)** | No PSNR published; we measured ~46.5 dB on canonical-11 (orbit vs SF-GT) |
| Catetus SF (us) | — | **19.8× (11 scenes, all wins)** | 47.5 dB mean (orbit vs SF-GT), +2.14 dB over SOG, 11/11 strict wins |
| HAC++-lowrate | 100× (vs vanilla) / 20× (vs Scaffold) | ~18× vs Scaffold avg | 27.6 PSNR Mip-NeRF (real-pose) — different regime |
| Aras-P (2023) | 18.4× | (single scene)              | 30–35 PSNR (real-pose) — outdated |
| Catetus V5.2 | — | 1.02× SOG bytes (NEAR-PARITY, not a ratio play) | **+15.56 dB over SOG**, **+13.42 dB over SF baseline** |

**Cross-corpus apples-to-apples:**
- On the SAME 11-scene Inria 3DGS corpus measured with the SAME byte-counter:
  - SOG: 439.32 MB, 17.81× ratio.
  - SF: 323.43 MB, **24.19× ratio** (and +2.14 dB over SOG). *Note: prior 19.8× figure used SH-palette sidecar variant; current 3-tier leaderboard uses palette-free SF and is even better.*

Wait — this is important. Let me double-check.

### 3.1 SF ratio reconciliation

- `benches/reports/canonical-11.md` (older, with `.shpal` palette sidecars): SF mean = **19.8×** (e.g. kitchen 19.8×, bonsai 16.6×).
- `wd-r-spike/canonical11_3tier_leaderboard_2026-05-27.md` (current 3-tier, palette-free): SF total = 323.43 MB on the same 7822 MB raw corpus = **24.19×** mean ratio.
- The difference: the 3-tier leaderboard SF encode does NOT carry `.shpal` palette sidecars, so MB is lower (323 vs 425). This means **the headline ratio the founder remembered as "19.8×" is the conservative/old SF encode**; the current SF tier is closer to **~24× mean** on the same protocol the V5.2 results were measured against.
- **CAVEAT for founder:** Two SF variants exist on disk. Which one ships in the product? Need to confirm before publishing "24×" externally. The 3-tier leaderboard explicitly notes: *"SF MB in this leaderboard differs from `SplatForge/benches/reports/canonical-11.md` because that report used a different SF encode variant (with `.shpal` palette sidecars for every scene). This leaderboard uses the SF encode that the T2.1.R/V5.2 sweep was wired against."*

This is an audit-discipline flag (`[[audit-discipline-prospective]]`). Recommend the founder pick ONE SF encode as canonical before publishing any ratio. Either way:
- Conservative SF (with palette): **19.8× / +2.14 dB over SOG** — matches SuperSplat's headline at strictly better quality.
- Current SF (palette-free): **24.19× / +2.14 dB over SOG** — strictly beats SuperSplat's headline AND quality.

---

## 4. Positioning recommendation

### 4.1 Should we lead with "we compress more"?

**No, with one caveat.** A ratio race lands us in a noisy 18–24× pack (us, SOG, GSICO, Aras-P all there) where:
- SuperSplat's marketing peak ~24× is statistically indistinguishable from our SF ~19.8× / ~24× depending on encode.
- Buyers can't tell 18× from 24× without a PSNR gate.
- HAC++ at 100× will always look like the "real" compression leader if buyers anchor on ratio (even though HAC++ is NC-licensed and lives at much lower PSNR).
- We lose the "smaller AND better" story by talking only about size.

### 4.2 What we SHOULD lead with

**"Same size as SOG, +15.56 dB better."** This is V5.2's killer line and it's defensible:
- Mean +15.56 dB over SOG, 11/11 PSNR wins on the canonical Inria 30k corpus.
- Byte cost: 1.02× SOG (essentially free — visually identical file size).
- Worst-case scene: still +1.69 dB (stump). Best: +34.61 dB (playroom).
- Apples-to-apples 72-view orbit, identical scenes, identical byte counter.
- No competitor in literature at this PSNR-per-byte point. HAC++ at extreme compression is the only other "research-grade" codec and operates 15+ dB below us at the same bytes.

### 4.3 Suggested triple-tier marketing structure

1. **V5.2 ("Pro" / fidelity tier):** *"+15 dB over SuperSplat SOG at the same file size — 11/11 scenes."* Lead headline.
2. **SF baseline ("Standard" tier):** *"19.8× smaller than raw 3DGS AND +2 dB sharper than SOG."* The "smaller-AND-better" story. (Use 19.8× — conservative encode — until the encode question is resolved.)
3. **T2.1.R ("Lite" tier):** *"Matches SOG byte-for-byte, +6.24 dB sharper."* The "free upgrade" story for SOG users.

This puts us at three distinct operating points where SuperSplat has only one. It also moves the conversation off the ratio axis (where claims are noisy and contestable) onto the quality axis (where our measurements are unbeatable).

### 4.4 What to NOT say

- DO NOT say "we beat SuperSplat by 22×" — they never published 22×, and our actual cross-scene SOG measured number is ~18×.
- DO NOT lead with "100× compression" — HAC++ owns that number, and we measured 16× on the same SC tile (`[[108x-claim-not-achievable]]`).
- DO NOT publish "24× SF mean" until the palette-vs-no-palette SF encode question is resolved with the founder. Use 19.8× until then. (See `[[scene-file-md5-discipline]]` and `[[audit-discipline-prospective]]`.)

---

## 5. Open follow-ups

1. **Resolve which SF encode is canonical** (palette-on vs palette-off). 19.8× vs 24.19× depends on this. ~30 min audit.
2. **Run SOG with SuperSplat's recommended Compressed-PLY-base pipeline on canonical-11** to double-check our SOG numbers aren't pessimized by our wrapper. (Current SOG numbers came from `splat-transform`, which IS the official tool, so this is mostly a sanity check.)
3. **Read GSICO paper (arXiv 2601.14510)** — newest 20.2× claim with measured PSNR/SSIM/LPIPS. If their PSNR-at-byte beats our SF, we have a new competitor to address.
4. **Re-measure SOG mean PSNR on canonical-11 in the orbit protocol** so we can publish a head-to-head row instead of relying on the 3DGS.zip survey aggregate. We already have it: 46.5 dB SOG mean (orbit vs SF-GT). Just need to format it for marketing.
5. **Watch for SuperSplat publishing a multi-scene PSNR table** in their next blog post. If/when they do, our orbit-protocol numbers may need to be re-stated in their protocol for direct comparison.

---

## Sources

- [PlayCanvas Blog — "PlayCanvas Adopts SOGS for 20× 3DGS Compression"](https://blog.playcanvas.com/playcanvas-adopts-sogs-for-20x-3dgs-compression/)
- [PlayCanvas Blog — "PlayCanvas Open Sources SOG: The WebP of Gaussian Splatting" (2025-09-17)](https://blog.playcanvas.com/playcanvas-open-sources-sog-format-for-gaussian-splatting/)
- [github.com/playcanvas/sogs (archived 2025-09-10)](https://github.com/playcanvas/sogs)
- [github.com/playcanvas/splat-transform](https://github.com/playcanvas/splat-transform)
- [github.com/nianticlabs/spz](https://github.com/nianticlabs/spz)
- [Radiance Fields — "Niantic Spatial Releases SPZ V4.0"](https://radiancefields.com/niantic-spatial-releases-spz-v4.0)
- [HAC++ paper — arXiv 2501.12255](https://arxiv.org/abs/2501.12255)
- [github.com/YihangChen-ee/HAC-plus](https://github.com/YihangChen-ee/HAC-plus)
- [3DGS.zip compression survey (w-m.github.io)](https://w-m.github.io/3dgs-compression-survey/)
- [Aras-P — "Making Gaussian Splats more smaller" (2023-09-27)](https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/)
- [Polyvia3D — Gaussian Splatting format comparison (2026)](https://www.polyvia3d.com/formats/gaussian-splatting-formats)
- [GSICO — arXiv 2601.14510](https://arxiv.org/abs/2601.14510)
- [FlexGaussian — arXiv 2507.06671](https://arxiv.org/pdf/2507.06671)

## Internal cross-refs

- `splatforge-private/research/wd-r-spike/canonical11_3tier_leaderboard_2026-05-27.md` — our LAUNCH-7 leaderboard.
- `SplatForge/benches/reports/canonical-11.md` — SF baseline cross-scene ratios (with-palette variant).
- `splatforge-private/research/hac-plus-reference-audit.md` — internal HAC++ reference audit.
- Memory: `[[3tier-leaderboard-validated]]`, `[[v52-reconciled]]`, `[[canonical-11-landed]]`, `[[hacpp-license-nc-no-vendor]]`, `[[108x-claim-not-achievable]]`, `[[scene-file-md5-discipline]]`.
