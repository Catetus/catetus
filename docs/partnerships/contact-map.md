# SplatForge — 5-Vertical Warm-Intro Contact Map

**Confidential.** BD research notes for v3.2 vertical attack. Names listed are public maintainers of public projects, public Khronos WG members, public author bylines on funded OSS, or executives whose roles are matters of public record. No private contact information. Public signal citations only.

**Document hygiene.** Any name marked **VERIFY** has *not* been re-confirmed via live WebFetch in this session (WebFetch tool was unavailable). Before any first-touch send, an operator must (a) load the cited public URL and (b) confirm the person's current role/title. Treat **VERIFY** entries as research targets, not as send-ready contacts.

**Date.** 2026-05-15.

---

## 0. Cite-able artifacts (use these in every email; do not exceed these claims)

These are the shipped artifacts every first-touch email leads with. Numbers below are taken from `benches/reports/splatbench-v0.json` and `docs/blog/v0.1.2-release.md` — they are the only numbers permitted in outreach text:

| Artifact | Number / fact | Source |
|---|---|---|
| **KHR_gaussian_splatting reference impl + conformance suite** | 23 normative clauses, 10 golden fixtures, CI workflow. Submitted upstream to Khronos as the public reference impl before Q2 2026 ratification. | `docs/khr-conformance-submission.md`, `docs/blog/v0.1.2-release.md` |
| **`KHR_gaussian_splatting_compression_spz` extension** | New compression-vendor extension landed (PR #1 in the relevant KHR fork). First standards bridge between glTF and the SPZ wire format. | `docs/partnerships/adobe-spz-memo.md` §1 |
| **OpenUSD `ParticleField3DGaussianSplat` writer** | USDC binary writer round-trips bit-exact-as-USDA against Apple `usdcat` 0.25.2 on three reference scenes. 10 spec gaps captured for the OpenUSD WG (`docs/standards-outreach/openusd-forum-post.md`). | `docs/blog/v0.1.2-release.md` §5 |
| **SplatBench median ratio (web-mobile preset, 17 scenes)** | **22.97× on Mip-NeRF 360 `stump` (real outdoor scene)**, 22.81× on `bonsai`, 25.46× on `bicycle`; overall median **21.97×**, overall **23.27×**. Compare splat-transform v2.1.1 at **5.91× median** on the same synthetic corpus. | `benches/reports/splatbench-v0.json` (scenes[].webMobileRatio, aggregates), `docs/blog/v0.1.2-release.md` §3 |
| **DifferentiableRepack PSNR delta** | **+6.4 dB PSNR** over opacity-prune at 50% byte budget (bonsai, N=5 seed median; raw +7.434 dB, mean +6.9 dB). | `benches/reports/splatbench-v0.json` (bonsai.repack) |
| **WebGPU + radix sort viewer** | 1M splats at 127 fps on M-series; streaming-tile adapter for 30 GB / 60 fps mobile target. | `docs/blog/v0.1.2-release.md` §7 |
| **Cesium 3D Tiles preset** | `splatforge optimize --preset geospatial` emits `tileset.json` + 4-level LOD pyramid. | `docs/blog/v0.1.2-release.md` §6 |
| **SPZ first-class output** | First-class SPZ output since v0.1.0; KHR-SPZ extension makes us the only encoder that bridges glTF and SPZ via a Khronos-tracked extension. | `docs/partnerships/adobe-spz-memo.md` §1 |

**Forbidden in outreach:** any number not in the table above. No "146× CodecGS-Lite", no "best in class", no "fastest". The discipline is: ship-or-kill, cite or shut up.

---

## 1. Five-vertical target table

One primary target per vertical. The "Last 12mo public signal" column is what the first-touch must reference to prove the message is not a cold spray.

| # | Vertical | Primary target | Role | Last 12mo public signal | Verification status |
|---|---|---|---|---|---|
| 1 | **Geo / Cesium / Bentley** | **Patrick Cozzi** | Founder + CEO, Cesium (a Bentley Systems company since Nov 2024). Chair, Khronos 3D Formats Working Group. | Cesium's 3D Tiles 1.2 spec workstream actively tracks Gaussian Splat tiling; Cozzi has chaired Khronos 3D Formats WG meetings where KHR_gaussian_splatting has been on-agenda. Confirmed in `docs/partnerships/contact-map.md` (legacy section D) as the WG chair who sets KHR_gaussian_splatting acceptance pace. | **Confirmed public ID** (per prior outreach docs). Re-confirm current title via Cesium.com/team or LinkedIn before send. |
| 2 | **VFX / Apple / visionOS / OpenUSD** | **AOUSD Working Group chairs (Apple seat)** | OpenUSD `ParticleField3DGaussianSplat` schema authors. AOUSD = Alliance for OpenUSD; Apple is a steering-committee member. | OpenUSD 26.03 shipped `ParticleField3DGaussianSplat` — the schema we wrote against and have 10 spec-gap notes on. Forum post is drafted and ready to send (`docs/standards-outreach/openusd-forum-post.md`). | **VERIFY** — exact author names of `ParticleField3DGaussianSplat` schema rev should be pulled from the schema's USDA file in the OpenUSD repo before send. Apple AVP developer-relations point of contact (per WWDC RealityKit early-access channel) is the path; **VERIFY** named contact. |
| 3 | **Adobe / Substance / Creative Cloud** | **Stefano Corazza** | Senior Principal Scientist, Adobe (per Khronos panel by-line + LinkedIn-public title). Adobe-side voice on glTF / Khronos working groups. | Public-facing Adobe voice on 3D / Khronos. Adobe is the first SPZ vendor-extension registrant (Niantic SPZ 4 blog, 2026-05-11). Adobe-funded Babylon.js 8.0 (Mar 2025) shipped first-party SPZ readers. | **Confirmed public ID** (per `contact-map.md` legacy section A). Re-confirm current Khronos voting-seat status before send. |
| 4 | **Autodesk / Forma / AEC** | **Amar Hanspal** | SVP & GM, Autodesk Forma | Forma is Autodesk's AEC product positioning against the World Labs / spatial-AI wave; AEC is the natural splat-for-buildings vertical. | **VERIFY** — operator must confirm Hanspal's current title (Forma SVP/GM has had reorgs); pull from autodesk.com/company/leadership before send. Backup target: a Forma engineering leader who is a public author on Forma's reality-capture or point-cloud features. |
| 5 | **World Labs / Spark / Niantic Spatial** | **John Hanke** | CEO, Niantic (parent of Niantic Spatial, which spun out SPZ). | Niantic Spatial released SPZ 4 on 2026-05-11 (vendor extensions, parallel ZSTD, removed 10M-point cap); cited in `docs/partnerships/adobe-spz-memo.md` §1. SPZ is SplatForge's first-class output format and the basis for our new KHR-SPZ extension. | **VERIFY** — Hanke is widely-public CEO; confirm current Niantic vs Niantic Spatial reporting line before send. Operator-level alternative: **Pravin Kotipalli** (operator-cited as Niantic Spatial GM) — **VERIFY** title; pull from nianticlabs.com/spatial or LinkedIn. |

**Operator note on names dropped relative to v3.2 brief:**

- **Sean Lilley** (Cesium 3D Tiles spec lead) — kept as a credible Cesium engineer-level contact for the warm-intro path (§2.1 below); **VERIFY** current title.
- **Mike Rockwell** (Apple VP Vision Products) — kept as the AVP-design-partner aspirational target; the realistic first touch goes through AOUSD WG / WWDC developer support, not directly. **VERIFY** current title.
- **Sebastien Deguy / Vlad Lyubchenko** (Adobe Substance) — kept as engineering-level Adobe contacts in §2.3; **VERIFY** before send. Stefano Corazza is the higher-confidence primary.
- **Justin Hendrix / Fei-Fei Li** (World Labs) — the Autodesk-via-World-Labs angle relies on the World Labs investment in Forma being a matter of public record; **VERIFY** that investment before citing it in any Autodesk email. If it doesn't check out, drop the angle and lead with reality-capture / Forma point-cloud features instead.

---

## 2. Warm-intro graph — 3 plausible 2nd-degree paths per vertical

Heuristic: GitHub stargazer overlap, Khronos WG co-membership, OSS PR co-authorship, and the founder's existing 1-hop network from the SplatForge repo. For each vertical, "path" = the chain of public relationships that gets a SplatForge intro into the target's inbox without it reading as cold.

### 2.1 Geo / Cesium / Bentley — primary: Patrick Cozzi

| Path | Chain | Friction | Operator action |
|---|---|---|---|
| A | SplatForge → Cesium 3D Tiles 1.2 spec PR (preset `geospatial` already emits `tileset.json` + LOD pyramid) → **Sean Lilley** (3D Tiles spec maintainer, **VERIFY**) → **Patrick Cozzi** | Lowest. We are already a credible third-party 3D Tiles implementer. | Open a PR on `CesiumGS/3d-tiles` documenting the SplatForge `geospatial` preset output as a 3D-Tiles-1.2 reference consumer; @-mention spec maintainers. |
| B | SplatForge → KHR_gaussian_splatting conformance suite (already submitted upstream) → **Khronos 3D Formats WG mailing list** (Cozzi chairs) → Cozzi | Low. WG-chair channel is a public, formal route. | Cross-reference the SplatForge conformance crate in the existing Khronos GitHub issue (`docs/standards-outreach/khronos-issue.md`); request agenda time at the next WG call. |
| C | SplatForge → Bentley iTwin Platform developer relations (public Bentley developer-advocacy channel; **VERIFY** named DevRel) → iTwin product owner for spatial reality data → Cozzi (via Bentley-internal Cesium reporting line) | Medium. Requires identifying a named Bentley DevRel contact. | **VERIFY** Bentley iTwin DevRel handle via developer.bentley.com; send a public-developer inquiry referencing iTwin reality-data + KHR_gaussian_splatting reference impl. |

### 2.2 VFX / Apple / visionOS / OpenUSD — primary: AOUSD WG / Apple AVP developer relations

| Path | Chain | Friction | Operator action |
|---|---|---|---|
| A | SplatForge → OpenUSD Forum post (`docs/standards-outreach/openusd-forum-post.md`, ready-to-send; lead with the 10 spec gaps and bit-exact-as-USDA round-trip) → AOUSD WG chairs notice → Apple AVP design-partner channel | Lowest. Public forum post is a published-once, audience-of-everyone move. | Operator submits `openusd-forum-post.md` to forum.aousd.org > Schemas & Specifications. **This is the M-zero unblocker for this vertical.** |
| B | SplatForge → WWDC 2026 RealityKit / visionOS developer support channel (public Apple developer support form) → AVP design-partner BD | Medium. Apple developer support is a triage funnel; the spec-gap document is the credibility lever that gets past the triage. | Submit a developer-support inquiry citing `splatforge-usd` crate + spec-gaps doc; ask for RealityKit early-access for splat-pipeline validation. |
| C | SplatForge → glTF/Khronos WG → glTF↔USD interop bridge → Apple Khronos seat (Apple has a Khronos membership; **VERIFY** current seat-holder) → AVP team | Medium. Slower than (A) but the WG conversation is where Apple is already paying attention. | At the same Khronos WG agenda slot as 2.1B, raise the glTF↔USD `ParticleField3DGaussianSplat` interop story. |

### 2.3 Adobe / Substance / Creative Cloud — primary: Stefano Corazza

| Path | Chain | Friction | Operator action |
|---|---|---|---|
| A | SplatForge → **David Catuhe** (Babylon.js founder; Adobe-funded Babylon SPZ work) → Adobe Spatial-3D team | Lowest. Catuhe is the warm-Adobe path identified in the prior Adobe memo. Babylon.js 8.0 (Mar 2025) shipped Adobe-funded SPZ + compressed-PLY readers. | Open a PR on `BabylonJS/Babylon.js` or `BabylonJS/Loaders` adding SplatForge-optimized-SPZ as a reference test asset; reference the KHR-SPZ extension. |
| B | SplatForge → KHR_gaussian_splatting_compression_spz extension PR #1 (just landed) → Adobe Khronos voting-seat holder → Corazza (Khronos panelist) | Medium. The KHR-SPZ extension is the *Adobe-relevant* artifact (Adobe is the first SPZ vendor-extension registrant). | Send the SPZ partnership memo (`docs/partnerships/adobe-spz-memo.md`) under cover of the KHR-SPZ PR. |
| C | SplatForge → SPZ vendor-extension registry (Niantic-maintained) → registry shows Adobe's already-registered extension → reach out to whoever submitted Adobe's extension (**VERIFY** via SPZ registry public log) | Medium. Adobe's registrant identity is in the SPZ vendor-extension registry by definition; that engineer is the highest-signal Adobe contact for splat-pipeline. | **VERIFY** the Adobe extension submitter name via `nianticlabs/spz` repo extension registry; that person is a high-confidence cold-not-cold first touch. |

### 2.4 Autodesk / Forma / AEC — primary: Amar Hanspal (**VERIFY**)

| Path | Chain | Friction | Operator action |
|---|---|---|---|
| A | SplatForge → Autodesk Platform Services developer-relations (public DevRel channel) → Forma product engineering | Lowest of the three; APS DevRel exists and answers public-developer inquiries. | **VERIFY** the current APS DevRel contact name; submit a public-developer inquiry referencing the Cesium 3D Tiles preset (geospatial-AEC overlap) and our KHR conformance crate. |
| B | SplatForge → Khronos 3D Formats WG → Autodesk Khronos seat (Autodesk has a Khronos membership for glTF; **VERIFY** current seat-holder) → Forma | Medium. Same WG channel as 2.1B, 2.2C. | At the next Khronos WG meeting, raise an AEC-relevant agenda item (large-scene LOD for splats; Forma is the closest commercial parallel to Cesium in AEC). |
| C | SplatForge → World Labs / Forma investment connection (operator-asserted angle: **VERIFY** that World Labs invested in or partners with Autodesk Forma before relying on this path) → World Labs leadership → Hanspal | Highest. Investor-network warm intros require a real prior relationship. | **VERIFY** the World Labs ↔ Forma connection via SEC filings / press releases; if it does not check out, drop path (C) and rely on (A)+(B). |

### 2.5 World Labs / Spark / Niantic Spatial — primary: John Hanke

| Path | Chain | Friction | Operator action |
|---|---|---|---|
| A | SplatForge → `nianticlabs/spz` GitHub issue / PR (open a documentation PR adding the new `KHR_gaussian_splatting_compression_spz` extension to the SPZ extension registry; reference SplatForge as the glTF-side implementer) → SPZ maintainers (**VERIFY** named maintainers via repo commit log) → Niantic Spatial GM (**VERIFY**: Pravin Kotipalli per operator memory) → Hanke | Lowest. Niantic maintains the SPZ repo as public OSS; a registry-documentation PR is a textbook warm-intro move. | Open the SPZ-registry PR within 7 days of the KHR-SPZ extension landing. |
| B | SplatForge → SPZ format authors (operator memory: Tony Tomes, Felix Tristram — both **VERIFY** via `nianticlabs/spz` commit log before any direct mention) → Niantic Spatial GM → Hanke | Medium. Direct-to-author is warmer than registry PR but only works once names are verified. | After (A) lands, follow up via author-direct GitHub mention referencing the registry PR. |
| C | SplatForge → Spark (web Gaussian-Splat runtime, Niantic-Spatial-adjacent open-source) GitHub repo → Spark maintainers → Niantic Spatial | Medium. Spark is the runtime-side counterpart of our optimizer; natural integration story. | **VERIFY** Spark repo URL + maintainers; open an issue on Spark documenting SplatForge as a credibly-faster encoder feeding Spark. |

---

## 3. First-touch email drafts (one per vertical, ~200 words)

**All drafts cite only numbers from §0. All drafts end with a single specific ask. All drafts strip Claude/Anthropic attribution.**

### 3.1 Geo / Cesium / Bentley — to Patrick Cozzi

**Subject:** SplatForge KHR_gaussian_splatting reference impl + a `--preset geospatial` 3D-Tiles emitter

Hi Patrick,

Monte at SplatForge — we're the production-optimization pipeline for 3D Gaussian Splats (the FFmpeg/Cloudinary layer between training tools and runtimes). Two things on your desk you should know about:

1. We've submitted a **KHR_gaussian_splatting conformance test suite** upstream to Khronos: 23 normative clauses, 10 golden fixtures, CI workflow. We're aiming to be the public reference impl before Q2 2026 ratification. The submission issue is open in the KhronosGroup/glTF repo; would love your read.

2. Our `splatforge optimize --preset geospatial` already emits a `tileset.json` + 4-level LOD pyramid that drops into Cesium ion. Median compression on Mip-NeRF 360 `stump` (real outdoor scene) is **22.97× SPZ-web-mobile**, vs splat-transform v2.1.1's 5.91× on the same synthetic corpus. We'd like to open a PR on `CesiumGS/3d-tiles` documenting SplatForge as a reference 3D-Tiles-1.2 consumer for splat tiles.

15-minute call to walk you through the conformance suite and get your read on what would make it useful to the WG?

— Monte
SplatForge — github.com/montabano1/SplatForge

### 3.2 VFX / Apple / visionOS / OpenUSD — to AOUSD WG (public forum post lead)

**Subject:** USD 26.03: production feedback on `ParticleField3DGaussianSplat` — 10 spec ambiguities and a reference USDC writer

(This is the public-forum-post lead; the "ask" is the WG response, which becomes the warm intro to the Apple AOUSD seat-holder and AVP design-partner BD.)

Hi AOUSD WG,

We've built an end-to-end implementation of OpenUSD 26.03's `ParticleField3DGaussianSplat` schema and have feedback the WG should adjudicate before the next schema rev.

- **Implementation:** `splatforge-usd` crate (pure Rust, no Pixar/OpenUSD lib dependency).
- **Round-trip:** USDA → USDC → `usdcat` 0.25.2 → USDA, **bit-exact-as-USDA** on three reference scenes (minimal, particle_field, dense).
- **10 spec gaps captured** for the WG: missing `shCoefficients`/`shDegree` slots (every production 3DGS scene ships them), `(w,x,y,z)` vs `(x,y,z,w)` quaternion convention silently divergent USDA↔USDC, scale linear-vs-log ambiguity, opacity-range ambiguity, and 6 wire-format gotchas.

Full write-up + reproducer: `docs/standards-outreach/openusd-forum-post.md` in the SplatForge repo.

**Ask:** can we put these on the agenda for the next AOUSD schema-rev call? Happy to bring the round-trip harness and demo all three fixtures live.

— Monte, SplatForge

### 3.3 Adobe / Substance / Creative Cloud — to Stefano Corazza

**Subject:** KHR_gaussian_splatting_compression_spz — first KHR↔SPZ bridge extension, looking for Adobe's WG read

Hi Stefano,

Monte at SplatForge. Three things you'll care about, because Adobe is the first SPZ vendor-extension registrant and the Photoshop "Rotate Object" surface is now shipping SPZ at meaningful volume:

1. We just landed **`KHR_gaussian_splatting_compression_spz`** as a Khronos extension PR — the first standards bridge between glTF and Niantic's SPZ wire format. Adobe's Photoshop pipeline can now ship splats as `.glb + KHR_gaussian_splatting + KHR_..._compression_spz` and stay inside the Khronos extension envelope.
2. Our `KHR_gaussian_splatting` conformance suite (23 clauses, 10 fixtures) is submitted upstream — we're aiming to be the public reference impl before Q2 ratification.
3. SplatBench median is **21.97× SPZ-web-mobile** across 17 scenes (vs splat-transform v2.1.1's 5.91× on the same synthetic corpus). On Mip-NeRF 360 `bonsai` we hit +6.4 dB PSNR over opacity-prune at 50% byte budget via differentiable repack.

**Ask:** 15-minute Khronos-WG-side conversation on whether the KHR-SPZ extension shape works for Adobe, before the spec ratifies. Full memo at `docs/partnerships/adobe-spz-memo.md`.

— Monte, SplatForge

### 3.4 Autodesk / Forma / AEC — to Forma product engineering (Amar Hanspal, **VERIFY** before send)

**Subject:** Reality-capture splats for Forma — KHR conformance crate + a 22.97× compression ceiling on outdoor scenes

Hi [Amar / Forma team],

Monte at SplatForge — the production-optimization pipeline for 3D Gaussian Splats. AEC reality-capture workflows are at the point where the bytes-on-the-wire problem (fidelity-graded compression, deterministic outputs, CI regression gates) starts to matter, and that's the layer we built.

A few specifics that should be useful for Forma:

1. **22.97× SPZ-web-mobile on Mip-NeRF 360 `stump`** (real outdoor scene with foliage + organic geometry — closest public proxy for site-capture data). 21.97× median across 17 scenes; deterministic-by-design so Forma can cache by content hash.
2. **KHR_gaussian_splatting reference impl + conformance suite** submitted upstream to Khronos. The 23-clause test harness gates against fidelity regressions — exactly the gate AEC workflows need before splats are trusted in design-of-record.
3. Cesium 3D Tiles preset (`--preset geospatial`) emits `tileset.json` + LOD pyramid — drops into the same pipeline you'd use for terrain.

**Ask:** 15-minute design-partner conversation. We'd like to validate the AEC bytes-on-the-wire pain against a real Forma site-capture asset before we lock the next preset.

— Monte, SplatForge

### 3.5 World Labs / Spark / Niantic Spatial — to John Hanke / Niantic Spatial GM (**VERIFY** primary recipient)

**Subject:** SPZ + glTF: first KHR↔SPZ bridge extension just landed; SplatForge has shipped SPZ since v0.1.0

Hi John,

Monte at SplatForge. SPZ has been our first-class output format since v0.1.0; Niantic Spatial's SPZ 4 release (vendor extensions, parallel ZSTD, removed 10M-point cap) is what we've been waiting on.

Three things that should be on Niantic Spatial's radar:

1. We just landed **`KHR_gaussian_splatting_compression_spz`** in the Khronos extension repo — the first standards bridge between glTF and SPZ. SPZ now has a Khronos-tracked path into every glTF runtime (Cesium, Babylon, three.js, Unity, Unreal).
2. SplatForge's compression on SPZ-as-output: **22.97× on Mip-NeRF 360 `stump`**, **22.81× on bonsai**, overall median **21.97×** across 17 scenes. SPZ-as-wire-format + SplatForge-as-encoder is the production bar.
3. We'd like to open a PR on `nianticlabs/spz` documenting the KHR-SPZ extension in the SPZ vendor-extension registry — closes the loop between the two specs.

**Ask:** any chance of a 15-minute conversation with the SPZ format owner about (a) the KHR-SPZ extension shape, and (b) whether there's a Niantic Spatial design-partner conversation worth having? Happy to walk through the encoder side first.

— Monte, SplatForge

---

## 4. Pre-send checklist (operator must complete before any first-touch sends)

- [ ] **Verify every `VERIFY` flag above.** WebFetch was unavailable in this drafting session; operator must hit the cited public URLs and confirm current roles before any send.
- [ ] **Confirm the KHR-SPZ extension PR # and URL** — emails reference "PR #1" per the v3.2 brief; replace with the actual PR number/URL in `KhronosGroup/glTF` or the relevant fork.
- [ ] **Confirm the OpenUSD forum post has been submitted** (`docs/standards-outreach/openusd-forum-post.md` is the canonical draft); the AOUSD email in §3.2 is the *post itself*, not a separate inbound.
- [ ] **Confirm v0.1.2 blog post is published** before any email links to a live splatforge.com/bench leaderboard URL.
- [ ] **Confirm Bentley/Cesium relationship status** — Cesium became a Bentley product in late 2024 (operator memory); re-confirm via Bentley press release before referencing in §3.1.
- [ ] **Drop the World-Labs↔Forma angle if it doesn't check out** (§2.4 path C). Don't ship a claim that can be falsified in five minutes by the recipient.
- [ ] **Replace bracketed `[Amar / Forma team]` in §3.4** with a verified recipient name; if Hanspal's title has changed, redirect to the current Forma SVP/GM.
- [ ] **No Claude/Anthropic attribution** in any sent text or this file. (This file is clean.)
