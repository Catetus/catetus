# Adobe / SPZ Partnership — Contact Map

**Confidential.** Research notes for the Adobe outreach sequence. Names listed are public maintainers of public projects, public Khronos WG members, or public author bylines on Adobe-funded OSS. No private contact information is recorded here.

---

## A. Adobe Spatial-3D / Substance 3D — primary targets

These are the people most likely to own the "Export 3D Gaussian Splat" path inside Photoshop and Substance 3D.

| Name / role | Why they matter | Source | Status |
|---|---|---|---|
| **Stefano Corazza** — Senior Principal Scientist, Adobe | Public-facing Adobe voice on 3D / glTF / Khronos. Has spoken on glTF working-group panels representing Adobe. Likely chairs or sits adjacent to the team that owns Photoshop's spatial-3D pipeline. | Khronos panel appearances; LinkedIn-public title | **Confirmed public ID. Best single point of entry for the standards conversation.** |
| **Michael Bond** — Engineer at Adobe (Babylon.js contributor) | Public contributor to Babylon.js 8.0 environment-lighting improvements; Adobe's engineering presence in the Babylon-Adobe partnership lane. | Babylon.js 8.0 release notes (March 2025) | Confirmed by-line; LinkedIn TODO to confirm current team. |
| **Photoshop "Rotate Object" eng lead** | Owns the user-facing surface that just shipped 800K SPZ files in 60 days. The person whose product decision determines whether "Optimize for web" toggles SplatForge on. | Photoshop release announcement | **TODO confirm name** — likely findable via the Photoshop release blog or the Adobe Max 2025 session presenter list. |
| **Substance 3D team lead** | The other CC surface where splat export will land. Substance 3D Sampler already supports scan-to-splat workflows. | Adobe Substance 3D product page | **TODO confirm name.** |

## B. Babylon.js maintainer team — the soft-intro lane

Babylon.js is Adobe-funded for the Gaussian-Splat work. The maintainer team is the lowest-friction route to a warm Adobe intro, because (a) they already work with Adobe weekly, and (b) we have meaningful technical overlap to lead with.

| Name / role | Why they matter | Source | Status |
|---|---|---|---|
| **David Catuhe** — Microsoft, Babylon.js founder / lead maintainer | Founding voice on Babylon.js. Drove the SPZ + compressed-PLY integration in Babylon.js 8.0. Sits on the Khronos 3D Formats WG. | Babylon Medium post on Khronos collaboration | **Confirmed.** Best single Babylon-side contact. |
| **Sebastien Vandenberghe** — Microsoft, Babylon.js team | Co-led Babylon's glTF prototyping; long-standing Khronos contributor. | Babylon Medium post on Khronos collaboration | **Confirmed.** |
| **Babylon.js Gaussian-Splat feature lead** | The engineer who actually implemented the SPZ reader and the static-texture optimization. They will care most about the SplatForge optimizer story because they own the perf budget. | Babylon.js forum announcements; GitHub PR history on `BabylonJS/Babylon.js` | **TODO confirm name** — pull from the SPZ-related PR author list. |

**Note on the operator's note** ("Patrick Cosson"): the closest public match is **Patrick Cozzi**, CEO/founder of Cesium and chair of the Khronos 3D Formats WG. Cozzi is *not* an Adobe contact, but he is the WG chair and a likely warm intro to Adobe via the WG itself. If the operator was thinking of someone at Adobe specifically and not Cozzi, that is a TODO — possibly a confusion of names worth resolving before outreach.

## C. Niantic — SPZ format owners

| Name / role | Why they matter | Source | Status |
|---|---|---|---|
| **SPZ format maintainers** (operator references: Tony Tomes, Felix Tristram) | Own the SPZ wire-format spec and the vendor-extension registry where Adobe just registered the first extension. Friendly party to either Adobe or us; not a partnership target per se, but a relationship to keep warm because they decide which extensions get standardized. | Niantic SPZ 4 release blog (May 2026); the named individuals were not surfaced by public-web search and remain **TODO confirm** — likely listed in the `nianticlabs/spz` GitHub commit history. | Names from operator memory only; verify before outreach. |

## D. Khronos 3D Formats Working Group — the standards lane

If Babylon doesn't land us a fast intro, Khronos is the backup. Adobe holds a voting seat; we co-author the conformance suite for the very extension Adobe is shipping in Photoshop.

| Name / role | Org | Why they matter |
|---|---|---|
| **Patrick Cozzi** — chair, 3D Formats WG | Cesium | Sets the agenda for KHR_gaussian_splatting acceptance. Friendly to SplatForge already (we have a Cesium 3D Tiles export integration; preset `geospatial` lands in their format). |
| **Stefano Corazza** | Adobe | (See section A. Both an Adobe contact and a Khronos contact — the same person bridges the two lanes.) |
| **WG members from `@adobe.com` on the `KhronosGroup/glTF` GitHub repo** | Adobe | **TODO** — pull commit-author list from `KhronosGroup/glTF` repo, filter `@adobe.com` email domain, cross-reference recent PR activity on the KHR_gaussian_splatting extension thread. This is the engineer-level Adobe contact list we want for the working-session demo. |

## E. Operator's existing relationships — to be filled in

The operator should annotate this section with which of the above intros are already 1-hop or 2-hop:

- **Stefano Corazza (Adobe)** — operator relationship: **TODO**.
- **David Catuhe (Babylon)** — operator relationship: **TODO**.
- **Patrick Cozzi (Cesium/Khronos)** — likely 1-hop via the SplatForge Cesium 3D Tiles export work and the KHR conformance submission. **TODO confirm.**
- **Niantic SPZ team** — operator relationship: **TODO**.

The cleanest sequence given the public-facing data is:
1. **Cozzi (Khronos)** — warmest, lowest-friction. Hands us a credible third-party endorsement.
2. **Catuhe (Babylon)** — high-trust Adobe-adjacent. Likely produces the Adobe intro.
3. **Corazza (Adobe)** — closes the loop directly inside Adobe.

This corresponds to the outreach sequence in `outreach-sequence.md`.
