# Catetus SDK — License Summary

**Resource URI:** `catetus://license/sdk-terms`
**SPDX:** `Apache-2.0`
**Issuer:** `api.catetus.com` (`apps/api/src/routes/sdk_license.rs`)
**Last updated:** 2026-05-27

---

## TL;DR

The Catetus open-source SDK (CLI, viewer, codec, MCP server) is licensed under **Apache-2.0**. You may use, modify, and redistribute it in commercial products with attribution. The hosted paid tier (encode, score_fidelity, repack, predict_quality, recommend_preset, batch_jobs) is gated by an API key and billed per usage; running these tools requires a valid `CATETUS_API_KEY`.

There is no "viral" license clause. No copyleft. Use the SDK in proprietary software without releasing your source.

---

## What is Apache-2.0?

A permissive open-source license that lets you:

- Use the software **commercially** without paying royalties.
- **Modify** the source code.
- **Distribute** original or modified versions.
- **Sublicense** under different terms in your derivative work.
- **Use in patent claims** — Apache-2.0 includes an explicit patent grant from contributors.

In exchange, you must:

- **Preserve the LICENSE file** and copyright notices in redistributed copies.
- **State changes** if you modify the source and distribute it.
- **Not use Catetus trademarks** (the name "Catetus", the logo) to endorse derivative products without permission.

Full text: https://www.apache.org/licenses/LICENSE-2.0

---

## Components and their licenses

| Component | License | Notes |
|---|---|---|
| `catetus` CLI (Rust binary) | Apache-2.0 | The local encoder. Bundle freely. |
| `@catetus/mcp` (this package) | Apache-2.0 | The MCP server. |
| `@catetus/viewer` (WebGPU viewer) | Apache-2.0 | The browser preview shell at splatforge.dev/viewer. |
| Encoder model weights (Catetus V5.2, T2.1.R) | Apache-2.0 | Shipped with the SDK; reusable. |
| `catetus.dev` website source | Apache-2.0 | |
| **SplatBench v0 corpus** | Mixed — per-scene; see `splatbench-v0.json` `license` field | Real scenes are under their upstream license (Mip-NeRF 360 is open research). Synthetic proxies are CC-BY-4.0. |
| **Hosted API** (`api.catetus.com`) | Service ToS | NOT open source. API-key-gated. See https://catetus.com/terms |
| Paid Modal A100 worker code | Closed (proprietary) | Not shipped with SDK. |

---

## Commercial use FAQ

### Can I ship Catetus in a paid product?

**Yes.** Apache-2.0 explicitly permits commercial use. No royalties owed.

### Can I rebrand the CLI or MCP server?

**Yes**, with two constraints:
1. You must preserve the `LICENSE` and `NOTICE` files in your distribution.
2. You may not call your fork "Catetus" or use Catetus branding/trademarks to imply endorsement.

### Can I sell encoding services using Catetus locally?

**Yes for the free tier.** The CLI runs the free presets (web-mobile, web-desktop, quality-max, etc.) entirely on your machine — no API key needed. You can wrap it in your own SaaS.

**No for the paid tier.** V5.2, T2.1.R, repack, score_fidelity, and predict_quality require `api.catetus.com` API calls and your `CATETUS_API_KEY`. You can't redistribute access to the paid tier; each end-user needs their own key (or you must hold a Catetus Enterprise reseller agreement — contact monte@catetus.com).

### Can I train a competing model on the SplatBench corpus?

**Yes for synthetic proxies** (`splatbench_*_proxy`, CC-BY-4.0 with attribution).
**Yes for Mip-NeRF 360 scenes** (`bonsai_mipnerf360_*`, `bicycle_mipnerf360_*`, `stump_mipnerf360_*`) under the Mip-NeRF 360 open-research license.
**Yes for the canonical-11 corpus** (Inria 3DGS pretrained models) under the upstream Inria research license — note that license restricts commercial reuse of the *trained models* but not of derived benchmarks/measurements.

When in doubt, check the per-scene `license` field in `catetus://bench/splatbench-v0`.

### Can I publish my own benchmarks using Catetus?

**Yes.** Independent benchmarks are encouraged. We ask that you cite the version (`catetus.version` field in any output) and the corpus you used (e.g. `splatbench-v0.1.1` or `canonical-11`). We'll happily link credible third-party benchmarks from `catetus.com/benchmarks`.

### Can I use the Catetus name in my product?

**Limited yes.** You may use "Catetus" in factual statements such as "Compresses with Catetus" or "Powered by Catetus." You may not use the Catetus logo, name, or marks in a way that implies endorsement, partnership, or co-branding without written permission. See https://catetus.com/brand for guidelines.

### Patent grant — what's covered?

Contributors to Catetus grant you a **perpetual, worldwide, royalty-free patent license** for any patent claims they own that read on their contributions. If you sue Catetus contributors (or downstream Catetus users) over those patent claims, you lose your patent license under Apache-2.0 §3.

---

## Attribution snippet

In your `THIRD_PARTY_LICENSES.md`, `NOTICE`, or product credits:

```
This product includes software developed by Catetus (https://catetus.com),
licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0
```

That's it. No corporate logo required, no co-marketing obligation.

---

## Contact

- License questions: monte@catetus.com
- Enterprise / reseller agreements (paid-tier sublicensing): monte@catetus.com
- Trademark / brand questions: monte@catetus.com
- Security issues: security@catetus.com

---

## See also

- `catetus://docs/preset-cheatsheet` — quick-pick preset reference.
- `catetus://bench/canonical-11` — published benchmark numbers (CC-BY-4.0).
- `catetus://corpus/competitor-codecs` — honest competitive audit.
- `catetus://bench/splatbench-v0` — corpus license details per scene.
