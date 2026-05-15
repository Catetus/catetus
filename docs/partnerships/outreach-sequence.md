# Adobe / SPZ Partnership — Outreach Sequence

**Confidential.** Order of operations for the Adobe partnership pitch. Companion to `adobe-spz-memo.md` and `contact-map.md`.

The principle: maximize external credibility *before* the Adobe-direct ask, so the working-session demo is the second time Adobe hears about us, not the first.

---

## Step 1 — Establish standing (T-0 to T+14 days)

**Goal:** Have at least one public, third-party-endorsed credential land before the Adobe outreach goes out.

**Concrete actions:**
- **Submit the KHR_gaussian_splatting conformance suite** to the Khronos 3D Formats WG via the draft issue in `docs/standards-outreach/khronos-issue.md`. The suite is 23 clauses, 10 golden fixtures, a validator binary, and CI workflow — it is ready. The submission needs the operator's signature on the GitHub issue.
- **Publish the v0.1.2 release blog** (`docs/blog/v0.1.2-release.md`) with the SplatBench comparison numbers front-and-center, including the 21.88× vs 5.91× splat-transform delta.
- **Confirm the KHR conformance suite is accepted upstream** as the official conformance vehicle for KHR_gaussian_splatting. This is the single biggest credibility lever for the Adobe meeting: we are not vendors pitching to Adobe, we are the team writing the standards Adobe will need to comply with.

**Why this matters first:** Adobe's BD/eng team will look us up the moment we reach out. The shape of what they find when they Google "splatforge" in week one materially changes the response rate. If `khronos.org` and the Khronos GitHub org are the first results, we are credible by default.

**Risk if skipped:** We come in as a six-person startup asking for the time of a senior Adobe leader. We come out with a polite "interesting, please follow up in Q4."

**Exit criteria:**
- Conformance submission filed in the Khronos GitHub.
- Public v0.1.2 announcement live with SplatBench numbers.
- Cesium / Patrick Cozzi acknowledgment of the conformance work — even a public LGTM comment is enough.

## Step 2 — Soft intro via Babylon.js (T+14 to T+30 days)

**Goal:** Get an Adobe-internal intro through the lowest-friction warm channel.

**Concrete actions:**
- **Reach out to David Catuhe** (Babylon.js lead) with the v0.1.2 release as the hook. Lead with the WebGPU compute decode (127 fps @ 1M splats) and the SPZ-compatible PostHAC stacking story. Frame: *"We've built the production optimizer side of what you've built the runtime side of; the next 12 months are about making the two sides compose well, and Adobe is the user that ties them together."*
- **In the same conversation, raise the Adobe partnership.** Ask Catuhe explicitly for an Adobe intro — by name where possible, by team where not. Be honest about the ask: "We want to ship behind Photoshop's 'Optimize for web' toggle. We know you don't make that decision, but you know who does." Babylon's team is Adobe-funded; they have weekly working contact with Adobe's spatial-3D eng team.
- **Backup channel: Patrick Cozzi.** If the Babylon path is slow, Cozzi as Khronos WG chair can route the conversation to Adobe's WG representatives (Stefano Corazza et al.) on the standards-rationale path: *"There's a conformance question that needs Adobe and SplatForge in the same room."*

**Risk:** Babylon team is polite but doesn't make the intro — they may not want to look like they're playing favorites. Mitigation: lead with the working-session demo as the ask, not "introduce me to your sponsor." The demo is a thing Babylon can attend too, which makes it less awkward to participate in.

**Exit criteria:** A named Adobe contact, a scheduled 30-minute discovery call, and an agreement on what corpus the demo will run against.

## Step 3 — Working-session demo + memo delivery (T+30 to T+60 days)

**Goal:** Replicate our SplatBench numbers on Adobe's own corpus, in the same room, with no smoke or mirrors.

**Concrete actions:**
- **90-minute working session.** Two SplatForge engineers + the operator. Two-to-four Adobe attendees (the named eng lead from Step 2 + Stefano Corazza or his eng delegate, plus optionally a CC product manager). Remote-acceptable; on-site at Adobe SF preferred for the production-impression value.
- **Live demo:**
  1. `splatforge optimize --preset web-mobile` on a SplatBench scene. Compare bytes-out vs `splat-transform`. Show the JSON fidelity report.
  2. **Hand the Adobe team a scene we have never seen** (this is the part that matters). Have them upload it to a private SplatForge endpoint. Run web-mobile. Run web-mobile + PostHAC. Show the 4× and the 8× respectively, on *their* bytes.
  3. **Show the export-as-SPZ-4 path** with the Adobe vendor-extension chunk preserved bit-exactly through our pipeline. This is the trust-establishment moment.
  4. Differentiable Repack: rate-distortion curve, +6.4 dB at 50% byte budget, $0.05–$0.12/scene. Make explicit that this is the licensable layer (Shape B in the memo).
- **Deliver the memo at the end of the session.** Not before. The memo references the numbers from the demo, so handing it over after the demo turns it from "marketing collateral" into "minutes of our meeting."

**Demo risks the operator must be ready to address:**

1. **Latency on Adobe's cloud.** Adobe will ask: *"If we're ripping splats out of Photoshop and your encoder lives on Fly, what does that add to the export time?"* Be ready with concrete numbers: median 1.99 MB → submitted job in <2s end-to-end (live-measured); web-mobile encode on a 1M-splat scene takes 4-7s on a single Fly-Sydney edge worker. If Adobe needs sub-second, the embedded Rust library option (Shape A's in-process variant) is the answer — and we should bring a measured local-encode benchmark for that path too.
2. **Security review for Stripe-billed paid passes.** Adobe will ask: *"Your hosted API is Stripe-metered today. How does that work inside Adobe's billing system?"* Be ready to explain that the licensing shape for Adobe is *not* end-user Stripe — it's a flat OEM license (Shape A) or a wholesale per-job rate aggregated into Adobe's CC billing (Shape B). The Stripe billing-meter scaffolding stays for SplatForge's direct customers; Adobe gets a separate billing surface that does not require Adobe users to ever see a SplatForge invoice.

**Exit criteria:** Adobe agrees in principle to an evaluation license, and we leave the room with a list of integration unknowns to resolve before contract drafting (security review, embedding shape, latency target, billing structure).

## Step 4 — Evaluation license + 90-day pilot (T+60 to T+150 days)

**Goal:** Put SplatForge in Adobe's internal hands and produce a numbers-on-Adobe-scenes report.

**Concrete actions:**
- **Mutual NDA + evaluation license.** Scope: Adobe Spatial-3D / Substance 3D internal use only. No production-end-user exposure. 90 days, renewable to 180. Zero fee.
- **Pilot deliverable:** a joint report (SplatForge + Adobe eng) measuring, on Adobe's corpus, the compression / fidelity / latency numbers for the public preset and the Pro preset. This report is what justifies the Term Sheet.
- **Joint planning** on the Khronos coordination: a co-signed comment on the KHR_gaussian_splatting issue noting Adobe and SplatForge's coordinated conformance position. Public, low-cost, high-signal.
- **Begin SOC 2 Type I prep** in parallel. Pilot success without a credible compliance plan is a contract-stage gotcha; head it off here.

**Exit criteria:**
- Joint pilot report drafted (even if internal-only) with numbers.
- Adobe's eng team has used the SplatForge library / API enough to have a credible internal opinion of it.
- Soc 2 Type I audit kicked off with a vendor.

## Step 5 — Decision point: formal partnership, or fall back gracefully (T+150 days)

**Two outcomes; both are fine.**

**Outcome A — Term Sheet.** Adobe wants Shape A and/or Shape B. We move to contract drafting. The contract should preserve:
- Our right to continue shipping the same algorithms outside the Adobe surface (no exclusivity).
- Adobe's right to migrate off SplatForge at the SPZ / KHR_gaussian_splatting interface (no lock-in). The bytes-out are standard; the only thing they're licensing is the encoder, and an open exit is the right thing to offer.
- An MFN clause on any subsequent CC-tier OEM integration we sign with a direct Adobe competitor. Two-way.

**Outcome B — "Adobe is an upstream library consumer, not a partner."** Adobe declines the OEM integration but continues to use `splatforge-spz` as an open-source library, contributes to the KHR conformance suite, and we maintain a working-level engineering relationship without commercial entanglement. This is **not a failure outcome**; it preserves the standards work, it leaves the door open for a re-pitch in 12-18 months when Adobe's internal splat-export volume forces the optimizer question again, and it doesn't burn the relationship. **We should leave the meeting that produces this outcome with a written "Adobe will consider re-engagement when X" condition.** Typical X: "When SplatForge ships SOC 2 Type II," or "When Adobe's splat-export monthly volume crosses 10M files."

---

## Single-page summary

| Step | Window | Hard prerequisite | Action |
|---|---|---|---|
| 1 | T-0 to T+14d | None | Submit KHR conformance to Khronos, publish v0.1.2 blog |
| 2 | T+14 to T+30d | Step 1 visibility | Reach Catuhe (Babylon) for Adobe warm intro |
| 3 | T+30 to T+60d | Step 2 intro confirmed | 90-min working-session demo + memo handoff |
| 4 | T+60 to T+150d | Step 3 produces evaluation-license agreement | 90-day pilot, mutual NDA, joint internal report |
| 5 | T+150d | Step 4 pilot report exists | Term Sheet OR documented "re-engage when X" fallback |

If we are not at Step 3 by July 2026, the sequence has stalled and the operator should re-plan. If we are at Step 5 by November 2026, this has gone well.
