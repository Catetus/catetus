# fidelity-ml v0.4 — human-rating collection

## Why v0.4

`fidelity-ml v0.3` (shipped in `crates/splatforge-fidelity-ml/data/weights-v0.3.json`)
is an MLP trained on a **synthetic corruption corpus**: 1,176 pairs of
rendered orbit frames where one half had a known artificial perturbation
(JPEG, blur, additive Gaussian noise, splat-pruning, etc.) applied to it
and the "ground truth" label was assigned by a deterministic per-corruption
schedule. That corpus is reproducible and cheap, and it gets v0.3 to a
useful working point — but it inherits two collapse modes:

1. **Label-generator collapse.** The model can only learn the artifact
   classes its label generator already understands. Real Gaussian-Splat
   encoders (web-mobile / size-min / SOG / SOGS / CodecGS-Lite) produce
   distributions of artifacts that the synthetic schedule never sees —
   particularly the joint distributions of *opacity prune + spatial
   reorder + quantization*. The v0.3 model assigns confident-but-wrong
   scores to those.
2. **Distribution mismatch.** Synthetic corruption is i.i.d. per frame;
   real encoder artifacts are correlated across orbit positions because
   the encoder's input is the underlying splat, not the rendered frame.
   v0.3 can't see that correlation.

v0.4 fixes both by replacing the synthetic labels with **real human
pairwise comparisons** of actual encoder outputs. The supervision signal
becomes "which encoder produced the closer-to-reference render?" — which
is exactly the question the leaderboard column is supposed to answer.

## The collection apparatus

The public page lives at **`splatforge.dev/rate`** and is wired to the
API endpoint `POST /v1/ratings`. The rater sees:

```
[ Reference ]   [ Candidate A ]   [ Candidate B ]
   ↑                 ↑                  ↑
lossless-repack    encoder X         encoder Y
   (same orbit position, same scene)
```

with the prompt

> **Which of these looks closer to the reference?**

and three buttons (`A looks closer` / `B looks closer` / `tie`) plus a
fourth "skip this pair — ambiguous or rendered badly" affordance that
fetches the next pair *without* writing a row. Skipped pairs are dropped
from the corpus, not coerced into ties. (See the *Risks* section below
for why this matters.)

### Schema (`apps/api/migrations/0003_ratings.sql`)

```sql
CREATE TABLE ratings (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  scene_id        TEXT NOT NULL,   -- from splatbench-v0.json
  left_preset     TEXT NOT NULL,   -- recorded in display order
  right_preset    TEXT NOT NULL,   --   (so we can detect side-bias)
  winner          TEXT NOT NULL,   -- "left" | "right" | "tie"
  respondent_hash TEXT NOT NULL,   -- SHA-256(IP || "|" || UA)
  created_at      TEXT NOT NULL    -- RFC-3339, server clock
);
```

No PII is stored. The respondent hash is one-way and exists only to
enforce the **100 ratings/hour cap** per browser. Plaintext IP and
User-Agent are never written to the database, log file, or any other
persisted store.

### API endpoints

* `POST /v1/ratings` — `{ scene_id, left_preset, right_preset, winner }`,
  returns `{ id, remaining }`. Rate-limit: 100/hour per respondent_hash;
  past the cap returns `429`.
* `GET /v1/ratings/summary` — `{ pairs: [...], total_ratings: N }` where
  each `pair` is `{ scene_id, left_preset, right_preset, left_wins,
  right_wins, ties, total }`. This is the input the v0.4 trainer pulls.

## Bradley-Terry aggregation

Pairwise comparisons on their own are not a supervised regression
target — they're sparse, noisy, and (per scene × preset) only constrain
a *partial order*, not absolute scores. The v0.4 training pipeline runs
a **Bradley-Terry** maximum-likelihood fit over the rating table to
produce per-(scene × preset × frame) absolute scalar fidelity scores
that *are* a clean MLP regression target.

Bradley-Terry models the probability of "preset A beats preset B" on a
given scene as:

```
P(A > B) = exp(s_A) / (exp(s_A) + exp(s_B))
```

where `s_A`, `s_B` are the latent scalar fidelities to be fit. Ties are
incorporated as the Davidson extension (`exp(s_A + s_B) / 2` weighted
component). The fit minimizes negative log-likelihood over all observed
pair outcomes; closed-form is impossible past ~3 presets, so we use
20-iteration L-BFGS — converges in <1 s on 10k ratings × 16 scenes.

The output of the fit is a tensor of shape `(n_scenes, n_presets,
n_frames)` of normalized scalar scores. Those scores are the supervision
signal for the v0.4 MLP, which is trained to predict the same scalar
from the rendered frame's image features (the same input head as v0.3).

## Sample-size requirements

The v0.4 training kicks off when we hit:

| Threshold | Per-scene ratings | Property |
| --------- | ----------------- | -------- |
| **Bootstrap** | 100 | Sanity check; BT fit converges but per-pair confidence intervals are wide. Useful for debug, not for training. |
| **Minimum**   | **1,000** | BT fit becomes statistically distinguishable from random. v0.4 training **starts here.** |
| **Preferred** | **10,000** | Per-frame scalar confidence intervals < 5% of the score range. v0.4 production weights ship from this corpus. |
| **Stretch**   | 100,000 | Stratified by rater demographic (when we have it); enables per-cohort sub-models. |

With 16 scenes in the public bench corpus, that means the v0.4 training
gate is **16,000 total ratings minimum / 160,000 preferred**. At a
conservative 20 seconds per rating that's ~89 person-hours minimum,
~890 preferred — well within reach with even a small influx of public
visitors. The page's 100 rating/hour cap is a flood gate, not a
collection ceiling.

## Privacy

What lands in the database, per row:

* `scene_id`, `left_preset`, `right_preset`, `winner` — public corpus
  identifiers and the rater's verdict. Free to publish.
* `respondent_hash` — **SHA-256 of `<IP> || "|" || <User-Agent>`**.
  Computed server-side. The plaintext is never written anywhere. Hash
  is non-reversible: we can detect "this is the 99th rating from the
  same browser" but cannot recover *which* browser.
* `created_at` — server clock.

The hash exists only to enforce the 100/hour rate limit. A visitor
changing User-Agent or rotating IPs effectively resets their bucket,
which is intentional — the cap is anti-flood, not anti-Sybil. We are
not trying to enforce true rater uniqueness (which would require auth,
which would shrink the rater pool by an order of magnitude). The
Bradley-Terry fit downstream is robust to per-respondent imbalance
because the latent variable is per *preset*, not per *rater*.

## Risks in data quality

The single biggest risk is **selection bias from technical visitors**.
SplatForge readers know what compression artifacts look like — they're
the people who built the encoders being compared. They will (a) notice
subtler differences than the general population, and (b) have priors
about which encoder is supposed to win. Both push the resulting scores
*away* from what a generalist user (a CesiumJS app's end user, a
viewer-app product designer) would actually rate.

Mitigations baked into the design:

1. **Side ordering is recorded.** The schema preserves `left_preset`
   vs `right_preset` in display order so the analysis stage can detect
   and correct for "raters always pick the left one" / "raters always
   pick the more recognizable artifact direction".
2. **Tie is a first-class outcome.** A rater who genuinely can't tell
   should not be forced into a winner — that injects noise. The "tie /
   can't tell" button is the same visual weight as the directional
   votes.
3. **Skip is wired.** Ambiguous or rendered-badly pairs are dropped,
   not coerced. A skipped pair leaves *no* row in the table. This
   prevents the trainer from learning artifacts of the bench renderer
   (e.g. a single bad orbit frame) as if they were encoder differences.
4. **Future:** stratify by rater demographic once we have a way to ask
   (post-hoc questionnaire, separate cohort tags). Out of scope for the
   bootstrap minimum.

The risk *cannot* be fully eliminated at the collection layer — only
the analysis layer can. The schema preserves enough metadata
(`respondent_hash`, side ordering, timestamps) that a post-hoc
de-biasing pass over the table is straightforward.

## Operational notes

* The page is statically served from `apps/web/src/pages/rate.astro` and
  POSTs to the same-origin API proxy `/api/v1/ratings` → backend at
  `splatforge-api.fly.dev`.
* The bench frame PNGs ship in `apps/web/public/rate-frames/` at build
  time via `scripts/sync-data.mjs`. Source of truth is
  `benches/reports/frames/<scene>/<preset>/<frame>.png`. Both the
  source and the destination are gitignored — frames are bench artefacts
  that re-render from scene + preset + camera, not committed assets.
* Rate-limit is enforced via `respondent_hash`-scoped SQLite count, not
  via memory state. Restarting the API process does not reset anyone's
  bucket; the cap is durable.
* The Bradley-Terry trainer lives in the private repo
  (`splatforge-pro/training/fidelity-ml-v0.4/`) and pulls `/v1/ratings/summary`
  on a cron. Public repo does not host the trainer because the v0.4 weights
  are part of `splatforge-pro`, the same as v0.3.
