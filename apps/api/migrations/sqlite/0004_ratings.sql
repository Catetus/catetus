-- fidelity-ml v0.4 — human pairwise rating collection.
--
-- One row per rating submitted via splatforge.com/rate. The page shows
-- two rendered orbit frames from different presets (e.g. lossless-repack
-- vs web-mobile) side-by-side with the prompt "which of these looks
-- closer to the reference?" and POSTs the outcome here. Once we have
-- enough rows (target: 1000 minimum / 10k preferred per scene — see
-- docs/fidelity-ml-v0.4-collection.md) the rows feed a Bradley-Terry
-- aggregation step that converts pairwise outcomes into per-frame
-- absolute scalar scores. Those scores are the supervision signal for
-- fidelity-ml v0.4 — the MLP that finally breaks out of the synthetic-
-- corruption-label collapse the v0.3 weights inherited.
--
-- Privacy. We deliberately do NOT store IP, User-Agent, geo, or any
-- session cookie. The `respondent_hash` column is SHA-256 of
-- (IP || "|" || User-Agent) computed server-side per request and never
-- persisted in its plaintext form. The hash is one-way and salted by
-- the User-Agent free-text — collision is theoretically possible but
-- the result is at most "two visitors share a rate-limit bucket",
-- which is a quality issue, not a privacy issue. The hash exists only
-- so the rate limiter can detect floods without holding PII.
--
-- Rate limiting. The API enforces 100 ratings/hour per respondent_hash
-- via a GROUP BY + count query (see apps/api/src/main.rs::post_rating).
-- Past the cap the API returns 429 and the page surfaces a soft "thanks,
-- come back later" state instead of POSTing more.
CREATE TABLE IF NOT EXISTS ratings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Scene id from splatbench-v0.json (e.g. `bonsai_mipnerf360_iter7k`,
    -- `splatbench_texture_proxy`). Matched against the bench corpus when
    -- the v0.4 aggregator runs; unknown scene ids are dropped at that
    -- stage rather than rejected here, so a future bench rev with new
    -- scenes can keep posting through the same endpoint.
    scene_id        TEXT NOT NULL,
    -- The two presets being compared. Always recorded in the order the
    -- page rendered them (left vs right) so post-hoc analysis can detect
    -- side-bias (raters preferring whichever is on the left).
    left_preset     TEXT NOT NULL,
    right_preset    TEXT NOT NULL,
    -- "left" | "right" | "tie". "skip" is NOT a valid winner — the page
    -- offers a skip affordance which simply fetches the next pair
    -- without POSTing, so skips are not represented in the table. This
    -- is deliberate: a row in `ratings` means "a human compared these
    -- two and committed to a verdict", which is the contract the
    -- aggregator's Bradley-Terry model expects.
    winner          TEXT NOT NULL,
    respondent_hash TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ratings_scene       ON ratings(scene_id);
-- Used by the rate-limit query (count rows by hash + recent created_at).
CREATE INDEX IF NOT EXISTS idx_ratings_respondent  ON ratings(respondent_hash, created_at);
-- Used by /v1/ratings/summary for the per-pair aggregation.
CREATE INDEX IF NOT EXISTS idx_ratings_summary     ON ratings(scene_id, left_preset, right_preset);
