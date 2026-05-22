# Lessons learned

## 2026-05-19 — Verify the experiment ledger before accepting a task brief's hypothesis

**Mistake pattern:** A task brief proposed "the 0.33 dB Rust V5.2 gap is
the IR→PLY round-trip in the residual baseline — switch to in-memory
recon and Phase D will close it." Going straight to implementation
would have regressed V5.2 by ~5 dB, because the in-memory subtract was
already tried in Phase C v1/v2/v3 (53.79 dB) and the +4.89 dB v3→v4
fix was the *opposite* direction.

**Rule:** When a task brief proposes a specific fix path, before
touching any code:

1. Read the existing `experiments/<feature>/RESULT.md` end-to-end —
   prior iterations are usually documented and may already refute the
   proposed fix.
2. If iterations are on the 4090, list the bench dirs (`ls -t
   ~/catetus/<feature>-bench/`) and `cat` the bench JSONs. The
   iteration ledger is sitting on disk.
3. Cross-check the brief's quoted reasoning against the actual line it
   cites. Brief said "more honest, slightly less headroom" was the
   reason to switch — that phrase actually describes the *correct*
   v4 path; brief author conflated "honest" with "wrong".
4. Only commit to the proposed fix if the existing ledger supports
   it. Otherwise stop, write a finding note, and surface the
   contradiction.

**Reference:** `experiments/v5-2-phase-d/RESULT.md` (refuted hypothesis,
honest fix paths documented but not implemented).

## 2026-05-20 — Find/replace passes must skip back-compat tables

**Mistake pattern:** Bulk `SplatForge → Catetus` / `SF_ → CT_` rewrite
during the public-flip scrub turned the LEFT-side keys of the
`LEGACY_EXTENSION_REMAP` table in `packages/glb-polyfill/src/index.ts`
into `CT_*` — i.e. self-mappings. The normalization loop then
`delete`d each `CT_*` key after copying its value onto itself, so a
fresh `CT_*`-only GLB came out of the polyfill with zero extensions.
Tests passed locally because the test fixtures still had the old `SF_*`
extension names at the time the regex ran.

**Rule:** Before any project-wide find/replace pass, grep for the
OLD identifier in any file that contains the word `legacy`, `compat`,
`alias`, `remap`, `deprecated`, `migration`, or `back-compat`. Every
hit in those files is a candidate for being *intentionally* the old
name — review by hand and exclude from the bulk pass.

**Belt-and-braces:** after a rename pass, run the test suite against a
fresh fixture file that uses ONLY the new names. The old-name fixtures
will keep passing because the legacy path falls through to the
identity case.

**Reference:** caught in the post-orphan-branch Cursor audit, fixed by
reverting just the left-side keys of `LEGACY_EXTENSION_REMAP` to `SF_*`.

## 2026-05-20 — Vercel CDN auto-Brotli strips Content-Length

**Mistake pattern:** Vendored antimatter15 viewer sized its splat
buffer with `new Uint8Array(req.headers.get("content-length"))`. Works
on origin; on Vercel the CDN auto-Brotli-compresses the `.splat`
response and strips Content-Length (because the on-the-wire length no
longer matches the decompressed length). `headers.get(...)` returns
`null`, `Number(null)` is 0, `new Uint8Array(0)` is a 0-byte buffer,
no splats parse, hero canvas is black. The `no-transform` cache
header in `vercel.json` is ignored by Vercel's CDN.

**Rule:** Any progressive-fetch path that allocates a buffer from
Content-Length needs a fallback for `null` / `0`. Pre-allocate a
reasonable upper bound (200 MB for hero scenes) and track bytes-
written separately. Don't trust Content-Length on responses that go
through a CDN that can transcode.

**Diagnostic shortcut:** if a `fetch()` returns 200 + a body that
parses to zero records, check `response.headers.get('content-length')`
first — if it's `null` and the response is `Content-Encoding: br` or
`gzip`, that's the bug.

**Reference:** `apps/web/public/am15/main.js` ~L763 (fix landed
2026-05-20 in the hero-restore chain).
