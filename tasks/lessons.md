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
