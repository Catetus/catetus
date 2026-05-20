# Contributing — repo hygiene & safety hooks

This page covers the local protections every contributor (read: every
maintainer with push access) is expected to install. For the broader
contribution guide, see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Pre-push hook: partnership / outreach doc guard

After cloning, run:

```bash
./scripts/install-githooks.sh
```

This points `core.hooksPath` at the repo-tracked `.githooks/` directory and
installs a `pre-push` hook that **refuses any push containing changes under**:

- `docs/partnerships/**`
- `docs/standards-outreach/**`

### Why

On 2026-05-15 `docs/partnerships/{adobe-spz-memo,contact-map,outreach-sequence}.md`
plus two `docs/standards-outreach/` drafts were pushed to the public
`test-hero-fast` branch and sat there for ~4 days. The Khronos maintainer
closed issue #2580 citing the leaked `outreach-sequence.md` as evidence the
submission was premature. The full audit lives at
[`experiments/partnership-docs-audit/AUDIT.md`](../experiments/partnership-docs-audit/AUDIT.md).

The repo is currently private but will be flipped public again. These docs
are **never** intended to be public. Manual discipline already failed once;
this hook is the automated guardrail.

### What it does

1. Reads the list of ref updates from git's pre-push protocol.
2. For each update, computes the diff range:
   - **Updating an existing branch** → `<remote_sha>..<local_sha>`
   - **First push of a new branch** → `origin/HEAD..local_sha`
     (falls back to `origin/main`, then `origin/master`, then the empty tree)
   - **Branch deletion** → skipped (no content to leak)
3. Lists added/modified paths and matches them against
   `^docs/(partnerships|standards-outreach)/`.
4. If any match, exits non-zero with a clear message and the offending paths.

The hook is fast (typically < 200 ms — one `git diff --name-only` per ref)
and **fails open** if the diff machinery errors out. CI (see
`.github/workflows/partnership-docs-check.yml`) is the belt-and-suspenders
backstop that catches anything the local hook misses.

### How to override (legitimate cases only)

Sometimes you genuinely need to push these paths — most commonly, when
deleting a leaked file from history with `git filter-repo` and force-pushing
the cleaned ref, or when pushing to an explicitly-private mirror.

```bash
CATETUS_PARTNERSHIP_OVERRIDE=1 git push origin <branch>
```

When the override fires, the hook writes an audit line to stderr capturing:

- `git config user.email`
- UTC timestamp
- remote being pushed to
- the exact protected paths that were overridden

The override is **per-invocation** — it does not persist. Do not export it
in your shell profile.

### Disabling the hook locally

If you need to temporarily disable all repo hooks (e.g. you're scripting a
batch rewrite), unset the config:

```bash
git config --unset core.hooksPath
```

Re-run `./scripts/install-githooks.sh` to restore.

### CI backstop

`.github/workflows/partnership-docs-check.yml` runs on every `push` and
`pull_request`. It fails if the diff against the base ref contains any
added/modified file under the two protected directories. There is no
override path in CI — if you genuinely need to land such a change (e.g.
intentionally publishing a redacted version), the workflow file itself
must be edited in the same PR and reviewed.
