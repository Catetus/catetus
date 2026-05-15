# SplatForge release runbook

End-to-end checklist for cutting a CLI release that ships through all three
distribution channels: GitHub Releases, Homebrew tap, npm registry.

Audience: the human operator. Every step below is something a machine
*could* do, but we keep it manual until we have signed-tag CI that
publishes to npm + the tap automatically.

---

## Pre-flight (once)

- [ ] Public repo `splatforge/splatforge` exists.
- [ ] Public tap repo exists: `gh repo create splatforge/homebrew-tap --public`.
      Seed it with a `README.md` and a `Formula/` directory.
- [ ] npm scope owned: `npm org create splatforge` and `npm login` as a
      member of the org.
- [ ] Two-factor auth on the npm account.
- [ ] (Optional) GitHub Secrets for code-signing:
        - `APPLE_P12_BASE64`, `APPLE_P12_PASSWORD`, `APPLE_IDENTITY`
        - `WIN_PFX_BASE64`, `WIN_PFX_PASSWORD`
      Without these, builds work but binaries are unsigned. Don't ship
      to mass-market until macOS notarization is set up — Gatekeeper
      will reject unsigned binaries downloaded via Homebrew.

## Pre-flight (every release)

- [ ] All open PRs targeting the release are merged.
- [ ] `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace --all-targets`
- [ ] Bump `version` in root `Cargo.toml`'s `[workspace.package]`. Pre-releases
      use `-rc.N`, e.g. `0.2.0-rc.1`.
- [ ] Update `CHANGELOG.md`.
- [ ] Bump `distrib/npm/package.json#version` to **match exactly** — the
      postinstall constructs the URL as `v${pkg.version}/splatforge-v${pkg.version}-…`,
      so a mismatch is a 404 at `npm install` time.
- [ ] `git commit -am "chore(release): vX.Y.Z"`.

## Cut the release

1. **Tag and push.**

    ```bash
    git tag -a vX.Y.Z -m "vX.Y.Z"
    git push origin main vX.Y.Z
    ```

    Pushing the tag triggers `.github/workflows/release.yml`. The
    five-target build matrix runs in parallel (~6-10 min wall time),
    then publishes a GitHub Release with:

      - `splatforge-vX.Y.Z-<target>.tar.gz` × 4 (unix targets)
      - `splatforge-vX.Y.Z-x86_64-pc-windows-msvc.zip`
      - `*.sha256` siblings
      - `SHASUMS256.txt` aggregated manifest

2. **Watch:** `gh run watch`. If a single target fails, re-run that job
   via the GH UI rather than retagging.

3. **Verify:**

    ```bash
    gh release view vX.Y.Z --repo splatforge/splatforge
    ```

## Update the Homebrew tap

Single largest distribution-risk: stale SHAs cause `brew install` to fail
with `SHA256 mismatch`. **Run the helper script — don't edit by hand.**

```bash
./scripts/release/update-homebrew-formula.sh vX.Y.Z
git diff distrib/homebrew/splatforge.rb
git add distrib/homebrew/splatforge.rb
git commit -m "release: homebrew formula vX.Y.Z"
git push origin main
```

Copy into the tap repo:

```bash
cp distrib/homebrew/splatforge.rb \
   /path/to/homebrew-tap/Formula/splatforge.rb
( cd /path/to/homebrew-tap && \
  git add Formula/splatforge.rb && \
  git commit -m "splatforge vX.Y.Z" && \
  git push origin main )
```

Verify end-to-end:

```bash
brew untap splatforge/tap || true
brew tap splatforge/tap
brew install splatforge
splatforge --version
```

## Publish to npm

```bash
cd distrib/npm
npm publish --access public
```

`--access public` is mandatory the first time you publish a scoped package.

Verify:

```bash
npm install -g @splatforge/cli
splatforge --version
# Or, without polluting the global cache:
npx --yes @splatforge/cli@latest --version
```

## Smoke testing (before tagging)

The release workflow has two non-release triggers:

- **PR with `[smoke-release]` in the commit message** — full matrix
  build, artifacts uploaded to the PR, publish step skipped.
- **`workflow_dispatch`** — manual trigger from the Actions tab.

Use these whenever you change:

  - `.github/workflows/release.yml`
  - `distrib/npm/install.js`
  - the workspace's `[profile.release]`
  - any crate's `[[bin]]` table

## Rollback

1. **GitHub release** — `gh release delete vX.Y.Z --yes` (keeps the tag),
   or `gh release edit vX.Y.Z --draft` to hide while you investigate.
2. **npm** — within 72 h: `npm unpublish @splatforge/cli@X.Y.Z`. After
   72 h: ship X.Y.Z+1 and `npm deprecate @splatforge/cli@X.Y.Z "use X.Y.Z+1"`.
3. **Homebrew** — revert the commit on the tap repo. Existing installs
   are unaffected; `brew upgrade` users pick up the revert.

## Future automation

The eventual goal is to make all three publishes automatic from a tag push:

  - Step 1 (GH release): already automatic.
  - Step 2 (Homebrew): a follow-up job on `release.yml` that runs
    `scripts/release/update-homebrew-formula.sh`, opens a PR on the tap
    repo via `gh pr create --repo splatforge/homebrew-tap`. Needs a PAT
    with write access to the tap.
  - Step 3 (npm): a follow-up job that does `npm publish --access public`
    with `NPM_TOKEN` secret. Needs an npm automation token (one that
    doesn't require interactive 2FA — generated from the npm web UI).
