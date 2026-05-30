# READY_TO_PUBLISH — Catetus MCP v1 release checklist

End-to-end runbook for shipping `@catetus/mcp` across all three distribution
channels: **npm**, **MCPB** (Claude Desktop), and the **Anthropic Connector
Directory** (hosted at `mcp.catetus.com`).

> Current published version targets: `1.0.0-alpha.1` (first npm + MCPB cut),
> then iterate to `1.0.0` once the alpha smoke-tests cleanly on real users.
> The version in `package.json` and `mcpb-manifest.json` MUST stay in lockstep.

> Owner: implementer B. Cross-references the architecture spec at
> `splatforge-private/docs/mcp/ARCHITECTURE.md`.

---

## 0. Pre-flight (do once)

- [ ] **npm org** `@catetus` exists and you are a member with `publish` scope.
      Check: `npm org ls catetus` → your username should appear with role `developer` or higher.
      Create it (one-time): `npm org create catetus` (free for public packages).
- [ ] **npm 2FA** enabled on the publishing account (`npm profile enable-2fa auth-and-writes`).
- [ ] **GitHub repo** `github.com/catetus/catetus-mcp` exists, public, MIT-licensed.
- [ ] **GitHub release-token secret** `NPM_TOKEN` set in repo settings → Actions secrets
      (Automation token, not a publish token — required for CI publishing under 2FA).
- [ ] **Apple developer cert** (optional, for `.mcpb` signing) installed in the
      keychain on whichever machine signs releases.
- [ ] **DNS** for `mcp.catetus.com` points at the chosen host (Fly.io app or
      Cloudflare Worker — see open decision §19.4 in ARCHITECTURE.md).
- [ ] **Anthropic Connector Directory** application form bookmarked
      (see §4 below for current URL).
- [ ] **Test API key** (`cat_test_…`) and **production API key** (`cat_live_…`)
      both work against `api.catetus.com`. Smoke: `curl -H "Authorization: Bearer
      cat_test_…" https://api.catetus.com/v1/me/usage` should return 200.

---

## 1. The definition-of-done gate

Before any publish step, every box below must be green (see ARCHITECTURE.md §18).

- [ ] All 15 tools registered, Zod schemas parse cleanly (`npm run typecheck`).
- [ ] All 7 resources + 1 template registered.
- [ ] All 4 prompts registered.
- [ ] `tools/list` filters by tier correctly (paid hidden without key).
- [ ] Both transports boot: `node dist/index.js --transport stdio` and
      `node dist/index.js --transport http --port 3000`.
- [ ] MCP Inspector smoke green: `npm run smoke`.
- [ ] CI green on Node 20 and 22: lint + typecheck + unit + integration.
- [ ] `docs/ERRORS.md` matches in-source `errorResult(...)` codes (lint rule).
- [ ] Migration banner committed to `SplatForge/crates/catetus-mcp/README.md`.
- [ ] `examples/*.json` (6 files) all present and valid JSON
      (`jq . examples/*.json` returns no errors — note: strip `//` comments first).
- [ ] At least one `evals/scenarios/*.yaml` passes end-to-end.

If any box is unchecked, **stop** and fix before continuing.

---

## 2. npm publish (`@catetus/mcp`)

### 2.1 Local pre-publish smoke

```bash
cd packages/mcp

# Fresh install + build
rm -rf node_modules dist
npm ci
npm run build

# Confirm both transports boot cleanly
node dist/server-stdio.js < /dev/null   # should print nothing on stdout (waits for JSON-RPC on stdin)
PORT=3000 node dist/server-http.js &     # should listen on :3000; test with `curl localhost:3000/healthz`
kill %1

# Use the MCP Inspector to list tools (smoke):
npm run inspector                        # → prints the tool list

# Verify the package contents that will ship
npm pack --dry-run                       # prints the file list

# Confirm only `dist/`, README.md, LICENSE are in the tarball.
# If extra files leak in, fix the `files` array in package.json or add to .npmignore.
```

### 2.2 Version + tag

```bash
# Bump version in package.json AND in mcpb-manifest.json (they must match)
# Tag format: v1.0.0-alpha.1 (or whatever you're cutting)
cd packages/mcp
VERSION=1.0.0-alpha.1
npm version "$VERSION" --no-git-tag-version
# Update mcpb-manifest.json `"version"` to match (the package.json change above doesn't touch it):
node -e "const f='mcpb-manifest.json'; const m=JSON.parse(require('fs').readFileSync(f,'utf8')); m.version=process.env.VERSION; require('fs').writeFileSync(f, JSON.stringify(m,null,2)+'\n');" VERSION="$VERSION"

git add packages/mcp/package.json packages/mcp/mcpb-manifest.json
git commit -m "chore(mcp): bump to v$VERSION"
git tag "v$VERSION"
git push origin main --tags
```

### 2.3 Publish

**Preferred — via CI** (`.github/workflows/publish-npm.yml` runs on tag push):

The CI workflow should do:
```yaml
- run: npm ci
- run: npm run build
# Use --tag alpha (or beta/rc) for pre-release versions so they don't become @latest:
- run: npm publish --access public --tag alpha
  env:
    NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

For the first 1.0.0 stable cut, drop `--tag alpha` (defaults to `latest`).

Wait for the workflow to go green. Then verify:

```bash
npm view @catetus/mcp version          # latest stable
npm view @catetus/mcp versions         # full history
npm view @catetus/mcp dist-tags        # latest, alpha, etc.
```

**Fallback — manual publish from your machine:**

```bash
cd packages/mcp
npm login                              # if not already
npm publish --access public --tag alpha --otp <6-digit-otp-from-authenticator>
# Drop --tag alpha for the 1.0.0 stable cut.
```

### 2.4 Post-publish smoke

On a clean machine (or fresh Docker container):

```bash
# Validate the live npm tarball works end-to-end (substitute the version you just published)
npx -y @catetus/mcp@1.0.0-alpha.1 < /dev/null      # boots stdio server; Ctrl-C to exit
npx -y @modelcontextprotocol/inspector --cli npx -y @catetus/mcp@1.0.0-alpha.1 --method tools/list
```

The Inspector should list the free tools (or all 15 with `--env CATETUS_API_KEY=…`).

### 2.5 Rollback procedure (only if production-breaking bug discovered)

You have **72 hours** to deprecate-then-publish-replacement; outright unpublish
is forbidden by npm for packages with downloads.

```bash
# 1. Publish a patch fix immediately (preferred)
npm version 1.0.1 ...
npm publish

# 2. If the bad version must not be installed by `@latest`, move the dist-tag:
npm dist-tag add @catetus/mcp@1.0.1 latest
npm dist-tag rm  @catetus/mcp@1.0.0 latest  # safe; just removes the latest pointer

# 3. As a last resort, deprecate with an actionable message:
npm deprecate @catetus/mcp@1.0.0 "Critical bug; upgrade to 1.0.1 via 'npx @catetus/mcp@latest'."
```

---

## 3. MCPB bundle publish (Claude Desktop one-click)

The MCPB bundle is **independent** of npm — users who install the `.mcpb` do
not need npm. But the bundle ships the same compiled code.

### 3.1 Build the bundle

```bash
cd packages/mcp
npm ci --omit=dev                      # production deps only — keeps the bundle small
npm run build
bash scripts/build-mcpb.sh             # writes dist-bundle/catetus-<version>.mcpb

# Validate one more time before signing
bash scripts/build-mcpb.sh --validate
```

### 3.2 Sign the bundle (recommended for distribution)

```bash
# Requires an Apple Developer ID Application certificate in the keychain.
# Signing prevents Gatekeeper warnings on macOS.
bash scripts/build-mcpb.sh --sign
# Produces dist-bundle/catetus-<version>.mcpb.sig
```

If you don't have the Apple cert yet, skip signing for v1 — users will see an
"unverified developer" prompt but installs still work. File a follow-up to get
the cert before v1.1.

### 3.3 Test locally before publishing

On a machine **without** the Catetus dev environment:

1. Download the `.mcpb` from your build machine.
2. Drag onto Claude Desktop (must be ≥0.10.0).
3. Verify the install dialog shows: name, version, tool list (15 tools),
   user_config fields (api_key, api_base, log_level, allowlist_roots).
4. Click Install, enter a test API key, click Connect.
5. In Claude Desktop, ask: *"List the Catetus tools you have access to"* —
   should see all 15.
6. Ask: *"Read the Catetus canonical-11 leaderboard resource"* — should return
   the leaderboard JSON.

### 3.4 Publish the bundle to GitHub releases

```bash
# Tag should already exist from step 2.2 (e.g. v1.0.0-alpha.1).
VERSION=1.0.0-alpha.1
gh release create "v$VERSION" \
  --repo catetus/catetus-mcp \
  --title "Catetus MCP v$VERSION" \
  --prerelease \
  --notes-file packages/mcp/CHANGELOG.md \
  "packages/mcp/dist-bundle/catetus-$VERSION.mcpb" \
  "packages/mcp/dist-bundle/catetus-$VERSION.mcpb.sig"

# For the stable 1.0.0 cut: drop --prerelease.
```

CI alternative (`.github/workflows/publish-mcpb.yml`) — runs on tag push,
builds + uploads automatically. Recommended.

After publish, update the README's MCPB install link to point at the new release.

---

## 4. Anthropic Connector Directory submission (hosted `mcp.catetus.com`)

For users who want to add Catetus via Claude.ai → Settings → Connectors,
the server must be listed in the Anthropic Connector Directory.

### 4.1 Pre-requisites for directory submission

Per Anthropic's directory acceptance criteria (current as of 2026-05):

- [ ] Server is publicly reachable at a stable HTTPS URL with valid TLS.
- [ ] Server speaks MCP **Streamable HTTP** (spec 2025-11-25 or compatible).
- [ ] `GET /healthz` returns 200 with JSON.
- [ ] `GET /.well-known/mcp.json` returns discovery metadata
      (server name, version, capabilities — see ARCHITECTURE.md §3.2).
- [ ] Server validates `Origin` header (DNS rebinding prevention).
- [ ] CORS is allowlisted (no `*`); includes `https://claude.ai`.
- [ ] Auth is **either** unauthenticated, OAuth (DCR), or Bearer token with
      `WWW-Authenticate` returning a clear `error="invalid_token"` on bad auth.
      *Catetus uses Bearer tokens — this is supported but flagged as Tier-1
      simple auth; Tier-2 OAuth is a Phase 2 upgrade.*
- [ ] Read-only and write tools are separated (no single tool that does both).
- [ ] Every tool ≤64 chars name, has a description, has annotations.
- [ ] Privacy policy URL: `https://catetus.com/privacy`
- [ ] Terms of service URL: `https://catetus.com/terms`
- [ ] Support contact: `support@catetus.com`

### 4.2 Deploy `mcp.catetus.com`

Per ARCHITECTURE.md §19.4 — pick Fly.io (matches existing api.catetus.com) or
Cloudflare Workers (simpler ops). Default recommendation: **Fly.io** for
consistency with the rest of the Catetus backend.

```bash
# (Fly.io path — adapt for your hosting choice)
# Dockerfile at packages/mcp/Dockerfile (implementer B authors, not yet in tree):
#   FROM node:20-alpine
#   WORKDIR /app
#   COPY package*.json ./
#   RUN npm ci --omit=dev
#   COPY dist/ ./dist/
#   ENV PORT=8080
#   EXPOSE 8080
#   CMD ["node", "dist/server-http.js"]

fly launch --name catetus-mcp --region iad --no-deploy
fly secrets set CATETUS_API_BASE=https://api.catetus.com
fly deploy
fly certs add mcp.catetus.com
# Point mcp.catetus.com DNS A/AAAA at the Fly app address shown by `fly ips list`.
```

Post-deploy smoke:

```bash
curl -fsSL https://mcp.catetus.com/healthz       # → {"status":"ok",...}
curl -fsSL https://mcp.catetus.com/.well-known/mcp.json
# POST a tools/list against it (requires a real key for paid tools)
curl -X POST https://mcp.catetus.com/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-11-25" \
  -H "Authorization: Bearer cat_test_..." \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### 4.3 Submit to the directory

1. Go to <https://claude.com/docs/connectors/building/submitting> (or the
   current Anthropic directory submission URL — bookmarked in §0).
2. Fill the application:
   - **Connector name:** Catetus
   - **Server URL:** `https://mcp.catetus.com`
   - **Description:** copy from `mcpb-manifest.json` `description`
   - **Long description:** copy from `mcpb-manifest.json` `long_description`
   - **Category:** Developer Tools (or 3D / Graphics if available)
   - **Logo:** `packages/mcp/icon.png` (256×256 PNG, transparent background)
   - **Pricing model:** Freemium (free tier + paid API key)
   - **Privacy policy:** `https://catetus.com/privacy`
   - **Terms of service:** `https://catetus.com/terms`
   - **Support email:** `support@catetus.com`
3. Submit. Review SLA is typically 5–10 business days.
4. Track the application thread; expect feedback on at least one of:
   - Tool annotations completeness (we cover this in §5.2 of ARCHITECTURE.md)
   - Error envelope quality (we cover this in §10)
   - Rate-limiting story for unauthenticated calls (see §17 — 100/h read,
     1000/h catalog)

### 4.4 Cursor / Cline / Zed listing (optional, separate)

Each editor has its own directory:

- **Cursor:** <https://docs.cursor.com/mcp> (community-maintained)
- **Cline:** <https://github.com/cline/mcp-marketplace> (PR your server)
- **Zed:** built into Zed's extension marketplace; submit via the Zed extensions
  repo

For each, the same `mcp.catetus.com` URL applies. Submit after the Anthropic
directory listing is live (it's social proof for the others).

---

## 5. Post-release announcement checklist

After all three channels are live:

- [ ] Update `https://catetus.com` landing page with the install snippets
      (lift from `packages/mcp/examples/README.md`).
- [ ] Publish a blog post / launch tweet: "Catetus MCP is live — install with
      `npx @catetus/mcp@latest` or one-click via the .mcpb."
- [ ] Add a "Get Catetus in Claude" badge to the README:
      `[![Install on Claude](https://...)](https://mcp.catetus.com)` (link to a
      Claude Desktop deep-link if one exists, else to the install docs).
- [ ] Open the deprecation banner PR for `crates/catetus-mcp/README.md`
      (ARCHITECTURE.md §14.2 T+0 action).
- [ ] Notify any dogfooding users who were on the Rust binary to migrate.
- [ ] Mark the v1.0.0 milestone closed on the `catetus-mcp` repo.

---

## 6. Versioning thereafter

Follow semver as locked in ARCHITECTURE.md §16:

- **Patch** (1.0.x): bugfix releases. Publish via tag → CI. No re-validation
  of MCPB manifest required.
- **Minor** (1.x.0): adds tools/resources/prompts/optional fields. Bump
  `mcpb-manifest.json` `version` to match `package.json`. Re-build and
  re-distribute the `.mcpb`. Update the `tools` array in the manifest.
- **Major** (x.0.0): breaking. Update Anthropic Connector Directory listing
  description; consider co-publishing a final `1.x` patch with the deprecation
  notice. Coordinate timing with `api.catetus.com` if backend contracts shift.

---

## 7. Quick-reference: every artifact and its source of truth

| Artifact | Source of truth | Re-publish on |
|---|---|---|
| `@catetus/mcp` npm package | `packages/mcp/package.json` `version` | every release (patch/minor/major) |
| `catetus-<v>.mcpb` GitHub release asset | `packages/mcp/mcpb-manifest.json` `version` (must match npm) | minor + major (patch optional) |
| `mcp.catetus.com` deployment | latest `main` branch image | every release; auto-deploy on tag |
| Anthropic Directory listing | submission form fields (long-lived) | only on major or when adding tools |
| `examples/*.json` install snippets | `packages/mcp/examples/` | only when public-API contract shifts |

When in doubt: bump **all three** versions together (`package.json`,
`mcpb-manifest.json`, git tag) and re-cut everything. The cost is low; the
confusion of mismatched versions is high.

---

## 8. If something goes wrong

| Failure | What to do |
|---|---|
| `npm publish` errors on `403 Forbidden` | OTP expired — re-enter `--otp`; or token scope wrong — regenerate Automation token. |
| MCPB bundle fails to install in Claude Desktop | Check `dist-bundle/build/manifest.json` validates: `npx @anthropic-ai/mcpb validate dist-bundle/build/manifest.json`. Common cause: `entry_point` doesn't exist under `server/`. |
| MCPB installs but server crashes on boot | Run `node dist-bundle/build/server/index.js` standalone — captures the crash. Usually missing `node_modules/` in the bundle. Re-run `npm ci --omit=dev` before `build-mcpb.sh`. |
| `mcp.catetus.com` returns 401 from the directory's automated checks | Anthropic's checker sends an unauthenticated `initialize` — server must respond with public-tier capability set, NOT 401. Verify `resolveTier()` returns `{ tier: "public" }` on missing `Authorization`. |
| Connector Directory rejects with "tool annotations missing" | Audit every `registerTool` call — every annotation in ARCHITECTURE.md §5.2 must be set. CI lint rule recommended. |
| `npx @catetus/mcp@latest` slow on first run | Normal — npm caches after first fetch. To pre-warm in CI, add `npx -y @catetus/mcp@latest --version` to a setup step. |

---

**Last updated:** when this file changes, also update the CHANGELOG and bump
the package version if the release process itself changed.
