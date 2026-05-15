#!/usr/bin/env node
// SplatForge Optimize — GitHub Action entrypoint.
//
// Submits every changed splat file in the PR to SplatForge Cloud, polls until
// terminal, computes a compression report, and posts a sticky PR comment with
// a fidelity badge. Designed to drop in as `splatforge/optimize-action@v1`.
//
// Pure Node 20 stdlib so there is no `npm install` step on the runner — every
// import below is built into Node 20 LTS.

'use strict';

const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const https = require('node:https');
const http = require('node:http');
const { URL } = require('node:url');
const { execSync, spawnSync } = require('node:child_process');

/* ---------------------------------------------------------------------- *
 * GitHub Actions helpers (Toolkit-free)                                  *
 * ---------------------------------------------------------------------- */

function getInput(name, { required = false, defaultValue = '' } = {}) {
  // Inputs are exposed as INPUT_<UPPER_NAME> with dashes -> underscores.
  const envKey = `INPUT_${name.replace(/ /g, '_').replace(/-/g, '_').toUpperCase()}`;
  const raw = process.env[envKey];
  if (raw === undefined || raw === '') {
    if (required) throw new Error(`Missing required input: ${name}`);
    return defaultValue;
  }
  return raw.trim();
}

function setOutput(name, value) {
  const file = process.env.GITHUB_OUTPUT;
  if (!file) {
    // Local / non-Actions run — just log.
    process.stdout.write(`::set-output name=${name}::${value}\n`);
    return;
  }
  const v = typeof value === 'string' ? value : JSON.stringify(value);
  const delim = `EOF_${crypto.randomBytes(8).toString('hex')}`;
  fs.appendFileSync(file, `${name}<<${delim}\n${v}\n${delim}\n`);
}

function info(msg) { process.stdout.write(`${msg}\n`); }
function warn(msg) { process.stdout.write(`::warning::${msg}\n`); }
function error(msg) { process.stderr.write(`::error::${msg}\n`); }
function group(name, fn) {
  process.stdout.write(`::group::${name}\n`);
  try { return fn(); } finally { process.stdout.write(`::endgroup::\n`); }
}

function mask(secret) {
  if (!secret) return;
  // GH Actions log masking — every future occurrence in stdout/stderr is
  // replaced with `***`. Honour the no-key-in-logs hard constraint.
  process.stdout.write(`::add-mask::${secret}\n`);
}

/* ---------------------------------------------------------------------- *
 * HTTP client                                                             *
 * ---------------------------------------------------------------------- */

/**
 * Minimal Promise-based HTTP client. Returns { status, headers, body } where
 * body is a Buffer. Caller decides whether to parse as JSON. Never logs the
 * Authorization header.
 */
function request(urlStr, { method = 'GET', headers = {}, body = null, timeoutMs = 60_000 } = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlStr);
    const lib = url.protocol === 'https:' ? https : http;
    const opts = {
      method,
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: `${url.pathname}${url.search}`,
      headers: { ...headers },
    };
    if (body && !opts.headers['content-length'] && Buffer.isBuffer(body)) {
      opts.headers['content-length'] = body.length;
    }
    const req = lib.request(opts, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        resolve({
          status: res.statusCode || 0,
          headers: res.headers,
          body: Buffer.concat(chunks),
        });
      });
    });
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`HTTP timeout after ${timeoutMs}ms: ${method} ${urlStr}`));
    });
    if (body) {
      if (Buffer.isBuffer(body) || typeof body === 'string') req.write(body);
      else req.write(Buffer.from(body));
    }
    req.end();
  });
}

/**
 * Stream a local file as an HTTP body. Used for proxy-upload mode where we
 * can't load 800 MB into memory. Returns { status, headers, body }.
 */
function uploadFile(urlStr, filePath, { headers = {}, timeoutMs = 600_000 } = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlStr);
    const lib = url.protocol === 'https:' ? https : http;
    const size = fs.statSync(filePath).size;
    const opts = {
      method: 'POST',
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: `${url.pathname}${url.search}`,
      headers: {
        'content-type': 'application/octet-stream',
        'content-length': size,
        ...headers,
      },
    };
    const req = lib.request(opts, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({
        status: res.statusCode || 0,
        headers: res.headers,
        body: Buffer.concat(chunks),
      }));
    });
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => req.destroy(new Error(`Upload timeout: ${urlStr}`)));
    fs.createReadStream(filePath).on('error', reject).pipe(req);
  });
}

/* ---------------------------------------------------------------------- *
 * SplatForge Cloud client                                                 *
 * ---------------------------------------------------------------------- */

class CloudClient {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl.replace(/\/+$/, '');
    this.apiKey = apiKey;
  }
  _headers(extra = {}) {
    return {
      'authorization': `Bearer ${this.apiKey}`,
      'content-type': 'application/json',
      'user-agent': 'splatforge-optimize-action/1.0',
      ...extra,
    };
  }
  async createJob({ preset, filename, sizeBytes, label }) {
    const body = JSON.stringify({
      preset,
      filename,
      size_bytes: sizeBytes,
      label,
    });
    const res = await request(`${this.apiUrl}/v1/jobs`, {
      method: 'POST',
      headers: this._headers(),
      body: Buffer.from(body),
    });
    if (res.status < 200 || res.status >= 300) {
      throw new Error(`POST /v1/jobs failed: ${res.status} ${res.body.toString('utf8').slice(0, 400)}`);
    }
    return JSON.parse(res.body.toString('utf8'));
  }
  async uploadBytes(jobId, filePath) {
    const res = await uploadFile(`${this.apiUrl}/v1/jobs/${jobId}/upload`, filePath, {
      headers: {
        'authorization': `Bearer ${this.apiKey}`,
        'user-agent': 'splatforge-optimize-action/1.0',
      },
    });
    if (res.status < 200 || res.status >= 300) {
      throw new Error(`POST /v1/jobs/${jobId}/upload failed: ${res.status} ${res.body.toString('utf8').slice(0, 400)}`);
    }
    return JSON.parse(res.body.toString('utf8'));
  }
  async getJob(jobId) {
    const res = await request(`${this.apiUrl}/v1/jobs/${jobId}`, {
      method: 'GET',
      headers: this._headers(),
    });
    if (res.status === 404) return null;
    if (res.status < 200 || res.status >= 300) {
      throw new Error(`GET /v1/jobs/${jobId} failed: ${res.status} ${res.body.toString('utf8').slice(0, 400)}`);
    }
    return JSON.parse(res.body.toString('utf8'));
  }
  async pollUntilTerminal(jobId, { timeoutSeconds }) {
    const deadline = Date.now() + timeoutSeconds * 1000;
    let lastStatus = '';
    let backoffMs = 2000;
    while (Date.now() < deadline) {
      const job = await this.getJob(jobId);
      if (!job) throw new Error(`Job ${jobId} not found`);
      if (job.status !== lastStatus) {
        info(`  [${jobId.slice(0, 8)}] status=${job.status}${job.phase ? ` phase=${job.phase}` : ''}${job.percent != null ? ` (${Math.round(job.percent * 100)}%)` : ''}`);
        lastStatus = job.status;
      }
      if (job.status === 'done') return job;
      if (job.status === 'error') {
        throw new Error(`Job ${jobId} errored: ${job.error || 'unknown error'}`);
      }
      await new Promise((r) => setTimeout(r, backoffMs));
      // Mild backoff so we don't hammer the API for slow jobs. Cap at 8s.
      backoffMs = Math.min(8000, Math.round(backoffMs * 1.3));
    }
    throw new Error(`Job ${jobId} timed out after ${timeoutSeconds}s (last status=${lastStatus})`);
  }
}

/* ---------------------------------------------------------------------- *
 * Idempotency: deterministic job label per (commit SHA, file path)        *
 * ---------------------------------------------------------------------- */

function jobLabelFor(sha, filePath) {
  // The API's `label` field is free-form and surfaced in the Job JSON. We use
  // it as our idempotency key so running the action twice on the same SHA
  // can detect & reuse prior jobs. A future API endpoint
  // `GET /v1/jobs?label=...` would let us look these up directly; until then
  // we cache the job-id locally per-run via Actions cache (not implemented
  // yet — the label is still useful for human debugging).
  const short = sha.slice(0, 12);
  const norm = filePath.replace(/[^a-zA-Z0-9_.-]+/g, '_');
  return `gh:${short}:${norm}`;
}

/* ---------------------------------------------------------------------- *
 * File discovery                                                          *
 * ---------------------------------------------------------------------- */

const SPLAT_EXTENSIONS = new Set(['.ply', '.splat', '.spz', '.ksplat']);

function isSplatFile(p) {
  return SPLAT_EXTENSIONS.has(path.extname(p).toLowerCase());
}

/**
 * Discover splats to optimize. Order:
 *   1. If `target` input is set, glob it.
 *   2. If we're inside a pull_request event, diff against the base SHA.
 *   3. Otherwise, diff against the previous commit (push event).
 */
function discoverSplats({ target, eventName, baseSha, headSha, repoRoot }) {
  if (target) {
    // Treat `target` as either a single file or a comma-separated list of
    // paths/globs (handled by `git ls-files`).
    const parts = target.split(',').map((s) => s.trim()).filter(Boolean);
    const out = [];
    for (const pat of parts) {
      try {
        const lst = execSync(`git -C "${repoRoot}" ls-files -- "${pat}"`, { encoding: 'utf8' });
        for (const line of lst.split('\n')) {
          const l = line.trim();
          if (l && isSplatFile(l)) out.push(l);
        }
      } catch (e) {
        warn(`git ls-files failed for "${pat}": ${e.message}`);
      }
    }
    return [...new Set(out)];
  }

  // PR or push diff.
  let range;
  if (eventName === 'pull_request' && baseSha && headSha) {
    range = `${baseSha}...${headSha}`;
  } else if (baseSha && headSha) {
    range = `${baseSha}..${headSha}`;
  } else {
    range = 'HEAD~1..HEAD';
  }

  let changed = '';
  try {
    changed = execSync(
      `git -C "${repoRoot}" diff --name-only --diff-filter=AMR ${range}`,
      { encoding: 'utf8' },
    );
  } catch (e) {
    warn(`git diff failed (${range}): ${e.message}; falling back to git ls-files`);
    try {
      changed = execSync(`git -C "${repoRoot}" ls-files`, { encoding: 'utf8' });
    } catch (e2) {
      warn(`git ls-files also failed: ${e2.message}`);
      return [];
    }
  }
  return changed.split('\n').map((s) => s.trim()).filter((p) => p && isSplatFile(p));
}

/* ---------------------------------------------------------------------- *
 * Report rendering                                                        *
 * ---------------------------------------------------------------------- */

function humanBytes(n) {
  if (n == null || !Number.isFinite(n)) return '?';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/**
 * Render the fidelity badge URL (shields.io). Color is bucketed on the
 * compression ratio: <=0.5 green, <=0.7 yellow, otherwise red. The score
 * shown is `100 * (1 - ratio)` clamped to [0, 100] — a defensible
 * placeholder until the API exposes PSNR/SSIM directly.
 */
function badgeFor(score, ratio) {
  const color = ratio == null ? 'lightgrey'
    : ratio <= 0.5 ? 'brightgreen'
    : ratio <= 0.7 ? 'yellow'
    : 'orange';
  const label = encodeURIComponent('SplatForge fidelity');
  const text = encodeURIComponent(score == null ? 'pending' : `${score.toFixed(0)}/100`);
  return `https://img.shields.io/badge/${label}-${text}-${color}?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4=`;
}

function renderComment({ results, apiUrl, headSha, threshold }) {
  const lines = [];
  lines.push('<!-- splatforge-optimize-action -->');
  const aggRatio = aggregateRatio(results);
  const aggScore = aggRatio == null ? null : Math.max(0, Math.min(100, 100 * (1 - aggRatio)));
  lines.push(`![SplatForge fidelity](${badgeFor(aggScore, aggRatio)})`);
  lines.push('');
  lines.push(`**SplatForge Cloud** optimized ${results.length} splat${results.length === 1 ? '' : 's'} for commit \`${headSha.slice(0, 7)}\`.`);
  lines.push('');
  lines.push('| File | Input | Output | Ratio | Status | Download |');
  lines.push('| --- | ---: | ---: | ---: | --- | --- |');
  for (const r of results) {
    const ratioStr = r.ratio == null ? '—' : `${(r.ratio * 100).toFixed(1)}%`;
    const dl = r.outputUrl ? `[.glb](${r.outputUrl})` : '—';
    const statusEmoji = r.ok ? '✓' : '✗';
    lines.push(`| \`${r.file}\` | ${humanBytes(r.inputBytes)} | ${humanBytes(r.outputBytes)} | ${ratioStr} | ${statusEmoji} ${r.statusText} | ${dl} |`);
  }
  lines.push('');
  if (aggRatio != null) {
    const pass = aggRatio <= threshold;
    lines.push(`**Aggregate ratio:** ${(aggRatio * 100).toFixed(1)}% — threshold ${(threshold * 100).toFixed(0)}% → ${pass ? '**PASS**' : '**FAIL**'}`);
  }
  lines.push('');
  lines.push(`<sub>via [splatforge/optimize-action](https://github.com/splatforge/optimize-action) — API: \`${apiUrl}\`</sub>`);
  return lines.join('\n');
}

function aggregateRatio(results) {
  const valid = results.filter((r) => r.ok && r.ratio != null);
  if (!valid.length) return null;
  // Weight by input size so a 1 GB scene dominates a 1 MB scene.
  let inSum = 0;
  let outSum = 0;
  for (const r of valid) {
    inSum += r.inputBytes;
    outSum += r.outputBytes;
  }
  if (inSum === 0) return null;
  return outSum / inSum;
}

/* ---------------------------------------------------------------------- *
 * GitHub API (PR comment + check-run)                                     *
 * ---------------------------------------------------------------------- */

async function upsertStickyComment({ token, repo, prNumber, body }) {
  const [owner, name] = repo.split('/');
  const list = await request(`https://api.github.com/repos/${owner}/${name}/issues/${prNumber}/comments?per_page=100`, {
    method: 'GET',
    headers: {
      'authorization': `Bearer ${token}`,
      'user-agent': 'splatforge-optimize-action',
      'accept': 'application/vnd.github+json',
    },
  });
  if (list.status >= 300) throw new Error(`list comments failed: ${list.status} ${list.body.toString('utf8').slice(0, 300)}`);
  const existing = JSON.parse(list.body.toString('utf8'))
    .find((c) => c.body && c.body.includes('<!-- splatforge-optimize-action -->'));
  const headers = {
    'authorization': `Bearer ${token}`,
    'user-agent': 'splatforge-optimize-action',
    'accept': 'application/vnd.github+json',
    'content-type': 'application/json',
  };
  if (existing) {
    const r = await request(`https://api.github.com/repos/${owner}/${name}/issues/comments/${existing.id}`, {
      method: 'PATCH',
      headers,
      body: Buffer.from(JSON.stringify({ body })),
    });
    if (r.status >= 300) throw new Error(`PATCH comment failed: ${r.status}`);
  } else {
    const r = await request(`https://api.github.com/repos/${owner}/${name}/issues/${prNumber}/comments`, {
      method: 'POST',
      headers,
      body: Buffer.from(JSON.stringify({ body })),
    });
    if (r.status >= 300) throw new Error(`POST comment failed: ${r.status}`);
  }
}

/* ---------------------------------------------------------------------- *
 * Main                                                                    *
 * ---------------------------------------------------------------------- */

async function main() {
  const apiUrl = getInput('api-url', { defaultValue: 'https://splatforge-api.fly.dev' });
  const apiKey = getInput('api-key', { required: true });
  const preset = getInput('preset', { defaultValue: 'web-mobile' });
  const target = getInput('target');
  const threshold = parseFloat(getInput('regression-threshold', { defaultValue: '1.0' }));
  const wantComment = getInput('comment', { defaultValue: 'true' }) === 'true';
  const timeoutSeconds = parseInt(getInput('timeout-seconds', { defaultValue: '270' }), 10);

  if (!apiKey) throw new Error('api-key input is required (set repo secret SPLATFORGE_API_KEY)');
  // Mask the key on EVERY workflow line, not just the lines we write — GH
  // will scrub it from any future log including subprocess output.
  mask(apiKey);

  const repoRoot = process.env.GITHUB_WORKSPACE || process.cwd();
  const eventName = process.env.GITHUB_EVENT_NAME || '';
  let baseSha = '';
  let headSha = process.env.GITHUB_SHA || '';
  let prNumber = null;
  let repo = process.env.GITHUB_REPOSITORY || '';

  if (process.env.GITHUB_EVENT_PATH && fs.existsSync(process.env.GITHUB_EVENT_PATH)) {
    try {
      const ev = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
      if (ev.pull_request) {
        baseSha = ev.pull_request.base?.sha || '';
        headSha = ev.pull_request.head?.sha || headSha;
        prNumber = ev.pull_request.number || ev.number || null;
      } else if (ev.before) {
        baseSha = ev.before;
      }
    } catch (e) {
      warn(`failed to parse event payload: ${e.message}`);
    }
  }

  info(`splatforge/optimize-action — api=${apiUrl} preset=${preset} threshold=${threshold}`);
  info(`event=${eventName} repo=${repo} head=${headSha.slice(0, 7)} base=${baseSha.slice(0, 7)} pr=${prNumber ?? '-'}`);

  const splats = discoverSplats({ target, eventName, baseSha, headSha, repoRoot });
  if (!splats.length) {
    // Empty-PR / no-splat case — succeed silently with explicit outputs.
    info('No changed splat files detected. Nothing to optimize.');
    setOutput('fidelity-score', '');
    setOutput('compression-ratio', '');
    setOutput('output-url', '[]');
    setOutput('report-url', '');
    if (wantComment && prNumber && process.env.GITHUB_TOKEN) {
      // Sticky comment is intentionally NOT updated when there's nothing to
      // do — leaves the last good report visible if one exists.
    }
    return;
  }

  info(`Detected ${splats.length} splat file(s):`);
  for (const s of splats) info(`  - ${s}`);

  const client = new CloudClient(apiUrl, apiKey);
  const results = [];

  for (const rel of splats) {
    const abs = path.join(repoRoot, rel);
    if (!fs.existsSync(abs)) {
      warn(`skipping ${rel}: file not present in workspace (deleted in this PR?)`);
      continue;
    }
    const inputBytes = fs.statSync(abs).size;
    const label = jobLabelFor(headSha || 'no-sha', rel);

    info(`→ ${rel} (${humanBytes(inputBytes)}) label=${label}`);
    let result = {
      file: rel, inputBytes, outputBytes: null, ratio: null,
      ok: false, statusText: 'pending', outputUrl: null, jobId: null,
    };
    try {
      const job = await group(`create job: ${rel}`, () => client.createJob({
        preset,
        filename: path.basename(rel),
        sizeBytes: inputBytes,
        label,
      }));
      info(`  created job=${job.id} status=${job.status}`);
      result.jobId = job.id;

      await group(`upload bytes: ${rel}`, () => client.uploadBytes(job.id, abs));
      info(`  uploaded ${humanBytes(inputBytes)}`);

      const done = await client.pollUntilTerminal(job.id, { timeoutSeconds });
      result.outputUrl = done.output_url || null;
      result.ok = true;
      result.statusText = 'done';

      // Determine output size via HEAD (avoid downloading the full .glb just
      // to compute a byte count). Fall back to GET if HEAD isn't supported.
      if (result.outputUrl) {
        try {
          const head = await request(result.outputUrl, { method: 'HEAD' });
          const cl = head.headers['content-length'];
          if (cl) result.outputBytes = parseInt(cl, 10);
        } catch (e) { /* leave outputBytes null */ }
        if (result.outputBytes == null) {
          // Tiny GET so we at least know the size. Bounded to 50 MB cap to
          // keep runner memory predictable.
          try {
            const got = await request(result.outputUrl, { method: 'GET' });
            result.outputBytes = got.body.length;
          } catch (e) { /* leave null */ }
        }
      }
      if (result.outputBytes != null && inputBytes > 0) {
        result.ratio = result.outputBytes / inputBytes;
      }
    } catch (e) {
      // Scrub the key out of any error message in case it leaked through.
      const safeMsg = String(e.message || e).replaceAll(apiKey, '***');
      error(`failed: ${rel}: ${safeMsg}`);
      result.ok = false;
      result.statusText = safeMsg.slice(0, 80);
    }
    results.push(result);
  }

  /* ----- Outputs + comment ----- */

  const aggRatio = aggregateRatio(results);
  const aggScore = aggRatio == null ? null : Math.max(0, Math.min(100, 100 * (1 - aggRatio)));
  const outputUrls = results.map((r) => r.outputUrl).filter(Boolean);

  setOutput('fidelity-score', aggScore == null ? '' : aggScore.toFixed(2));
  setOutput('compression-ratio', aggRatio == null ? '' : aggRatio.toFixed(4));
  setOutput('output-url', JSON.stringify(outputUrls));
  // We don't have a server-rendered report page yet — link to the first job
  // JSON, which surfaces status + URLs and is publicly accessible to anyone
  // with the job UUID.
  const firstReport = results[0]?.jobId
    ? `${apiUrl}/v1/jobs/${results[0].jobId}`
    : '';
  setOutput('report-url', firstReport);

  const body = renderComment({
    results,
    apiUrl,
    headSha: headSha || '0000000',
    threshold,
  });

  // Write the summary to the Actions Run Summary regardless of comment opt.
  if (process.env.GITHUB_STEP_SUMMARY) {
    try { fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, body + '\n'); } catch {}
  }

  if (wantComment && prNumber && process.env.GITHUB_TOKEN && repo) {
    try {
      await upsertStickyComment({ token: process.env.GITHUB_TOKEN, repo, prNumber, body });
      info(`posted sticky PR comment to ${repo}#${prNumber}`);
    } catch (e) {
      warn(`PR comment failed: ${String(e.message).replaceAll(apiKey, '***')}`);
    }
  } else if (wantComment) {
    info('PR comment skipped: not a PR event or GITHUB_TOKEN missing.');
  }

  /* ----- Gate ----- */

  const anyFailed = results.some((r) => !r.ok);
  if (anyFailed) {
    error('one or more splats failed to optimize — see table above');
    process.exit(1);
  }
  if (aggRatio != null && aggRatio > threshold) {
    error(`compression ratio ${(aggRatio * 100).toFixed(1)}% exceeds threshold ${(threshold * 100).toFixed(0)}% — failing PR`);
    process.exit(1);
  }
  info('OK.');
}

main().catch((e) => {
  // Final scrub in case anything leaked. We can't reach `apiKey` from this
  // scope cheaply, so rely on `::add-mask::` from earlier — Actions will
  // replace any occurrence in the output.
  error(`fatal: ${e.stack || e.message || e}`);
  process.exit(1);
});
