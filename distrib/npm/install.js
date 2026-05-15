#!/usr/bin/env node
// SplatForge npm postinstall.
//
// Downloads the per-arch binary archive for the current package version
// from the SplatForge GitHub release, verifies its SHA-256 against the
// release's SHASUMS256.txt manifest, and extracts the three binaries
// (`splatforge`, `splatforge-khr-validate`, `splatforge-usd-validate`)
// into <package>/native/. The thin JS shims in `bin/` exec into those.
//
// We deliberately DO NOT bundle the binary in the npm tarball. Each
// archive is ~2-15 MB and we ship five of them; combined that would
// balloon the registry payload to >75 MB per release. Postinstall fetch
// keeps the registry clean.
//
// Failure modes:
//   - No network / 404      -> exits non-zero, prints actionable error.
//                              We do NOT silently fall back, because then
//                              `npm i -g @splatforge/cli` would "succeed"
//                              with a broken `splatforge` shim and the
//                              user would only learn at first run.
//   - SHA-256 mismatch       -> exits non-zero with the expected/actual.
//   - Unsupported OS/arch    -> exits non-zero; lists supported matrix.
//   - SPLATFORGE_SKIP_DOWNLOAD=1 -> bypasses the download (test/CI).

"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const crypto = require("crypto");
const https = require("https");
const zlib = require("zlib");
const { spawnSync } = require("child_process");

const PKG = require("./package.json");
const VERSION = PKG.version;
const TAG = `v${VERSION}`;
const REPO = "splatforge/splatforge";
const RELEASE_URL = `https://github.com/${REPO}/releases/download/${TAG}`;
const NATIVE_DIR = path.join(__dirname, "native");

// (node-platform, node-arch) -> rust target triple.
const TARGET_MAP = {
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64":   "x86_64-apple-darwin",
  "linux-x64":    "x86_64-unknown-linux-gnu",
  "linux-arm64":  "aarch64-unknown-linux-gnu",
  "win32-x64":    "x86_64-pc-windows-msvc",
};

const BINS = ["splatforge", "splatforge-khr-validate", "splatforge-usd-validate"];

function fail(msg) {
  // Use stderr + non-zero exit. Don't `throw` — npm prints throws as
  // unhelpful stack traces to the user.
  process.stderr.write(`\n[@splatforge/cli postinstall] ${msg}\n\n`);
  process.exit(1);
}

function info(msg) {
  process.stdout.write(`[@splatforge/cli] ${msg}\n`);
}

function targetForHost() {
  const key = `${process.platform}-${process.arch}`;
  const target = TARGET_MAP[key];
  if (!target) {
    fail(
      `Unsupported platform/arch: ${key}. ` +
      `Supported: ${Object.keys(TARGET_MAP).join(", ")}. ` +
      `Install from source via 'cargo install --path crates/splatforge-cli' instead.`
    );
  }
  return target;
}

function archiveExt(target) {
  return target.includes("windows") ? "zip" : "tar.gz";
}

// --------------------------------------------------------------------------
// HTTP: follow redirects, stream to disk, never buffer the whole archive.
// --------------------------------------------------------------------------
function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const req = https.get(url, { headers: { "User-Agent": "splatforge-npm-installer" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        file.close();
        fs.unlinkSync(dest);
        return resolve(download(res.headers.location, dest));
      }
      if (res.statusCode !== 200) {
        file.close();
        fs.unlinkSync(dest);
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      res.pipe(file);
      file.on("finish", () => file.close(resolve));
    });
    req.on("error", (err) => {
      file.close();
      try { fs.unlinkSync(dest); } catch (_) {}
      reject(err);
    });
  });
}

function fetchText(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { "User-Agent": "splatforge-npm-installer" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return resolve(fetchText(res.headers.location));
      }
      if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      let buf = "";
      res.setEncoding("utf8");
      res.on("data", (c) => { buf += c; });
      res.on("end", () => resolve(buf));
    }).on("error", reject);
  });
}

function sha256File(p) {
  const h = crypto.createHash("sha256");
  h.update(fs.readFileSync(p));
  return h.digest("hex");
}

// --------------------------------------------------------------------------
// Archive extraction — tar.gz via streams + a tiny POSIX-ustar tar reader,
// zip via Node-bundled `tar.exe` on Windows (built-in since Win10 1803).
// --------------------------------------------------------------------------
async function extractTarGz(archive, outDir) {
  return new Promise((resolve, reject) => {
    const gunzip = zlib.createGunzip();
    const chunks = [];
    gunzip.on("data", (c) => chunks.push(c));
    gunzip.on("end", () => {
      try {
        const buf = Buffer.concat(chunks);
        let off = 0;
        while (off + 512 <= buf.length) {
          const header = buf.slice(off, off + 512);
          // POSIX ustar: name @ 0..100, size @ 124..136 (octal), type @ 156.
          const name = header.slice(0, 100).toString("utf8").replace(/\0.*$/, "");
          if (!name) break; // two zero blocks => EOF
          const sizeStr = header.slice(124, 136).toString("utf8").replace(/[\0 ]/g, "");
          const size = parseInt(sizeStr, 8) || 0;
          const type = header[156]; // '0' or 0 = regular file
          off += 512;
          const data = buf.slice(off, off + size);
          off += Math.ceil(size / 512) * 512;
          if ((type === 0x30 || type === 0) && size > 0) {
            const base = path.basename(name);
            if (BINS.includes(base) || BINS.map(b => b + ".exe").includes(base)) {
              const outPath = path.join(outDir, base);
              fs.writeFileSync(outPath, data, { mode: 0o755 });
            }
          }
        }
        resolve();
      } catch (e) { reject(e); }
    });
    gunzip.on("error", reject);
    fs.createReadStream(archive).pipe(gunzip);
  });
}

function extractZip(archive, outDir) {
  // On Windows we always have `tar.exe` (built-in since Win10 1803) which
  // also reads .zip. Use it — robust, avoids hand-rolling a zip parser.
  const r = spawnSync("tar", ["-xf", archive, "-C", outDir], { stdio: "inherit" });
  if (r.status !== 0) throw new Error("tar extraction failed");
}

// --------------------------------------------------------------------------
// Main.
// --------------------------------------------------------------------------
(async () => {
  if (process.env.SPLATFORGE_SKIP_DOWNLOAD === "1") {
    info("SPLATFORGE_SKIP_DOWNLOAD=1 — skipping binary download.");
    return;
  }

  // `--check-only` mode just verifies install.js itself parses + the
  // platform is recognized. Used by `npm test`.
  if (process.argv.includes("--check-only")) {
    targetForHost();
    info("ok");
    return;
  }

  const target = targetForHost();
  const ext = archiveExt(target);
  const archiveName = `splatforge-${TAG}-${target}.${ext}`;
  const archiveUrl = `${RELEASE_URL}/${archiveName}`;
  const manifestUrl = `${RELEASE_URL}/SHASUMS256.txt`;

  fs.mkdirSync(NATIVE_DIR, { recursive: true });
  const tmpArchive = path.join(os.tmpdir(), archiveName);

  info(`fetching ${archiveName}`);
  try {
    await download(archiveUrl, tmpArchive);
  } catch (e) {
    fail(
      `Failed to download ${archiveUrl}: ${e.message}\n` +
      `If you are offline or behind a strict firewall, set ` +
      `SPLATFORGE_SKIP_DOWNLOAD=1 and install the binary manually from ` +
      `${RELEASE_URL}.`
    );
  }

  let manifest = "";
  try {
    manifest = await fetchText(manifestUrl);
  } catch (e) {
    fail(`Failed to fetch SHASUMS256.txt manifest: ${e.message}`);
  }

  // SHASUMS256.txt is plain `sha256sum` output: "<hex>  <filename>".
  const want = manifest
    .split("\n")
    .map((l) => l.trim())
    .map((l) => l.split(/\s+/))
    .find((parts) => parts[1] === archiveName);
  if (!want) {
    fail(`No entry for ${archiveName} in SHASUMS256.txt — refusing to install untrusted binary.`);
  }
  const expected = want[0];
  const actual = sha256File(tmpArchive);
  if (expected.toLowerCase() !== actual.toLowerCase()) {
    fail(
      `SHA-256 mismatch for ${archiveName}:\n` +
      `  expected: ${expected}\n  actual:   ${actual}\n` +
      `Refusing to install. May indicate a tampered download or transient ` +
      `mirror corruption — retry, and if persistent file an issue at ` +
      `https://github.com/${REPO}/issues.`
    );
  }
  info("SHA-256 verified");

  fs.rmSync(NATIVE_DIR, { recursive: true, force: true });
  fs.mkdirSync(NATIVE_DIR, { recursive: true });
  try {
    if (ext === "tar.gz") {
      await extractTarGz(tmpArchive, NATIVE_DIR);
    } else {
      extractZip(tmpArchive, NATIVE_DIR);
    }
  } finally {
    try { fs.unlinkSync(tmpArchive); } catch (_) {}
  }

  // Sanity-check that every advertised binary actually landed.
  const suffix = target.includes("windows") ? ".exe" : "";
  for (const b of BINS) {
    const p = path.join(NATIVE_DIR, b + suffix);
    if (!fs.existsSync(p)) {
      fail(`Archive did not contain expected binary: ${b}${suffix}`);
    }
    if (!suffix) fs.chmodSync(p, 0o755);
  }
  info(`installed ${BINS.length} binaries to ${NATIVE_DIR}`);
})().catch((e) => fail(e.message || String(e)));
