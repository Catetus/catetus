// Shared exec shim used by every entry under bin/. Resolves the native
// binary placed by install.js into ../native/<name>{.exe}, then spawnSyncs
// it and propagates the exit code. Keeping this in a single file means the
// per-bin shims are 2 lines each.
"use strict";

const path = require("path");
const fs = require("fs");
const { spawnSync } = require("child_process");

function nativePath(name) {
  const suffix = process.platform === "win32" ? ".exe" : "";
  return path.join(__dirname, "native", name + suffix);
}

function fail(name, msg) {
  process.stderr.write(
    `\n[@splatforge/cli] cannot run '${name}': ${msg}\n` +
    `\nIf install was skipped (SPLATFORGE_SKIP_DOWNLOAD=1) or failed, ` +
    `re-run\n  npm install -g @splatforge/cli --force\n\n`
  );
  process.exit(127);
}

exports.run = function run(name) {
  const bin = nativePath(name);
  if (!fs.existsSync(bin)) {
    fail(name, `binary not found at ${bin}`);
  }
  // spawnSync (not execFileSync) so the parent node process exits with the
  // *child's* exit code, including signals. `stdio: 'inherit'` keeps
  // stdin/out/err transparent for piping.
  const r = spawnSync(bin, process.argv.slice(2), {
    stdio: "inherit",
    windowsHide: false,
  });
  if (r.error) fail(name, r.error.message);
  if (r.signal) {
    // Forward signal-exits as 128 + signum so shells see the right code.
    const sigs = { SIGINT: 2, SIGTERM: 15, SIGHUP: 1, SIGQUIT: 3 };
    process.exit(128 + (sigs[r.signal] || 0));
  }
  process.exit(r.status == null ? 1 : r.status);
};
