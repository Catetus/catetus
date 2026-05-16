#!/usr/bin/env bash
# Run on the 4090 (inside WSL Ubuntu): fetch sf-154 branch, build, run LODGE bench.
set -eu
cd /mnt/c/Users/monta/SplatForge
echo "=== git fetch + checkout ==="
git fetch origin --quiet
git checkout research/154-wgpu-stage6-buffer-split 2>&1 | tail -3
git pull --ff-only --quiet
echo "=== HEAD ==="
git log -1 --oneline
echo "=== pnpm install ==="
pnpm install --silent --filter @splatforge/viewer 2>&1 | tail -5 || true
echo "=== build viewer ==="
pnpm --filter @splatforge/viewer build 2>&1 | tail -10
echo "=== run LODGE bench (Sweet Corals) ==="
export SF_BENCH_PLY_DIR="/mnt/c/Users/monta/SplatForge/.bench-scenes"
export SF_BENCH_LODGE_ONLY=1
node packages/viewer/scripts/run-bench-windows.mjs 2>&1 | tail -300
echo "=== results-4090-lodge.json (or results.json) ==="
ls -la packages/viewer/bench/results*.json 2>/dev/null | head -10
node -e "
const fs=require('fs');
const path=require('path');
const candidates=['packages/viewer/bench/results-4090-lodge.json','packages/viewer/bench/results.json'];
for (const c of candidates) {
  if (fs.existsSync(c)) {
    const j=JSON.parse(fs.readFileSync(c,'utf8'));
    const s = j.realSceneLodge ?? j;
    console.log('===', c, '===');
    console.log(JSON.stringify(s, null, 2).slice(0, 5000));
  }
}
" 2>&1 | tail -200
