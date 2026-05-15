#!/usr/bin/env bash
# End-to-end smoke test for the hosted splatforge API.
#
# Creates a URL-mode job against a known-public splat, polls until terminal,
# and **fetches the resulting .glb to confirm it parses** — per the
# artifact-pipelines lesson, HTTP-200 at every hop does not mean the file
# is usable. Exits non-zero on any failure.
#
# Usage:
#   API_URL=https://splatforge-api.fly.dev API_KEY=... ./smoke.sh
#   API_URL=http://127.0.0.1:8080 ./smoke.sh                  # local, no key
#   API_URL=... API_KEY=... PAID=1 PAID_API_KEY=... ./smoke.sh # also test /repack

set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8080}"
API_KEY="${API_KEY:-}"
PAID="${PAID:-0}"
PAID_API_KEY="${PAID_API_KEY:-$API_KEY}"
PRESET="${PRESET:-web-mobile}"
# 17 MB bonsai sample on HuggingFace — small enough to round-trip in <30s on
# the CPU optimizer, big enough to exercise the whole pipeline (PLY parse,
# all optimization passes, glTF/GLB packaging, Vercel Blob upload).
SOURCE_URL="${SOURCE_URL:-https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/iteration_7000/point_cloud.ply}"
TIMEOUT_SECS="${TIMEOUT_SECS:-300}"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }
die()   { red "FAIL: $*"; exit 1; }

auth_header() {
    [[ -n "$API_KEY" ]] && echo "-H" "Authorization: Bearer $API_KEY"
}

bold "→ POST $API_URL/v1/jobs (preset=$PRESET, URL-mode)"
create_response=$(curl -sSf -X POST "$API_URL/v1/jobs" \
    -H "Content-Type: application/json" \
    $(auth_header) \
    -d "{\"preset\":\"$PRESET\",\"source_url\":\"$SOURCE_URL\",\"label\":\"smoke\"}") \
    || die "job creation request failed"
echo "$create_response"

JOB_ID=$(echo "$create_response" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
[[ -n "$JOB_ID" ]] || die "could not parse job id from response"

bold "→ polling /v1/jobs/$JOB_ID until terminal (max ${TIMEOUT_SECS}s)"
deadline=$(( $(date +%s) + TIMEOUT_SECS ))
status=""
output_url=""
while (( $(date +%s) < deadline )); do
    poll=$(curl -sSf "$API_URL/v1/jobs/$JOB_ID" $(auth_header)) || die "poll failed"
    status=$(echo "$poll" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
    case "$status" in
        done)
            output_url=$(echo "$poll" | sed -n 's/.*"output_url":"\([^"]*\)".*/\1/p')
            green "✓ status=done; output_url=$output_url"
            break
            ;;
        error)
            err=$(echo "$poll" | sed -n 's/.*"error":"\([^"]*\)".*/\1/p')
            die "job failed: $err"
            ;;
        *)
            printf '  status=%s\n' "${status:-pending}"
            ;;
    esac
    sleep 5
done
[[ "$status" == "done" ]] || die "timed out waiting for terminal status (last=$status)"
[[ -n "$output_url" ]]    || die "done state but no output_url"

bold "→ fetching $output_url and validating .glb header"
tmp=$(mktemp -t splatforge-smoke.XXXXXX.glb)
trap 'rm -f "$tmp"' EXIT
curl -sSfL "$output_url" -o "$tmp" || die "output fetch failed"
size=$(wc -c < "$tmp" | tr -d ' ')
[[ "$size" -gt 0 ]] || die "output file is empty"
# GLB magic: 0x46546C67 = "glTF" little-endian in the first 4 bytes.
magic=$(head -c 4 "$tmp" | xxd -p)
[[ "$magic" == "676c5446" ]] || die "output is not a GLB (magic=$magic, expected 676c5446)"
green "✓ valid GLB header ($size bytes)"

if [[ "$PAID" == "1" ]]; then
    bold "→ POST $API_URL/v1/jobs/$JOB_ID/repack (paid tier, A100 differentiable)"
    target=$(( size / 2 ))  # 50% byte budget — bonsai reference point
    repack_response=$(curl -sSf -X POST "$API_URL/v1/jobs/$JOB_ID/repack" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $PAID_API_KEY" \
        -d "{\"target_bytes\":$target,\"iterations\":1000}") \
        || die "repack request failed"
    echo "$repack_response"

    bold "→ polling for repack completion"
    deadline=$(( $(date +%s) + TIMEOUT_SECS * 2 ))  # repack is longer
    while (( $(date +%s) < deadline )); do
        poll=$(curl -sSf "$API_URL/v1/jobs/$JOB_ID" $(auth_header))
        status=$(echo "$poll" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
        tier=$(echo "$poll" | sed -n 's/.*"tier":"\([^"]*\)".*/\1/p')
        if [[ "$status" == "done" && "$tier" == "paid" ]]; then
            new_url=$(echo "$poll" | sed -n 's/.*"output_url":"\([^"]*\)".*/\1/p')
            green "✓ repack done; output_url=$new_url"
            curl -sSfL "$new_url" -o "$tmp"
            new_size=$(wc -c < "$tmp" | tr -d ' ')
            [[ "$new_size" -le "$target" ]] || die "repack output ($new_size) exceeds target ($target)"
            green "✓ repack output $new_size bytes ≤ target $target"
            break
        fi
        if [[ "$status" == "error" ]]; then
            err=$(echo "$poll" | sed -n 's/.*"error":"\([^"]*\)".*/\1/p')
            die "repack failed: $err"
        fi
        printf '  status=%s tier=%s\n' "${status:-?}" "${tier:-?}"
        sleep 10
    done
fi

bold "✅ smoke test passed"
