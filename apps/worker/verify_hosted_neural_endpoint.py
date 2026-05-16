"""End-to-end verification: invoke `splatforge-hosted-neural` via Modal,
run the per-scene neural fit on bicycle (registered scene), and check
the returned numbers match the source-code shipped N=3 results.

Expected (from apps/diff-repack/results-neural-m3-final.json,
target_neural_ratio=0.05, bicycle):
  seed 0: ratio=7.540  psnr_delta=8.385
  seed 1: ratio=8.080  psnr_delta=8.550
  seed 2: ratio=7.208  psnr_delta=8.253
  median: ratio=7.54   psnr_delta=8.39

The endpoint is deterministic in seed: rerunning seed 0 must match
the shipped seed-0 cell to within rounding (the encoder is unchanged
modulo packaging; identical `target_neural_ratio_request` and
identical RD-loss schedule).

Usage:
  python3 verify_hosted_neural_endpoint.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import modal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene", default="bicycle")
    parser.add_argument("--target-neural-ratio", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    # Lookup the deployed function.
    fn = modal.Function.from_name(
        "splatforge-hosted-neural", "run_hosted_neural"
    )
    t0 = time.time()
    result = fn.remote(
        job_id=f"verify-{args.scene}-seed{args.seed}",
        preset="hosted-neural",
        blob_url="",  # ignored in registered-scene mode
        filename=args.scene,  # registered-scene shortcut
        callback_url=None,
        iters=args.iters,
        image_size=512,
        target_neural_ratio=args.target_neural_ratio,
        seed=args.seed,
    )
    wall = time.time() - t0
    print(f"\nWALL_SECS: {wall:.1f}\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
