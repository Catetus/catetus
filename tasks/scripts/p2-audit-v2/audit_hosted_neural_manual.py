"""One-shot audit invocation for hosted-neural after its encode landed.

The first hosted-neural run completed encode (744s wall) and the output
tar landed at the URL captured below. The local driver was killed before
it could complete its (now-broken) Modal arg-size upload path; this
script bypasses that and invokes audit_pair directly with the tar URL +
decoded_is_tar=True so the decoded.ply is extracted inside the GPU
container.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import modal

HF_BICYCLE = (
    "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/"
    "bicycle/point_cloud/iteration_7000/point_cloud.ply"
)
TAR_URL = (
    "https://xmcqr5nqjygbqjqw.public.blob.vercel-storage.com/jobs/"
    "p2audit-hn-06e2813d/hosted_neural-RFqce62ots0cUgTrVHJ89fCNf17zPj.tar"
)
OUT = Path(__file__).parent / "results" / "hosted-neural_bicycle.json"

audit_pair = modal.Function.from_name("p2-audit-psnr-v2", "audit_pair")
result = audit_pair.remote(
    label="hosted-neural",
    scene="bicycle_iter7000",
    source_url=HF_BICYCLE,
    decoded_url=TAR_URL,
    decoded_is_tar=True,
    decoded_member="repacked.ply",
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
print(f"wrote {OUT}")
if "error" in result:
    print(f"ERROR: {result['error']}", file=sys.stderr)
    sys.exit(1)
print(f"content_mean={result.get('content_mean_psnr_db')} "
      f"content_min={result.get('content_min_psnr_db')}")
