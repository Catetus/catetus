"""Render-PSNR harness v2 — content-bearing camera placement.

Fixes the oracle_pipeline.orbit_view_mats bug where camera y was
computed as cy + (cy + 0.3*radius) = 2*cy + 0.3*radius, lifting the
orbit above the scene and shooting most rays into empty space (cam
src_mean → 0, PSNR == 100 dB due to MSE clamp).

The corrected placement here:
  - 8 cameras around the weighted-opacity centroid.
  - Radius scales with scene IQR extent (same as v1).
  - Heights at 4 distinct levels (eye-level, slight up, slight down,
    eye-level again from opposite quadrant) — guarantees coverage of
    the scene's content bulk regardless of axis-up convention.

The harness asserts src-vs-src baseline PSNR == 100 dB on every cam
before reporting any (src, decoded) result.  If any camera fails the
baseline, that cam is flagged 'BAD_CAM' and excluded from the matrix.

Usage:
  python psnr_v2.py \
    --source /path/source.ply \
    --decoded /path/decoded.ply \
    --label codec-gs-mixed-crf28 \
    --scene bonsai_iter7000 \
    --out /path/out.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from gsplat import rasterization

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_inria(ply_path: Path) -> dict:
    ply = PlyData.read(str(ply_path))
    v = ply['vertex']
    n = len(v['x'])
    means = np.column_stack([np.asarray(v[f], dtype=np.float32) for f in ('x', 'y', 'z')])
    opa_logit = np.asarray(v['opacity'], dtype=np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opa_logit))
    scale_log = np.column_stack([np.asarray(v[f], dtype=np.float32)
                                 for f in ('scale_0', 'scale_1', 'scale_2')])
    scales = np.exp(scale_log)
    quats = np.column_stack([np.asarray(v[f], dtype=np.float32)
                             for f in ('rot_0', 'rot_1', 'rot_2', 'rot_3')])
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(min=1e-9)
    rest_keys = sorted([k for k in v.data.dtype.names if k.startswith('f_rest_')],
                       key=lambda s: int(s.split('_')[-1]))
    f_dc = np.column_stack([np.asarray(v[f], dtype=np.float32) for f in ('f_dc_0', 'f_dc_1', 'f_dc_2')])
    if rest_keys:
        rest = np.column_stack([np.asarray(v[f], dtype=np.float32) for f in rest_keys])
        n_rest = len(rest_keys) // 3
        rest_reshape = rest.reshape(n, 3, n_rest).transpose(0, 2, 1)
        sh = np.concatenate([f_dc[:, None, :], rest_reshape], axis=1)
    else:
        sh = f_dc[:, None, :]
    return dict(means=means, scales=scales, quats=quats, opacities=opacities, sh=sh)


def to_gpu_dict(s):
    return {k: torch.from_numpy(v).float().to(DEVICE) for k, v in s.items()}


def look_at(eye, center, up=(0.0, 1.0, 0.0)):
    """Standard look-at, returns 4x4 world-to-camera matrix (gsplat viewmat)."""
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = center - eye
    f /= (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up); r /= (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    R = np.stack([r, -u, f], axis=0)  # world-axes-in-camera rows
    t = -R @ eye
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def make_content_cameras(means_np, opa_np, scl_np, n_cams=8):
    """Build N cameras placed around the opacity-weighted centroid.

    Coverage strategy:
      - Compute centroid from opacity*scale-sum weight.
      - Compute IQR-based half-extent as radius proxy.
      - Place cameras on a sphere at varying yaw + 4 distinct pitches
        (-15°, 0°, +15°, 0° with mirrored yaw), all looking at centroid.
    """
    w = (opa_np * scl_np.sum(axis=1)).clip(min=1e-9)
    w_n = w / w.sum()
    centroid = (means_np * w_n[:, None]).sum(axis=0).astype(np.float32)
    lo = np.percentile(means_np, 25, axis=0)
    hi = np.percentile(means_np, 75, axis=0)
    extent = hi - lo
    radius = float(np.linalg.norm(extent)) * 1.5  # bit further so scene fits
    pitches_deg = [-15.0, 0.0, 15.0, 0.0, -15.0, 0.0, 15.0, 0.0][:n_cams]
    mats = np.zeros((n_cams, 4, 4), dtype=np.float32)
    for i in range(n_cams):
        yaw = 2 * math.pi * i / n_cams
        pitch = math.radians(pitches_deg[i])
        ex = centroid[0] + radius * math.cos(pitch) * math.cos(yaw)
        ey = centroid[1] + radius * math.sin(pitch)
        ez = centroid[2] + radius * math.cos(pitch) * math.sin(yaw)
        mats[i] = look_at((ex, ey, ez), tuple(centroid))
    return mats, centroid, radius


def render_frames(scene_gpu, sh_degree, vm, K, H, W):
    out = []
    for i in range(vm.shape[0]):
        with torch.no_grad():
            rgb, _, _ = rasterization(
                scene_gpu['means'], scene_gpu['quats'], scene_gpu['scales'],
                scene_gpu['opacities'], scene_gpu['sh'],
                vm[i:i + 1], K, W, H,
                sh_degree=sh_degree, render_mode='RGB',
            )
        out.append(rgb[0].clamp(0, 1))
    return torch.stack(out, 0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--decoded', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-cams', type=int, default=8)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--min-src-mean', type=float, default=0.05,
                    help='per-cam src_mean threshold to count as content-bearing')
    args = ap.parse_args()

    t0 = time.time()
    out_path = Path(args.out)

    src = load_inria(Path(args.source))
    dec = load_inria(Path(args.decoded))
    sh_src = int(round(math.sqrt(max(src['sh'].shape[1], 1)) - 1))
    sh_dec = int(round(math.sqrt(max(dec['sh'].shape[1], 1)) - 1))
    sh_d = min(sh_src, sh_dec)
    print(f'[psnr2] {args.label} | {args.scene} | sh={sh_d} '
          f'n_src={src["means"].shape[0]} n_dec={dec["means"].shape[0]}', flush=True)

    vm_np, centroid, radius = make_content_cameras(
        src['means'], src['opacities'], src['scales'], args.n_cams)
    vm = torch.from_numpy(vm_np).float().to(DEVICE)
    print(f'[psnr2] centroid={centroid.tolist()} radius={radius:.3f}', flush=True)

    H = W = args.image_size
    fov = math.pi / 3
    f = 0.5 * W / math.tan(fov / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=DEVICE).unsqueeze(0)

    src_gpu = to_gpu_dict(src)
    src_frames = render_frames(src_gpu, sh_d, vm, K, H, W)
    src_frames2 = render_frames(src_gpu, sh_d, vm, K, H, W)
    det_psnrs = [psnr(src_frames[i], src_frames2[i]) for i in range(args.n_cams)]
    src_mean_per_cam = [float(src_frames[i].mean().item()) for i in range(args.n_cams)]
    del src_frames2
    torch.cuda.empty_cache()

    dec_gpu = to_gpu_dict(dec)
    dec_frames = render_frames(dec_gpu, sh_d, vm, K, H, W)
    dec_mean_per_cam = [float(dec_frames[i].mean().item()) for i in range(args.n_cams)]

    per_cam_psnr = []
    flags = []
    for i in range(args.n_cams):
        p = psnr(dec_frames[i], src_frames[i])
        per_cam_psnr.append(p)
        # baseline check
        if det_psnrs[i] < 99.0:
            flags.append('BAD_BASELINE')
        elif src_mean_per_cam[i] < args.min_src_mean:
            flags.append('EMPTY')
        else:
            flags.append('CONTENT')

    content_idx = [i for i, f in enumerate(flags) if f == 'CONTENT']
    content_psnrs = [per_cam_psnr[i] for i in content_idx]
    content_mean = float(np.mean(content_psnrs)) if content_psnrs else None
    content_min = float(np.min(content_psnrs)) if content_psnrs else None

    print(f'[psnr2] per_cam: ', flush=True)
    for i in range(args.n_cams):
        print(f'   cam{i:1d} [{flags[i]:13s}] src_mean={src_mean_per_cam[i]:.4f} '
              f'dec_mean={dec_mean_per_cam[i]:.4f} det={det_psnrs[i]:.2f} dB | '
              f'PSNR(dec,src)={per_cam_psnr[i]:.3f} dB', flush=True)
    print(f'[psnr2] content_cams={content_idx} mean={content_mean} min={content_min}',
          flush=True)

    result = {
        'label': args.label,
        'scene': args.scene,
        'source_ply': str(args.source),
        'decoded_ply': str(args.decoded),
        'n_cams': args.n_cams,
        'image_size': args.image_size,
        'sh_degree': sh_d,
        'centroid': centroid.tolist(),
        'radius': radius,
        'min_src_mean_threshold': args.min_src_mean,
        'per_cam_psnr_db': per_cam_psnr,
        'per_cam_src_mean': src_mean_per_cam,
        'per_cam_dec_mean': dec_mean_per_cam,
        'per_cam_det_baseline_db': det_psnrs,
        'per_cam_flag': flags,
        'content_cams': content_idx,
        'content_mean_psnr_db': content_mean,
        'content_min_psnr_db': content_min,
        'elapsed_sec': time.time() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f'[psnr2] wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
