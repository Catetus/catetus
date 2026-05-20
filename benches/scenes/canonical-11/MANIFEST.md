# Canonical-11 Pretrained PLY Manifest

Source: Inria 3DGS pretrained release.
Archive: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip> (14.66 GB)
Path inside zip: `<scene>/point_cloud/iteration_30000/point_cloud.ply`
Date staged: 2026-05-17

All checkpoints below are the official 30k-iteration Gaussian Splatting outputs
released by the paper authors. Source images / COLMAP outputs are NOT staged
locally (handled by 4090 agent #51). The local pretrained PLYs live under
`benches/scenes/canonical-11/pretrained/<scene>.ply` and are git-ignored.

## Per-scene status

| Scene     | Status     | Size (bytes)  | Splats      | MD5                              |
|-----------|------------|---------------|-------------|----------------------------------|
| bicycle   | downloaded | 1520726124    | 6,131,954   | 3745b7a650432cde504bd1ae7d14b130 |
| bonsai    | downloaded | 308716644     | 1,244,819   | ad5377ebb1845c7eaa64fa7712fca49a |
| counter   | downloaded | 303294620     | 1,222,956   | 10acbca9de3fd0944541796c7cb1d6ea |
| garden    | downloaded | 1447027964    | 5,834,784   | 06b7ecb60d0cc6c43ef72306ab62dfb5 |
| kitchen   | downloaded | 459380612     | 1,852,335   | ed2a44a58b54988f9044e566aa5943e2 |
| room      | downloaded | 395158780     | 1,593,376   | 4e6f7d0bf8b063d301ff8ecf0db6a0bd |
| stump     | downloaded | 1230527188    | 4,961,797   | e7d7f3d131f42128a4edf2ca84fd32b0 |
| truck     | downloaded | 630225580     | 2,541,226   | 801ad447156a6c0cf0c5f456db21252d |
| train     | downloaded | 254575516     | 1,026,508   | 7a84ee9d6958614c836f959e80d743b2 |
| drjohnson | downloaded | 844479476     | 3,405,153   | 1942e23826d52d99907edb1ef67df607 |
| playroom  | downloaded | 631438300     | 2,546,116   | 4419e224a2903161b4b1fe7209282ce9 |

Validation: each PLY parses as binary little-endian, has `element vertex` >
100k, size > 50 MB. All 11 scenes match the canonical Mip-NeRF 360 / Tanks &
Temples / Deep Blending set used by the paper.

Bonus scenes also present in `models.zip` but not part of the canonical-11
benchmark and not extracted: `flowers`, `treehill`.

## MD5 cross-check (dupe guard)

All 11 MD5 hashes are pairwise distinct — verified by sorting hashes and
grouping. NO duplicate files in the staged set.

For reference, the pre-existing staged file at
`tasks/bench-input/inria_3dgs_iter30k.ply` (size 286,972,500 bytes,
md5 `5d616d87252e26372106ac50e36ce3b1`) does NOT match any of the canonical
PLYs above. Per the task note ("the prior bench-input was secretly a dupe of
bonsai"), expected bonsai md5 is `ad5377eb...` — the legacy file's hash and
size do not match bonsai (294.4 MB vs 273.7 MB). The legacy file is therefore
a different artifact, probably a re-saved or re-quantized derivative of one of
the canonical scenes. Bench scripts that reference
`tasks/bench-input/inria_3dgs_iter30k.ply` should be migrated to the
appropriate canonical PLY in this directory.

## NEEDS_TRAINING

None — Inria's pretrained zip carries 30k-iter checkpoints for all 11
canonical scenes.

## Reproduce

```
mkdir -p benches/scenes/canonical-11/pretrained
cd benches/scenes/canonical-11/pretrained
curl -L -o models.zip \
  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
python3 - <<'PY'
import zipfile, shutil
scenes = ['bicycle','bonsai','counter','garden','kitchen','room','stump',
          'truck','train','drjohnson','playroom']
z = zipfile.ZipFile('models.zip')
for s in scenes:
    src = f'{s}/point_cloud/iteration_30000/point_cloud.ply'
    with z.open(src) as fin, open(f'{s}.ply','wb') as fout:
        shutil.copyfileobj(fin, fout, 16*1024*1024)
PY
rm models.zip
```
