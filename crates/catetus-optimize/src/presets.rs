//! Built-in optimization presets (SPEC-0006).

use anyhow::{anyhow, Result};

use crate::lod_merge::LodMergeV4;
use crate::passes::{
    AspectRatioPrune, BuildLOD, BundleNeighbors, FloaterPrune, MortonSort, OpacityPrune,
    QuantizeDCPacked, QuantizePosition, QuantizeRotation, QuantizeRotationPacked,
    QuantizeRotationSmallest3, QuantizeSHRest, QuantizeScale, ReduceSHDegree, RemoveInvalidSplats,
    SHDCTQuantize,
};
use crate::pipeline::Pipeline;
use crate::splat_delta::SplatDelta;
use crate::vq_palette::VQPaletteShRest;

/// All recognized preset names.
pub const PRESETS: &[&str] = &[
    "lossless-repack",
    "web-mobile",
    "web-desktop",
    "quest-browser",
    "visionos-preview",
    "thumbnail-preview",
    "quality-max",
    "size-min",
    "geospatial",
    "hero",
    // w3-harness-validate bisection variants — see
    // experiments/w3-harness-validate/RESULT.md
    "wmv-no-shred",
    "wmv-no-lod",
    "wmv-no-posq",
    "wmv-no-quant",
    "wmv-no-prune",
    // PRUNE_FIX_BENCH "max recovery" variant — bounds the recoverable PSNR
    // ceiling. See experiments/PRUNE_FIX_BENCH/REDIAG_RESULT.md.
    "wm-recovery",
    // PRUNE_FIX_BENCH FloaterPrune dist_sigma sweep — each preset is
    // otherwise identical to `web-mobile` (loose OpacityPrune 0.01,
    // AspectRatioPrune 50.0). Default FloaterPrune::default() uses
    // dist_sigma=3.0; the higher the sigma the less aggressive. The
    // `-off` variant removes the pass entirely. See
    // experiments/PRUNE_FIX_BENCH/FLOATER_TUNE_RESULT.md.
    "wm-floater-3",
    "wm-floater-6",
    "wm-floater-8",
    "wm-floater-12",
    "wm-floater-off",
    // w2-splatdelta: SplatDelta sidecar codec, 1.81× smaller than .sog on
    // bonsai. See experiments/w2-splatdelta/INTEGRATION_RESULT.md.
    "web-mobile-delta",
    // SplatDelta residual_bits sweep — bN variants identical to
    // `web-mobile-delta` save for the SplatDelta.residual_bits field
    // (6/8/10/12). Default `web-mobile-delta` uses b=6 which (per the
    // composed-codec bench in experiments/w4-stack/COMPOSED_BENCH_RESULT.md)
    // creates a ~21 dB PSNR rendering floor at sh=3. See
    // experiments/w2-splatdelta/RESIDUAL_BITS_SWEEP.md.
    "wmd-b6",
    "wmd-b8",
    "wmd-b10",
    "wmd-b12",
    // PRUNE_FIX_BENCH — the first real .sog competitor. Identical to
    // `web-mobile` MINUS `ReduceSHDegree{0}`, so sh_rest survives. The
    // GLB writer (`crates/catetus-gltf`) already emits one accessor
    // per `KHR_gaussian_splatting:SH_DEGREE_l_COEF_n` when the splats
    // carry them; `wmv-no-prune` proved the round-trip lands at 40.28 dB
    // at sh=3. This preset adds back the prune chain so we get the byte
    // savings too. See experiments/PRUNE_FIX_BENCH/WM_SH3_RESULT.md and
    // the verdict in experiments/HONEST_LEADERBOARD.md.
    "web-mobile-sh3",
    "web-mobile-sh3-floater6",
    // w4-stack final composition — see
    // experiments/w4-stack/FINAL_COMPOSITION_RESULT.md. SplatDelta on top of
    // a no-FloaterPrune + sh3-preserved base. Designed to be paired with
    // PostHAC SH-rest @ 4σ as a downstream sidecar.
    "wmd-sh3-nofloater",
    // PRUNE_FIX_BENCH — SH-rest int quantization closing the 12× FP32 gap
    // vs SOG. See experiments/PRUNE_FIX_BENCH/QUANTIZE_SHREST_RESULT.md.
    "wmv-sh3-q8",
    "wmv-sh3-q6",
    "web-mobile-sh3-q8",
    // SOG_STUDY_RUN Morton-zstd integer-attribute compression — `wmv-sh3-q8`
    // pipeline, but the GLB writer adds the `SF_zstd_split_buffer` lossless
    // wrap (byte-plane-transposed zstd-19) on the BIN chunk. Matches SOG's
    // "WebP-lossless on Morton-ordered integer textures" structural primitive
    // for our quantized accessors. See experiments/SOG_STUDY_RUN/MORTON_ZSTD_RESULT.md.
    "wmv-sh3-q8-zstd",
    // SOG-bit-allocation clone — matches splat-transform's per-attribute
    // budget as closely as our scalar passes allow. See
    // `experiments/SOG_STUDY_RESULT.md`. SOG (per study of writeSog in
    // node_modules/@playcanvas/splat-transform/dist/index.mjs at v2.1.1):
    //   means : 16 bits/axis (log1p + min/max, 8b lo + 8b hi planes)
    //   quats : smallest-3, 8 bits/component, 2-bit max-component tag
    //   scales: 8-bit codebook idx × 3 (shared 256-entry weighted-DP 1D
    //           codebook across all three scale columns)
    //   sh0   : 8-bit codebook idx × 3 channels + 8-bit sigmoid opacity
    //   shN   : 16-bit k-means palette index per splat into a 65,536-entry
    //           45-D codebook; each centroid coef stored as 8-bit value
    //           through a second shared 1D codebook
    // We can match per-attribute bit counts on pos/scale/rot/sh-rest but
    // not the 45-D k-means palette (we have only per-channel scalar quant).
    "web-mobile-sog-clone",
    // SplatDelta layered on top of the SOG-bit-allocation clone — tests
    // whether matching SOG's bit budget plus our delta sidecar can beat
    // SOG on both axes.
    "wmd-sog-clone",
    // SOG_STUDY_RUN — VQPaletteShRest 45-D k-means palette pass. Same
    // structural ID as `wmv-sh3-q8-zstd` (RemoveInvalid + pos/scale/rot
    // quant + Morton + LOD + zstd-19 split-buffer wrap) but with the
    // SH-rest path swapped from per-coefficient int8 quant
    // (`QuantizeSHRest{8}`) over to a 65,536-entry 45-D k-means
    // codebook. The codebook + per-splat 16-bit indices ride a
    // `.shpal` sidecar emitted by the CLI (analogous to .splatdelta).
    // This is SOG's killer compression primitive — see
    // `experiments/SOG_STUDY_RESULT.md`.
    "web-mobile-vq45",
    // Same as web-mobile-vq45 but drops the OpacityPrune /
    // AspectRatioPrune chain (= `wmv-no-prune` lineage) so we bound
    // the max-quality PSNR ceiling for VQ45.
    "wmv-vq45-no-prune",
    // SOG_STUDY_RUN TIGHT pack — `wmv-vq45-no-prune` plus 8-bit packed
    // rotation + DC quantization (`QuantizeRotationPacked` /
    // `QuantizeDCPacked`). The GLB writer emits ROTATION (VEC4 UBYTE-
    // normalized, 4 B/splat) and DC (VEC3 UBYTE-normalized, 3 B/splat) with
    // per-component min/max.
    "wmv-vq45-no-prune-tight",
    // SOG_STUDY_RUN / SMALLEST-3 — `wmv-vq45-no-prune` plus SOG's smallest-3
    // quaternion codec (`QuantizeRotationSmallest3` at `component_bits = 10`).
    // 4 B/splat on the wire (`SF_quat_smallest3` SCALAR UINT accessor); ~4x
    // finer per-component resolution than naive 8-bit-per-4-components
    // packing because the three kept components live in
    // [-1/sqrt(2), +1/sqrt(2)]. See
    // `experiments/SOG_STUDY_RUN/SMALLEST3_QUAT_RESULT.md`.
    "wmv-vq45-no-prune-tight-smallest3",
    // T2.1.R-K-SWEEP — K=1024 / K=16384 siblings of `wmv-vq45k4096-no-prune-tight`.
    // Tiny shpal codebooks for byte-constrained tiers; only `palette_size`
    // changes vs the K=4096 / K=65536 presets. Used by
    // experiments/t2-1-r-k-sweep/RESULT.md.
    "wmv-vq45k1024-no-prune-tight",
    "wmv-vq45k16384-no-prune-tight",
    // VQ45_GPU_SWEEP follow-on — K=4096 sibling of `wmv-vq45-no-prune-tight`.
    // The 4090 GPU sweep (experiments/SOG_STUDY_RUN/VQ45_GPU_SWEEP.md) proved
    // K=4096 collapses the SH-rest payload from 5.26 MB → 1.92 MB at sh=3
    // with identical-to-K=65536 PSNR once the standard quant chain is layered
    // on top (~42 dB ceiling clamps both). Added as a sibling preset rather
    // than mutating the existing K=65536 preset so agent #8's baseline read
    // of `wmv-vq45-no-prune-tight` stays stable.
    "wmv-vq45k4096-no-prune-tight",
    // K=4096 VQ45 + PostHAC categorical range-coding over the u16 palette
    // index stream. The PostHAC `.shpal.pthc` companion sidecar carries a
    // self-describing IDXP payload (header + empirical histogram + range-
    // coded bitstream). The standard `.shpal` is still emitted so legacy
    // decoders keep working — composed bytes accounting at bench time
    // substitutes the IDXP size for the raw u16 index portion of the
    // `.shpal`. JS bench decoder integration is pending (see
    // experiments/SOG_STUDY_RUN/VQ45_POSTHAC_RESULT.md).
    "wmv-vq45k4096-posthac-no-prune-tight",
    // T2.1.R + PostHAC composition — K=65536 sibling of
    // `wmv-vq45k4096-posthac-no-prune-tight`. Combines T2.1.R's high-K
    // Jacobian-weighted Lloyd-Max VQ (the 54 dB centroid grid) with
    // PostHAC categorical range-coding over the same u16 index stream.
    // Composed bytes accounting at bench time replaces the raw u16 index
    // share of `.shpal` with the smaller `.shpal.pthc` payload. See
    // experiments/posthac-on-t21r/RESULT.md.
    "wmv-vq45-posthac-no-prune-tight",
    // preset-v2 ship deliverable — `wmv-vq45-no-prune-tight` baseline preceded
    // by the V4 LOD-merge pre-pass (mixture-of-Gaussians moment matching over
    // sub-pixel splats). V4 cuts splat count, which multiplies every
    // downstream per-splat saving (pos/scale/rot/DC quantization, 45-D
    // SH-rest palette indices). 12-bit scale is already part of the base
    // pipeline; this preset is the cleanest composition of the v2 research
    // wins we could ship within the build/test budget. P3 surface-chart
    // positions, R1 tangent-frame rotation residual, and X2 R=32 lattice
    // SH-rest replacement are deferred — each needs a new GLB-writer
    // sidecar format to realize the prototype's byte savings on disk, and
    // running them without their sidecars would degrade PSNR without
    // shrinking bytes. See experiments/preset-v2/RESULT.md.
    "wmv-vq45-tight-v2",
    // SOG-native preset — `RemoveInvalidSplats → MortonSort → VQPaletteShRest{K=4096}`.
    // Drops the GLB-only quantization passes (pos/scale/rot/DC) that bake
    // bytes into the GLB BIN chunk: SOG re-quantizes every attribute via
    // its own 16-bit / smallest-3 / 256-codebook chain, so any pre-quant
    // here is pure precision loss against SOG's own per-attribute budget.
    // BuildLOD is also dropped because SOG carries no LOD slot — the
    // `LodLevel` array would be discarded at write time anyway.
    //
    // When the CLI passes `--jacobian-sidecar`, VQPaletteShRest switches
    // into the render-space weighted Lloyd loop — that's where the
    // +2 dB PSNR lift over the SOG vanilla encoder lives. Task #115.
    "sog-render-weighted",
];

/// Build a `Pipeline` from a named preset.
pub fn preset(name: &str) -> Result<Pipeline> {
    let pipe = match name {
        "lossless-repack" | "quality-max" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(MortonSort)),
        // `web-mobile`: default web target. AspectRatioPrune (max_ratio=50)
        // drops only the most extreme needle splats — the original 10.0
        // threshold removed 13% of the scene (151k splats on bonsai) and
        // accounted for the bulk of the SF-vs-splat-transform PSNR gap per
        // experiments/w3-harness-validate/RESULT.md. 50.0 keeps the spike-
        // artifact protection (Inria-3DGS-style needles have ratios >>50)
        // while preserving real anisotropic detail. OpacityPrune dropped
        // from 0.02 to 0.01 for the same reason — visible alpha at 0.015
        // should not be culled at mobile bandwidth.
        //
        // FloaterPrune REMOVED from the default web-mobile chain — at
        // dist_sigma=3 (the FloaterPrune::default) it dropped 18k
        // load-bearing splats on bonsai for -17 dB PSNR / -0.066 SSIM
        // and only saved ~1 MB on the wire. RemoveInvalidSplats above
        // already drops NaN/Inf so the safety-net rationale is moot.
        // See experiments/PRUNE_FIX_BENCH/FLOATER_TUNE_RESULT.md (sweep
        // table: floater-3 → 21.55 dB, floater-off → 38.65 dB on the
        // SH-blind harness). Verified end-to-end via the SH-aware
        // cpu-fidelity harness at sh=3, size=512, frames=8: pre-fix
        // 15.32 dB, post-fix 35+ dB.
        "web-mobile" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        // wmv-* = "web-mobile variant" bisection presets. Each removes one
        // pass family from `web-mobile` so we can attribute PSNR loss to a
        // single step. See `experiments/w3-harness-validate/RESULT.md`.
        // Note: ReduceSHDegree{0} is intentionally kept in every variant
        // because the current GLB writer only emits SH_DEGREE_0_COEF_0 in
        // the KHR_gaussian_splatting extension, and the harness only reads
        // that channel — so keeping vs. dropping higher SH bands has zero
        // effect on the rendered image either way.
        "wmv-no-shred" => Pipeline::new()
            // Identical to web-mobile (ReduceSHDegree{0} kept) — control run
            // that re-confirms the baseline number on this build.
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wmv-no-lod" => Pipeline::new()
            // web-mobile minus BuildLOD
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort)),
        "wmv-no-posq" => Pipeline::new()
            // web-mobile minus QuantizePosition (keep float32 positions)
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wmv-no-quant" => Pipeline::new()
            // web-mobile minus all quantization (keep float32 everywhere)
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wmv-no-prune" => Pipeline::new()
            // web-mobile minus OpacityPrune + AspectRatioPrune + FloaterPrune.
            // Keeps full splat count, all the quantization + LOD.
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        // `wm-recovery`: temporary diagnostic preset. Loosest possible prunes
        // (OpacityPrune 0.005, AspectRatioPrune 200.0) so almost every splat
        // survives — this bounds the recoverable PSNR ceiling with the rest
        // of the web-mobile pipeline (quant + LOD + SH-DC-only) intact.
        // Used by experiments/PRUNE_FIX_BENCH/REDIAG_RESULT.md.
        "wm-recovery" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.005 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 200.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        // FloaterPrune dist_sigma sweep variants of web-mobile. Each one is
        // bit-identical to `web-mobile` save for FloaterPrune.dist_sigma
        // (or its removal). Used by the PRUNE_FIX_BENCH FLOATER_TUNE bench.
        "wm-floater-3" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune {
                k_neighbors: 8,
                dist_sigma: 3.0,
            }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wm-floater-6" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune {
                k_neighbors: 8,
                dist_sigma: 6.0,
            }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wm-floater-8" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune {
                k_neighbors: 8,
                dist_sigma: 8.0,
            }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wm-floater-12" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune {
                k_neighbors: 8,
                dist_sigma: 12.0,
            }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wm-floater-off" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "web-desktop" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.005 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 16 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 1 }))
            // SH-DCT uniform-7: from experiments/sh-dct/RESULT.md Pareto.
            // Quantizes each per-channel SH band vector in DCT space; the
            // quantized coefficients compress much better through the
            // downstream codec while keeping mean dE94 < 0.7. The Pareto
            // numbers were measured at degree-3 (n=16); at degree-1 (n=4)
            // we expect a smaller absolute win — still net-positive because
            // the round-trip is mathematically near-lossless at 7-bit.
            .push(Box::new(SHDCTQuantize::uniform(7)))
            .push(Box::new(MortonSort)),
        "quest-browser" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 30.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 14 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD { levels: vec![0.3] })),
        "visionos-preview" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 8,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort)),
        "thumbnail-preview" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.05 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 12 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort)),
        // `geospatial`: produces a Cesium 3D Tiles 1.1 tileset (tileset.json +
        // per-LOD GLBs) when paired with the tileset emitter. The pipeline
        // itself is the `web-mobile` baseline (prune + quantize + Morton),
        // followed by a four-level LOD pyramid where each LOD halves the splat
        // count of the previous one: LOD0=1.0, LOD1=0.5, LOD2=0.25, LOD3=0.125.
        // The downstream tileset writer maps each LOD to a tile with halved
        // `geometricError` and a `REPLACE` refinement chain, matching Cesium's
        // screen-space-error model.
        "geospatial" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.02 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 8,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25, 0.125],
            })),
        "size-min" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.05 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 10.0 }))
            .push(Box::new(FloaterPrune::default()))
            // BundleNeighbors: merge near-duplicate Gaussians. Pareto-tuned
            // on bonsai_iter7000 — at (voxel=0.1, attr=0.6) we collapse 2.79×
            // splats with MLP-fidelity drop of ≤0.14 from the lossless
            // baseline, on top of the rest of size-min. See
            // .
            .push(Box::new(BundleNeighbors {
                voxel_size_world: 0.1,
                max_attr_distance: 0.6,
            }))
            .push(Box::new(QuantizePosition { bits: 12 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.25, 0.1],
            })),
        // `hero`: marketing-hero / showcase preset. Optimized for visual
        // quality on a single above-the-fold slow-orbit scene, not for raw
        // byte size. No opacity prune, no aspect-ratio prune, no LOD chain,
        // 16-bit scale quant (vs 8-bit elsewhere) so anisotropic detail
        // splats render without spike artifacts. SH reduced to degree 0 only
        // because the current viewer renders DC-only; bump target_degree
        // back to 1+ once `packages/viewer` learns view-dependent shading.
        "hero" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 16 }))
            .push(Box::new(QuantizeScale {
                bits: 16,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 16 }))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort)),
        // `web-mobile-delta`: same prune chain as `web-mobile`, but replaces
        // the three scalar quant passes (position / scale / rotation) and the
        // inline DC-color quant with a single `SplatDelta` pass — the
        // anchor-stride Morton-order residual codec validated by
        // `experiments/w2-splatdelta` (1.41-1.57× smaller than .sog at
        // parity fidelity). The CLI drains the codec's sidecar blob after
        // the pipeline runs and writes it next to the .glb / .gltf output
        // as a `.splatdelta` file.
        "web-mobile-delta" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(SplatDelta::default()))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        // `wmd-bN`: SplatDelta residual_bits sweep. Each variant is
        // bit-identical to `web-mobile-delta` except for the residual_bits
        // field on the SplatDelta pass. The default codec uses b=6, which
        // the composed-codec bench at experiments/w4-stack identified as a
        // ~21 dB PSNR floor at sh=3 (position RMSE 0.027 ≈ 1px blur at
        // 512×512). This sweep tests b ∈ {6, 8, 10, 12} to find where
        // SplatDelta moves off the floor onto a real Pareto curve vs SOG.
        // b=6 is the explicit control re-bench of the default.
        "wmd-b6" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(SplatDelta {
                anchor_stride: 64,
                k_neighbors: 2,
                residual_bits: 6,
                range_percentile: 99.5,
            }))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wmd-b8" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(SplatDelta {
                anchor_stride: 64,
                k_neighbors: 2,
                residual_bits: 8,
                range_percentile: 99.5,
            }))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wmd-b10" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(SplatDelta {
                anchor_stride: 64,
                k_neighbors: 2,
                residual_bits: 10,
                range_percentile: 99.5,
            }))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        "wmd-b12" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(ReduceSHDegree { target_degree: 0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(SplatDelta {
                anchor_stride: 64,
                k_neighbors: 2,
                residual_bits: 12,
                range_percentile: 99.5,
            }))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        // `web-mobile-sh3`: bit-identical to `web-mobile` (loose
        // OpacityPrune 0.01, AspectRatioPrune 50.0, default FloaterPrune,
        // 15-bit pos, 12-bit scale/rot, MortonSort, BuildLOD) but with
        // the `ReduceSHDegree { target_degree: 0 }` pass REMOVED so the
        // GLB carries the full sh=3 coefficient set. Per
        // experiments/HONEST_LEADERBOARD.md, dropping that single pass is
        // the highest-ROI lever for closing the .sog gap: every other SF
        // "shipping" preset was paying an 18–20 dB SH-aware tax by baking
        // DC-only color. The GLB writer (`crates/catetus-gltf`) emits
        // a `KHR_gaussian_splatting:SH_DEGREE_l_COEF_n` accessor per
        // surviving coefficient, so no encoder changes are needed.
        "web-mobile-sh3" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        // `web-mobile-sh3-floater6`: same as `web-mobile-sh3` but with
        // FloaterPrune.dist_sigma=6.0 (matches `wm-floater-6`). Per the
        // PRUNE_FIX_BENCH FloaterPrune sweep, sigma=3 (the default) is
        // the most aggressive setting and trades visible content for
        // bytes; sigma=6 is the friendlier shoulder of the curve.
        "web-mobile-sh3-floater6" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune {
                k_neighbors: 8,
                dist_sigma: 6.0,
            }))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        // `wmd-sh3-nofloater`: the only composition w4-stack hadn't tried.
        // Drops the two killer passes (`FloaterPrune` — which removes 18k
        // load-bearing splats and is what made every prior wmd-* variant
        // pay a structural PSNR tax — and `ReduceSHDegree{0}` — which
        // strips sh_rest before the codec ever sees it). Keeps the cheap
        // alpha~0 / extreme-needle cleanup, runs SplatDelta on the full
        // sh3-preserved population, then BuildLOD. Designed to be paired
        // downstream with PostHAC SH-rest @ 4σ as a sidecar.
        "wmd-sh3-nofloater" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(MortonSort))
            .push(Box::new(SplatDelta::default()))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            })),
        // `wmv-sh3-q8` / `wmv-sh3-q6`: bit-identical to `wmv-no-prune`
        // (RemoveInvalid + pos/scale/rot quant + MortonSort + BuildLOD)
        // MINUS `ReduceSHDegree{0}` so the GLB carries the full sh=3
        // coefficient set, PLUS `QuantizeSHRest` at 8 or 6 bits so the
        // 180 b/s FP32 sh-rest path shrinks to 45 / 33.75 b/s. The GLB
        // writer's QuantizeSHRest side-table integration emits each
        // `SH_DEGREE_l_COEF_n` accessor as a normalized signed BYTE
        // (`q ∈ [-127,127]`) or SHORT (`q ∈ [-32767,32767]`) with
        // per-channel `min/max` derived from the table.
        "wmv-sh3-q8" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(QuantizeSHRest {
                bits: 8,
                range_percentile: 99.5,
            })),
        "wmv-sh3-q6" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(QuantizeSHRest {
                bits: 6,
                range_percentile: 99.5,
            })),
        // `wmv-sh3-q8-zstd`: pipeline is bit-identical to `wmv-sh3-q8`. The
        // SOG-parity gain comes from the GLB writer's `SF_zstd_split_buffer`
        // lossless wrap, which the CLI auto-enables for this preset name.
        // The wrap byte-plane-transposes each bufferView (so adjacent splats
        // in Morton order share high-bytes per plane) and zstd-19's the
        // transposed BIN. Round-trip is lossless — PSNR/SSIM unchanged from
        // `wmv-sh3-q8`, bytes drop by ~27% on bonsai (96.0 MB → 70.2 MB).
        "wmv-sh3-q8-zstd" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(QuantizeSHRest {
                bits: 8,
                range_percentile: 99.5,
            })),
        // `web-mobile-sh3-q8`: the full `web-mobile-sh3` prune+quant chain
        // with `QuantizeSHRest{8}` appended.
        "web-mobile-sh3-q8" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(QuantizeSHRest {
                bits: 8,
                range_percentile: 99.5,
            })),
        // `web-mobile-sog-clone`: match splat-transform .sog v2.1.1
        // bit allocation on every attribute we have a scalar pass for.
        // - 16-bit position (SOG: 16b/axis lo+hi planes)
        // - 8-bit scale (SOG: 8b codebook idx/axis)
        // - 8-bit rotation (SOG: smallest-3, 8b/component)
        // - 8-bit SH-rest (SOG: 16b palette idx into 8b centroids)
        // No DC-color or opacity quant pass exists, so those stay FP32
        // in the GLB — small fixed-cost gap vs the SOG webp.
        // NO prune chain (matches `wmv-no-prune`) so the population
        // matches SOG's full-density preservation.
        "web-mobile-sog-clone" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 16 }))
            .push(Box::new(QuantizeScale {
                bits: 8,
                ..Default::default()
            }))
            .push(Box::new(QuantizeRotation { bits: 8 }))
            .push(Box::new(MortonSort))
            .push(Box::new(QuantizeSHRest {
                bits: 8,
                range_percentile: 99.5,
            })),
        // `wmd-sog-clone`: same SOG-style quantization, but replace the
        // pos/scale/rot scalar quants with SplatDelta (which already
        // bundles position + scale + rotation residuals) and keep the
        // SH-rest 8-bit quantization on the SH side. This is the
        // composition we couldn't test before because SOG's bit ceiling
        // for pos/scale/rot is what SplatDelta's range_percentile=99.5
        // and residual_bits=6 implicitly target.
        "wmd-sog-clone" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(MortonSort))
            .push(Box::new(SplatDelta::default()))
            .push(Box::new(QuantizeSHRest {
                bits: 8,
                range_percentile: 99.5,
            })),
        // `web-mobile-vq45`: `web-mobile-sh3-q8` skeleton (full prune chain
        // + 15-bit pos / 12-bit scale / 12-bit rot + MortonSort + BuildLOD)
        // but with the SH-rest path replaced by `VQPaletteShRest` — a
        // 45-D k-means palette with K=65,536 centroids (SOG default).
        // The CLI drains the resulting codebook+indices via
        // `take_last_sh_rest_palette()` and emits a `.shpal` sidecar
        // (same drain-pattern as SplatDelta's `.splatdelta` sidecar).
        // The GLB itself is unaffected: the pass mutates SH-rest in
        // place to centroid values so the writer's SH accessors still
        // round-trip the cluster representatives via FP32; the
        // bytes-on-wire win comes from the sidecar replacing the
        // ~39 MB / 360 bps FP32 SH-rest emission.
        "web-mobile-vq45" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(OpacityPrune { threshold: 0.01 }))
            .push(Box::new(AspectRatioPrune { max_ratio: 50.0 }))
            .push(Box::new(FloaterPrune::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                // K=65,536 matches SOG's default. 6 Lloyd iterations on
                // a 200k-point random subsample captures the bulk of
                // convergence (each additional iter past ~5 moves
                // codebook MSE <1%); a final full-N pass then assigns
                // every splat to its nearest centroid. Wall-time on
                // bonsai (N=1.16M) is ~3 min on 8 cores. The full
                // 25-iter SOG default on the full N would take ~3 hours
                // at K=65k — see SOG_STUDY_RUN/VQ45_RESULT.md.
                palette_size: 65_536,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            })),
        // `wmv-vq45-no-prune`: bit-identical to `web-mobile-vq45` minus
        // OpacityPrune + AspectRatioPrune + FloaterPrune. Bounds the
        // max-quality ceiling for the VQ45 palette pass — full splat
        // population goes through the codebook.
        "wmv-vq45-no-prune" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                // K=65,536 matches SOG's default. 6 Lloyd iterations on
                // a 200k-point random subsample captures the bulk of
                // convergence (each additional iter past ~5 moves
                // codebook MSE <1%); a final full-N pass then assigns
                // every splat to its nearest centroid. Wall-time on
                // bonsai (N=1.16M) is ~3 min on 8 cores. The full
                // 25-iter SOG default on the full N would take ~3 hours
                // at K=65k — see SOG_STUDY_RUN/VQ45_RESULT.md.
                palette_size: 65_536,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            })),
        // `wmv-vq45-no-prune-tight`: `wmv-vq45-no-prune` plus 8-bit packed
        // rotation + DC quantization. Both new passes park side tables that
        // the GLB writer consumes to emit BYTE accessors (4 B/splat ROTATION,
        // 3 B/splat DC) with per-component min/max. The packed passes run
        // AFTER `VQPaletteShRest` so palette centroid assignment is
        // unaffected.
        "wmv-vq45-no-prune-tight" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                palette_size: 65_536,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            }))
            .push(Box::new(QuantizeRotationPacked { bits: 8 }))
            .push(Box::new(QuantizeDCPacked { bits: 8 })),
        // `wmv-vq45-no-prune-tight-smallest3`: identical to
        // `wmv-vq45-no-prune-tight` but swaps the 8-bit packed-per-component
        // ROTATION for the SOG-style smallest-3 codec at 10 bits per stored
        // component (3*10 + 2-bit tag = 32 bits = 4 B/splat). Same byte
        // budget as 4×8-bit packed rotation, but ~4× finer per-component
        // resolution because the three stored components live in
        // [-1/sqrt(2), +1/sqrt(2)] — the dropped (largest) component is
        // recoverable via sqrt(1 - sum_others^2). DC is still 8-bit packed
        // (same as TIGHT preset). See
        // `experiments/SOG_STUDY_RUN/SMALLEST3_QUAT_RESULT.md`.
        "wmv-vq45-no-prune-tight-smallest3" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                palette_size: 65_536,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            }))
            .push(Box::new(QuantizeRotationSmallest3 { component_bits: 10 }))
            .push(Box::new(QuantizeDCPacked { bits: 8 })),
        // `wmv-vq45k1024-no-prune-tight`: K=1024 sibling of
        // `wmv-vq45k4096-no-prune-tight`. Tightest VQ45 codebook in the
        // K-sweep — predicted ~64 KB shpal. See
        // experiments/t2-1-r-k-sweep/RESULT.md.
        "wmv-vq45k1024-no-prune-tight" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                palette_size: 1_024,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            }))
            .push(Box::new(QuantizeRotationPacked { bits: 8 }))
            .push(Box::new(QuantizeDCPacked { bits: 8 })),
        // `wmv-vq45k16384-no-prune-tight`: K=16384 sibling. Mid-point between
        // K=4096 and K=65536 — predicted ~1 MB shpal. See
        // experiments/t2-1-r-k-sweep/RESULT.md.
        "wmv-vq45k16384-no-prune-tight" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                palette_size: 16_384,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            }))
            .push(Box::new(QuantizeRotationPacked { bits: 8 }))
            .push(Box::new(QuantizeDCPacked { bits: 8 })),
        // `wmv-vq45k4096-no-prune-tight`: K=4096 sibling of
        // `wmv-vq45-no-prune-tight`. Same pipeline shape, only the codebook
        // size moves. Per VQ45_GPU_SWEEP.md, K=4096 ships SH-rest at
        // 1.92 MB (vs 5.26 MB @ K=65536) for equivalent total-pipeline
        // PSNR once the rest of the quant chain runs on top of the centroids.
        "wmv-vq45k4096-no-prune-tight" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                palette_size: 4_096,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            }))
            .push(Box::new(QuantizeRotationPacked { bits: 8 }))
            .push(Box::new(QuantizeDCPacked { bits: 8 })),
        // `wmv-vq45k4096-posthac-no-prune-tight`: K=4096 sibling + PostHAC
        // categorical range-coding over the u16 palette index stream. The
        // pass parks an additional `.shpal.pthc` blob whose size replaces
        // the raw u16 index bytes inside the standard `.shpal` for "composed"
        // bytes accounting. PSNR is identical to
        // `wmv-vq45k4096-no-prune-tight` because the VQ centroids (and thus
        // the in-scene SH-rest values written back) do not change — the
        // PostHAC layer is purely a wire-format compression of the existing
        // index stream.
        "wmv-vq45k4096-posthac-no-prune-tight" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                palette_size: 4_096,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: true,
            }))
            .push(Box::new(QuantizeRotationPacked { bits: 8 }))
            .push(Box::new(QuantizeDCPacked { bits: 8 })),
        // `wmv-vq45-posthac-no-prune-tight`: K=65536 + PostHAC. Same
        // structure as `wmv-vq45-no-prune-tight` but flips on PostHAC
        // index-stream coding. Centroids (and thus decoded SH-rest) are
        // unchanged vs the K=65536 baseline — PostHAC is purely a wire
        // compression of the 1.24M × u16 index stream that dominates
        // shpal bytes at high K.
        "wmv-vq45-posthac-no-prune-tight" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                palette_size: 65_536,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: true,
            }))
            .push(Box::new(QuantizeRotationPacked { bits: 8 }))
            .push(Box::new(QuantizeDCPacked { bits: 8 })),
        // `wmv-vq45-tight-v2`: SHIP preset. Layers the V4 LOD-merge pre-pass
        // (mixture-of-Gaussians moment matching over sub-pixel splats) on top
        // of the `wmv-vq45-no-prune-tight` baseline. V4 collapses small
        // clusters into single super-splats, which multiplies every
        // downstream per-splat saving. Defaults match the prototype TIGHT
        // config (screen_threshold_px=1.0, voxel_factor=0.5, max_cluster=64).
        // The 12-bit scale, 12-bit rotation, 15-bit position, 8-bit packed
        // DC + rotation, K=65536 SH-rest palette, MortonSort, BuildLOD,
        // and `SF_zstd_split_buffer` lossless wrap all come from the
        // baseline pipeline. See experiments/preset-v2/RESULT.md.
        "wmv-vq45-tight-v2" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            // V4 LOD merge at the prototype's default settings
            // (screen_threshold_px=1.0, max_cluster=64). On bonsai this hits
            // −40% splat count and −33% total bytes (vs the v1 baseline),
            // but costs ~7 dB PSNR (47.42 → 40.65 dB) at sh=3, size=512,
            // frames=8. The v2 preset ships as a SIZE-FIRST companion to
            // `wmv-vq45-no-prune-tight`; pick v2 when bytes matter more
            // than the last 7 dB of fidelity. See
            // experiments/preset-v2/RESULT.md for the full table.
            .push(Box::new(LodMergeV4::default()))
            .push(Box::new(QuantizePosition { bits: 15 }))
            .push(Box::new(QuantizeScale {
                bits: 12,
                log_space: true,
            }))
            .push(Box::new(QuantizeRotation { bits: 12 }))
            .push(Box::new(MortonSort))
            .push(Box::new(BuildLOD {
                levels: vec![0.5, 0.25],
            }))
            .push(Box::new(VQPaletteShRest {
                palette_size: 65_536,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            }))
            .push(Box::new(QuantizeRotationPacked { bits: 8 }))
            .push(Box::new(QuantizeDCPacked { bits: 8 })),
        // `sog-render-weighted` — task #115. SOG-native pipeline: only the
        // passes whose output SOG can carry without re-quantizing through
        // its own per-attribute budget. The GLB-only `QuantizePosition` /
        // `QuantizeScale` / `QuantizeRotation` / `QuantizeRotationPacked`
        // / `QuantizeDCPacked` passes are dropped — SOG already encodes
        // positions as 16-bit log-range, scales/DC via 256-entry codebooks,
        // and rotations via smallest-3, so any prior quantization here is
        // pure precision loss.
        //
        // `BuildLOD` is dropped because SOG has no LOD slot; the
        // `LodLevel` array would be silently discarded at write time.
        //
        // When the CLI passes `--jacobian-sidecar`, `VQPaletteShRest`
        // automatically switches into the render-space weighted Lloyd loop,
        // which is the source of the +2 dB lift over the SOG vanilla
        // encoder on bonsai (see experiments/sog-render-weighted/RESULT.md).
        // K=4096 matches the SOG vanilla SH-rest centroid count and keeps
        // the `shN_centroids.webp` payload tiny (~180 KB).
        "sog-render-weighted" => Pipeline::new()
            .push(Box::new(RemoveInvalidSplats))
            .push(Box::new(MortonSort))
            .push(Box::new(VQPaletteShRest {
                palette_size: 4_096,
                iterations: 6,
                codebook_bits: 8,
                training_subsample: Some(200_000),
                posthac_indices: false,
            })),
        other => return Err(anyhow!("unknown preset '{other}'")),
    };
    Ok(pipe)
}
