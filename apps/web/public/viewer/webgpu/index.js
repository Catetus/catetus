// SPDX-License-Identifier: Apache-2.0
/**
 * GPU compute-decode + radix-sort pipeline for `@catetus/viewer`.
 *
 * Replaces the CPU decode + JS sort path used by {@link WebGPURenderer}
 * when the viewer is constructed with `useComputeDecode: true`.
 *
 * Pipeline (all in WGSL, all on the GPU):
 *
 *   raw chunk bytes                 (storage<read>)
 *           │
 *           ▼  cs_decode  — SoA → canonical DecodedSplat (position, scale,
 *           │              rotation, opacity, colorDC). 1 thread per splat.
 *           ▼
 *      decoded splats               (storage<read_write>)
 *           │
 *           ▼  cs_project — Canonical splats + camera → per-instance vertex
 *           │              attributes (clipPos, 2D covariance, color) and
 *           │              depth-sort keys + indirection indices.
 *           ▼
 *      instance buffer (unsorted)   (storage)
 *      keys + indices                (storage)
 *           │
 *           ▼  radix sort (8×4-bit passes, three kernels per pass).
 *           ▼
 *      sorted indices                (storage)
 *           │
 *           ▼  cs_gather — write the final instance buffer in sorted order.
 *           ▼
 *      sorted instance buffer        (vertex)
 *
 * The final buffer is bound as a vertex buffer to the existing render
 * pipeline in `renderer/webgpu.ts` — the rasterizer stays untouched.
 *
 * Determinism: every step is data-parallel without atomic order-dependence on
 * storage buffers. The only atomics are workgroup-shared counters (mandatory
 * in WebGPU 1.0). Equal-depth splats fall back to splat-index by virtue of
 * the radix sort packing the splat index into the value buffer; ties in the
 * key resolve in scatter order, which is deterministic for a fixed dispatch.
 */
import { DECODE_WGSL, RADIX_SORT_WGSL, PROJECT_GATHER_WGSL, SCAN_MULTIBLOCK_WGSL, RADIX_MERGE_WGSL, } from './shaders.generated.js';
import { BufferPager, templateSplatsAccess } from './buffer-pager.js';
import { createRadixSortPipelines, RadixSort } from './radix_sort.js';
import { createCullPipelines, CullPipeline } from './cull.js';
import { WSRPipeline, WSR_DEFAULT_BG_WEIGHT } from './wsr.js';
import { WSRTilePipeline } from './wsr_tile.js';
import { dispatchPerSplat } from './multi-dispatch.js';
/**
 * Byte offset of `chunk_offset: u32` inside each kernel's uniform struct.
 * Multi-dispatch wrappers rewrite only this slot between chunks.
 *
 * Keep in lock-step with the WGSL struct layouts. WebGPU std140-ish rules
 * apply: the offset is `<previous fields aligned to vec4 boundaries>`.
 */
export const UNIFORM_CHUNK_OFFSET_BYTES = {
    /** decode.wgsl::DecodeUniforms — chunk_offset at slot 1 (after splat_count). */
    decode: 4,
    /** decode.wgsl::ProjectUniforms (cs_project) and cs_project_gather.wgsl::ProjectUniforms.
     *  2×mat4(64) + viewport(8) + focal(8) + splat_count(4) = 148. */
    project: 148,
    /** Inline GATHER_WGSL::Uniforms — chunk_offset at slot 1 (after count). */
    gather: 4,
    /** cs_cull.wgsl::CullUniforms — 2×mat4 + viewport + focal + splat_count + tau = 152. */
    cull: 152,
    /** cs_cull.wgsl::CompactUniforms — chunk_offset at slot 1 (after splat_count). */
    compact: 4,
    /** cs_lod_blend.wgsl::LodBlendUniforms — chunk_offset at slot 3
     *  (after splat_count, chunk_count, force_passthrough). */
    lodBlend: 12,
    /** cs_lod_blend.wgsl::ResetUniforms — chunk_offset at slot 1 (after splat_count). */
    lodReset: 4,
    /** cs_tile_bin.wgsl::TileBinUniforms — 2×mat4(128) + viewport(8) + focal(8)
     *  + splat_count(4) + tile_size(4) + tiles_x(4) + tiles_y(4) + max_per_tile(4) = 164. */
    tileBin: 164,
};
/** Floats per per-instance render record. Mirrors `FLOATS_PER_INSTANCE` in webgpu.ts. */
export const FLOATS_PER_INSTANCE = 12;
/** SH-rest coefficient count per channel given a degree. Mirrors the
 *  loader-side `shRestCoefCount` so this file has no loader dependency. */
function shCoefCountForDegree(deg) {
    // 1 → 3 (l=1), 2 → 3+5 = 8 (l=1,2), 3 → 3+5+7 = 15 (l=1,2,3)
    let c = 0;
    if (deg >= 1)
        c += 3;
    if (deg >= 2)
        c += 5;
    if (deg >= 3)
        c += 7;
    return c;
}
/** Bytes per canonical decoded splat (4×vec4 = 64 bytes). Matches `DecodedSplat` in WGSL. */
export const BYTES_PER_DECODED_SPLAT = 64;
const FLOAT_CT = 5126;
const UBYTE_CT = 5121;
const USHORT_CT = 5123;
/* --------------------------------------------------------------------------- */
/* Gather shader (sorted indices → final instance buffer).                     */
/*                                                                             */
/* Kept tiny + inline because it doesn't need to live in its own .wgsl file.   */
/* --------------------------------------------------------------------------- */
const GATHER_WGSL = /* wgsl */ `
struct Instance {
  clip_pos: vec4<f32>,
  cov:      vec4<f32>,
  color:    vec4<f32>,
};
struct Uniforms { count: u32, chunk_offset: u32, _pad: vec2<u32> };
@group(0) @binding(0) var<storage, read>       src    : array<Instance>;
@group(0) @binding(1) var<storage, read>       order  : array<u32>;
@group(0) @binding(2) var<storage, read_write> dst    : array<Instance>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;
@compute @workgroup_size(256)
fn cs_gather(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + u.chunk_offset;
  if (i >= u.count) { return; }
  dst[i] = src[order[i]];
}
`;
/**
 * Replace the EWA-anti-aliasing 2D-covariance dilation floor `let reg = 0.3;`
 * (marked with the `SF_EWA_DILATION` token) with `let reg = <dilation>;` so
 * a caller can shrink or disable the screen-space sigma_min^2 = 0.3 floor
 * that's inherited from the Inria CUDA rasterizer. See research log entry
 * for 2026-05-15 (novel-2-renderer) for why this matters: trained scenes
 * have median sigma 0.55-1.3 px so the radius cull never fires, but dropping
 * the floor to ~0.05-0.1 px brings cull rate from <1% up to 10-30%+ on
 * already-trained PLYs with negligible visible-pixel impact.
 */
function applyDilationOverride(wgsl, dilation) {
    if (dilation === 0.3)
        return wgsl;
    // Render a clean WGSL literal: small but non-zero floor stays positive,
    // exact 0.0 emits "0.0" to keep WGSL happy with type inference.
    const lit = dilation === 0 ? '0.0' : dilation.toFixed(6);
    return wgsl.replace(/let reg = 0\.3; \/\/ SF_EWA_DILATION/g, `let reg = ${lit}; // SF_EWA_DILATION(override=${dilation})`);
}
function createDecodePipelines(device, includeFused, dilation = 0.3) {
    const decodeMod = device.createShaderModule({ code: applyDilationOverride(DECODE_WGSL, dilation) });
    const decodeBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const projectBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            // Phase 2b: SH-rest blob (read-only). Bound to a 16-byte dummy buffer
            // when the scene has no SH-rest data — the shader's sh_params.x == 0
            // path short-circuits the fetch.
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });
    const decode = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [decodeBgl] }),
        compute: { module: decodeMod, entryPoint: 'cs_decode' },
    });
    const project = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [projectBgl] }),
        compute: { module: decodeMod, entryPoint: 'cs_project' },
    });
    const gatherMod = device.createShaderModule({ code: GATHER_WGSL });
    const gatherBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const gather = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [gatherBgl] }),
        compute: { module: gatherMod, entryPoint: 'cs_gather' },
    });
    if (!includeFused) {
        return { decode, project, gather, decodeBgl, projectBgl, gatherBgl };
    }
    // -----------------------------------------------------------------
    // Fused project+gather pipeline. See cs_project_gather.wgsl.
    //
    //   cs_keygen:           splats + camera → keys[] + indices[].
    //                        Bindings: (0 splats, 1 keys, 2 indices, 3 uniforms).
    //   cs_project_gather:   splats + sorted_indices[] + camera → instanceBuffer[].
    //                        Bindings: (0 splats, 1 indices, 2 inst_out, 3 uniforms).
    // -----------------------------------------------------------------
    const fusedMod = device.createShaderModule({ code: applyDilationOverride(PROJECT_GATHER_WGSL, dilation) });
    const keygenBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const projectGatherBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            // Phase 2b: SH-rest blob. See the cs_project binding above for rationale.
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });
    const keygen = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [keygenBgl] }),
        compute: { module: fusedMod, entryPoint: 'cs_keygen' },
    });
    const projectGather = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [projectGatherBgl] }),
        compute: { module: fusedMod, entryPoint: 'cs_project_gather' },
    });
    return {
        decode, project, gather, decodeBgl, projectBgl, gatherBgl,
        keygen, projectGather, keygenBgl, projectGatherBgl,
    };
}
function componentTypeId(t) {
    // WGSL only knows about three component types. Default to f32 if absent.
    if (t === USHORT_CT)
        return USHORT_CT;
    if (t === UBYTE_CT)
        return UBYTE_CT;
    return FLOAT_CT;
}
/**
 * Pack the per-slice decode uniforms (5 attribute slices + count) into a
 * 4-byte-aligned buffer that matches the WGSL `DecodeUniforms` struct.
 */
function buildDecodeUniforms(device, layout, splatCount) {
    const SLICE_FLOATS = 12; // 4 ints + vec4 min + vec4 max
    const FLOATS = 4 + 5 * SLICE_FLOATS;
    const ab = new ArrayBuffer(FLOATS * 4);
    const i32 = new Int32Array(ab);
    const f32 = new Float32Array(ab);
    let o = 0;
    i32[o++] = splatCount;
    i32[o++] = 0;
    i32[o++] = 0;
    i32[o++] = 0;
    for (const slice of [layout.positions, layout.rotations, layout.scales, layout.opacities, layout.colorDC]) {
        i32[o++] = slice.byteOffset;
        i32[o++] = componentTypeId(slice.componentType);
        i32[o++] = slice.normalized ? 1 : 0;
        i32[o++] = 0; // _pad
        // vmin
        for (let k = 0; k < 4; k++)
            f32[o + k] = slice.min?.[k] ?? 0;
        o += 4;
        // vmax
        for (let k = 0; k < 4; k++)
            f32[o + k] = slice.max?.[k] ?? 0;
        o += 4;
    }
    const buf = device.createBuffer({
        size: ab.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, ab);
    return buf;
}
/**
 * Per-sub-range variant of buildDecodeUniforms. Same layout, but with
 * each SoA slice's byteOffset shifted by `srcSplatOffset` × the slice's
 * per-splat stride so the kernel's chunk_offset = 0 reads the sub-range.
 *
 * Stage 6 (sf-154): used by uploadChunk when a chunk straddles a
 * splats-page boundary; we split into per-page sub-dispatches and rebase
 * the SoA reads here.
 */
function buildDecodeUniformsForRange(device, layout, splatCount, srcSplatOffset) {
    // Per-slice stride: components × component-size.
    // positions: 3 components, rotations: 4, scales: 3, opacities: 1, colorDC: 3.
    // Component size: f32=4, u16=2, u8=1.
    const compSize = (t) => {
        if (t === USHORT_CT)
            return 2;
        if (t === UBYTE_CT)
            return 1;
        return 4;
    };
    const componentsForSlice = (idx) => {
        // 0:pos(3) 1:rot(4) 2:scale(3) 3:opacity(1) 4:colorDC(3)
        return [3, 4, 3, 1, 3][idx];
    };
    const rebased = {
        positions: { ...layout.positions },
        rotations: { ...layout.rotations },
        scales: { ...layout.scales },
        opacities: { ...layout.opacities },
        colorDC: { ...layout.colorDC },
    };
    const slices = [rebased.positions, rebased.rotations, rebased.scales, rebased.opacities, rebased.colorDC];
    for (let i = 0; i < slices.length; i++) {
        const stride = componentsForSlice(i) * compSize(slices[i].componentType);
        slices[i].byteOffset = slices[i].byteOffset + srcSplatOffset * stride;
    }
    return buildDecodeUniforms(device, rebased, splatCount);
}
/**
 * GPU compute-decode + radix-sort pipeline. Holds device-lifetime state
 * (compiled pipelines, sorter, transient buffers).
 *
 * Usage:
 *   1. Construct once after `device.requestDevice()`.
 *   2. For each chunk, call `uploadChunk(descriptor, bytes)`.
 *   3. Each frame, call `encode(encoder, view, viewProj, focal, viewport)`
 *      and bind the resulting `instanceBuffer` to the render pipeline.
 */
export class ComputeDecodePipeline {
    device;
    capacity;
    /** True when the fused project+gather path is active. */
    useFusedProject;
    /** EWA dilation floor applied to projected 2D covariance. See `ComputeDecodePipelineInit.dilation`. */
    dilation;
    pipes;
    radixPipes;
    /** Canonical decoded-splat buffer pager (Stage 6 / sf-154).
     *  When numPages == 1 this is functionally identical to the old single-buffer
     *  path. When numPages > 1 the fused project_gather path uses templated
     *  multi-page bindings; non-fused / cull / WSR paths are unsupported and
     *  throw at construction. */
    pager;
    /** Convenience: page 0's buffer. Used by single-page bind groups for the
     *  cull/WSR/non-fused paths and by uploadChunk's dstView. */
    get splatsBuffer() { return this.pager.pageBuffers[0]; }
    /** Unsorted instance buffer (project shader output). Only allocated in the non-fused path. */
    instUnsorted;
    /** Sorted final instance buffer. Used as the vertex buffer by the renderer. */
    instanceBuffer;
    /** Stage 6 (sf-154): instance buffer pages. Length 1 in the
     *  single-page path; > 1 when the total instance bytes exceed
     *  the per-binding cap. Each page covers a contiguous range of
     *  splat-output indices `[splatStart, splatStart + splatCount)`. */
    instancePages;
    /** Splats per instance page (last page may be shorter). */
    instanceSplatsPerPage;
    /** Radix-sort runner. `keysA`/`valuesA` are scratch we write into in cs_project. */
    sorter;
    /** Project pass uniform buffer (view + viewProj + viewport + focal + count). */
    projectUniforms;
    /** Bind group for cs_project (non-fused only). Phase 2b: mutable so
     *  uploadChunk can re-bind to the per-scene SH-rest buffer. */
    projectBindGroup;
    /** Gather pass uniform (count). Non-fused only. */
    gatherUniforms;
    gatherBindGroup;
    /** Bind groups for the fused path (keygen + project_gather). Phase 2b:
     *  project_gather bind group becomes mutable for SH-rest rebind. */
    keygenBindGroup;
    projectGatherBindGroup;
    /** Layouts kept around so Phase 2b can rebuild bind groups after the
     *  SH-rest buffer is replaced on first SH-bearing chunk. */
    _projectGatherBglSingle;
    _projectGatherBglPaged = null;
    /** Stage 6 (sf-154): one project_gather bind group per INSTANCE page.
     *  Empty for the cull / WSR / non-fused paths. */
    projectGatherBindGroups = [];
    /** Optional opacity-radius pre-sort cull. Allocated when useCull=true. */
    cull;
    /** True when useCull was requested at construction. */
    useCull;
    /** Optional WSR pipeline. Allocated when useWSR=true. */
    wsr;
    /** True when useWSR was requested at construction. */
    useWSR;
    /** Optional tile-prefix-sum WSR pipeline. Allocated when useWSRTile=true. */
    wsrTile;
    /** True when useWSRTile was requested at construction. */
    useWSRTile;
    /**
     * Scene-wide WSR depth-weight scale. Initialised host-side from the scene
     * bounding box (`2 × mean_scene_depth`); the renderer or the unit test
     * sets this before the first `encode()` call. PR1 default keeps the
     * heuristic in safe territory for sub-cube synthetic scenes.
     */
    /** Stage 6 paged-keygen pipeline (multi-page splats). Null when numPages==1. */
    _pagedKeygen = null;
    /** Stage 6 paged project_gather pipeline. Null when numPages==1. */
    _pagedProjectGather = null;
    /**
     * Phase 2b: per-splat SH-rest blob bound to cs_project / cs_project_gather.
     * Lazily allocated on first uploadChunk that carries a non-null
     * `attributeLayout.shRest`. When no chunk carries SH-rest, this stays as
     * the 16-byte dummy buffer created at construction so the shader's
     * read-only binding remains valid.
     */
    shRestBuffer;
    /** SH degree carried by `shRestBuffer` (0 / 1 / 2 / 3). */
    shDegree = 0;
    /** SH coefficient count (per channel) derived from `shDegree`. */
    shCoefCount = 0;
    /** Runtime ON/OFF toggle for the SH-rest evaluator. */
    shRestEnabled = true;
    wsrSigma = 2.0;
    /** WSR background color + weight `w_B`. PR1 default = (black, 1e-4). */
    wsrBgColor = [0, 0, 0, WSR_DEFAULT_BG_WEIGHT];
    chunks = [];
    /** Splats already decoded (offset into `splatsBuffer`). */
    decodedSplats = 0;
    constructor(init) {
        this.device = init.device;
        this.capacity = init.capacity;
        // PERFORMANCE NOTE: the fused `cs_project_gather` path (true) eliminates
        // a 640 MB unsorted-scratch pass at 10 M splats but pays a 2.3× cost in
        // per-frame GPU time on the laptop 4090 — projection is recomputed in
        // sorted order with cache-hostile reads back into `splats[]`, vs the
        // non-fused path which projects once into a coherent scratch buffer and
        // the per-frame gather is a 64-byte memcpy. We default to the non-fused
        // path (false) until the fuse is made bandwidth-efficient. Real per-stage
        // numbers in `docs/perf/webgpu-10m-profile.md`.
        this.useFusedProject = init.useFusedProject ?? false;
        this.dilation = init.dilation ?? 0.3;
        this.pipes = createDecodePipelines(this.device, this.useFusedProject, this.dilation);
        // Stage 5: pass the multi-block scan + merge WGSL so the sorter can
        // handle splat counts above the WebGPU 1.0 dispatch cap (16.7M). The
        // subgroup-histogram path stays opt-in (requires a 'subgroups'-enabled
        // device); we pass '' to keep the older atomic-add histogram.
        this.radixPipes = createRadixSortPipelines(this.device, RADIX_SORT_WGSL, SCAN_MULTIBLOCK_WGSL, '', RADIX_MERGE_WGSL);
        // Stage 6 (sf-154): split the canonical decoded-splat storage across N
        // GPUBuffers (each <= adapter.maxStorageBufferBindingSize). For
        // <= 33 M splats this is one page (identical layout to the old single-
        // buffer path). At LODGE L1 (~54 M, 3.5 GB) it's 2 pages; at L0
        // (~119 M, 7.6 GB) it's 4 pages at a 2 GiB cap.
        const lim = this.device.limits;
        const maxBufferBytes = Math.min(lim.maxStorageBufferBindingSize ?? (2 * 1024 * 1024 * 1024 - 1), lim.maxBufferSize ?? (2 * 1024 * 1024 * 1024 - 1));
        this.pager = new BufferPager(this.device, this.capacity, maxBufferBytes);
        if (this.pager.numPages > 1 && !this.useFusedProject) {
            throw new Error(`ComputeDecodePipeline: capacity ${this.capacity} requires ${this.pager.numPages} ` +
                `splat pages but useFusedProject=false (only the fused project_gather path supports multi-page splats — ` +
                `set useFusedProject=true or reduce capacity to <= ${this.pager.splatsPerPage}).`);
        }
        const INSTANCE_BYTES = FLOATS_PER_INSTANCE * 4; // 48 B / splat
        const instSize = Math.max(this.capacity * INSTANCE_BYTES, INSTANCE_BYTES);
        // The 640-MB-at-10M scratch buffer is only needed for the non-fused path.
        // In the fused path we write the final instance record directly from the
        // project_gather kernel, so we skip the allocation entirely.
        this.instUnsorted = this.useFusedProject
            ? null
            : this.device.createBuffer({ size: instSize, usage: GPUBufferUsage.STORAGE });
        // Stage 6 (sf-154): page the instance buffer the same way we page the
        // splats buffer. At 48 B/splat, 2 GiB caps us at ~44.7M splats per page;
        // L1 (54M) needs 2 pages, L0 (119M) needs 3.
        //
        // Each page is allocated VERTEX | STORAGE | COPY_SRC. The first page
        // serves as `instanceBuffer` for the renderer (single-page paths) and
        // backward compat; multi-page consumers must iterate `instancePages`.
        const instSplatsPerPage = Math.floor(maxBufferBytes / INSTANCE_BYTES);
        // Round down to a multiple of 256 to keep workgroup-aligned dispatches.
        const instSplatsPerPage256 = Math.floor(instSplatsPerPage / 256) * 256;
        if (instSplatsPerPage256 < 256) {
            throw new Error(`ComputeDecodePipeline: maxBufferBytes ${maxBufferBytes} too small for one instance workgroup`);
        }
        const numInstPages = Math.max(1, Math.ceil(this.capacity / instSplatsPerPage256));
        this.instancePages = [];
        for (let p = 0; p < numInstPages; p++) {
            const start = p * instSplatsPerPage256;
            const count = Math.min(instSplatsPerPage256, Math.max(this.capacity - start, 0));
            const byteSize = Math.max(count * INSTANCE_BYTES, INSTANCE_BYTES);
            const buf = this.device.createBuffer({
                size: byteSize,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            this.instancePages.push({ splatStart: start, splatCount: count, byteSize, buffer: buf });
        }
        this.instanceSplatsPerPage = instSplatsPerPage256;
        // Backward-compat: `instanceBuffer` is the first page. Single-page
        // paths see the same buffer they always saw; multi-page consumers
        // walk `instancePages` instead.
        this.instanceBuffer = this.instancePages[0].buffer;
        this.sorter = new RadixSort(this.device, this.capacity, this.radixPipes);
        // Project uniforms (Phase 2b): 2 mat4 (32 floats) + viewport vec2 + focal vec2
        // + (splat_count, chunk_offset, _pad2) = 40 floats + vec4 cam_pos (4 floats)
        // + vec4<u32> sh_params (4 ints) = 48 floats / 192 bytes total. Padding to
        // 192 keeps WebGPU's 16-byte uniform-stride alignment happy.
        this.projectUniforms = this.device.createBuffer({
            size: 4 * (16 + 16 + 2 + 2 + 4 + 4 + 4), // 192 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Phase 2b: dummy SH-rest buffer (16 B is the smallest a STORAGE binding
        // can be). Replaced by a real per-scene buffer the first time a chunk
        // with SH-rest data is uploaded. Holds a single vec4 of zeros — never
        // read in production because the shader short-circuits on shDegree=0.
        this.shRestBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // Phase 2b: project_gather BGL handle for SH-rest rebind. Set on the
        // fused single-page branch only; multi-page uses `_projectGatherBglPaged`.
        this._projectGatherBglSingle = this.useFusedProject && this.pipes.projectGatherBgl
            ? this.pipes.projectGatherBgl
            : null;
        if (!this.useFusedProject) {
            this.projectBindGroup = this.device.createBindGroup({
                layout: this.pipes.projectBgl,
                entries: [
                    { binding: 0, resource: { buffer: this.splatsBuffer } },
                    { binding: 1, resource: { buffer: this.instUnsorted } },
                    { binding: 2, resource: { buffer: this.sorter.keysA } },
                    { binding: 3, resource: { buffer: this.sorter.valuesA } },
                    { binding: 4, resource: { buffer: this.projectUniforms } },
                    // Phase 2b: SH-rest blob. Bound to the dummy buffer until a chunk
                    // uploads real coefficients (see uploadChunk).
                    { binding: 5, resource: { buffer: this.shRestBuffer } },
                ],
            });
            // 32 B minimum: WGSL Uniforms { count: u32, _pad: vec3<u32> } occupies
            // 16 B but WebGPU pads uniform-buffer bindings up to the next power-of-2
            // ≥ the struct size, with a 32 B minimum on most adapters.
            this.gatherUniforms = this.device.createBuffer({
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.gatherBindGroup = this.device.createBindGroup({
                layout: this.pipes.gatherBgl,
                entries: [
                    { binding: 0, resource: { buffer: this.instUnsorted } },
                    { binding: 1, resource: { buffer: this.sorter.valuesA } },
                    { binding: 2, resource: { buffer: this.instanceBuffer } },
                    { binding: 3, resource: { buffer: this.gatherUniforms } },
                ],
            });
            this.keygenBindGroup = null;
            this.projectGatherBindGroup = null;
        }
        else {
            this.projectBindGroup = null;
            this.gatherUniforms = null;
            this.gatherBindGroup = null;
            // Fused path: reuse the same projectUniforms buffer (matching struct).
            // When the splats are multi-page (Stage 6 / sf-154), rebuild the keygen
            // and project_gather pipelines from templated WGSL with N page bindings
            // and a `read_splats_*` switch helper.
            if (this.pager.numPages > 1) {
                const pagedPipes = this._buildPagedFusedPipelines(this.pager.numPages, this.pager.splatsPerPage);
                this._pagedKeygen = pagedPipes.keygen;
                this._pagedProjectGather = pagedPipes.projectGather;
                this._projectGatherBglPaged = pagedPipes.projectGatherBgl;
                // Build per-pipeline bind groups that bind ALL pages on bindings
                // [0, numPages), with downstream bindings rebased (see
                // templateSplatsAccess in buffer-pager.ts).
                const N = this.pager.numPages;
                const pageEntries = (downstream) => [
                    ...this.pager.pageBuffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
                    ...downstream.map((e) => ({ ...e, binding: e.binding + (N - 1) })),
                ];
                this.keygenBindGroup = this.device.createBindGroup({
                    layout: pagedPipes.keygenBgl,
                    entries: pageEntries([
                        { binding: 1, resource: { buffer: this.sorter.keysA } },
                        { binding: 2, resource: { buffer: this.sorter.valuesA } },
                        { binding: 3, resource: { buffer: this.projectUniforms } },
                    ]),
                });
                // Stage 6 (sf-154): build one project_gather bind group per
                // INSTANCE page so the kernel writes the correct slice. Indices
                // (4 B/splat → fits in 2 GiB even at 119M) and inst_out are
                // dynamic-offset-sliced; chunk_offset is set per dispatch in encode().
                this.projectGatherBindGroups = [];
                for (const ipage of this.instancePages) {
                    const idxOffset = ipage.splatStart * 4;
                    const idxSize = Math.max(ipage.splatCount * 4, 4);
                    void ipage.splatStart;
                    const outSize = Math.max(ipage.splatCount * INSTANCE_BYTES, INSTANCE_BYTES);
                    this.projectGatherBindGroups.push(this.device.createBindGroup({
                        layout: pagedPipes.projectGatherBgl,
                        entries: pageEntries([
                            { binding: 1, resource: { buffer: this.sorter.valuesA, offset: idxOffset, size: idxSize } },
                            { binding: 2, resource: { buffer: ipage.buffer, offset: 0, size: outSize } },
                            { binding: 3, resource: { buffer: this.projectUniforms } },
                            // Phase 2b: SH-rest. Same buffer for all pages — the kernel
                            // indexes by global splat index.
                            { binding: 4, resource: { buffer: this.shRestBuffer } },
                        ]),
                    }));
                }
                // Backward-compat: first page's bind group serves as the legacy
                // projectGatherBindGroup for code paths that still read it.
                this.projectGatherBindGroup = this.projectGatherBindGroups[0];
            }
            else {
                this.keygenBindGroup = this.device.createBindGroup({
                    layout: this.pipes.keygenBgl,
                    entries: [
                        { binding: 0, resource: { buffer: this.splatsBuffer } },
                        { binding: 1, resource: { buffer: this.sorter.keysA } },
                        { binding: 2, resource: { buffer: this.sorter.valuesA } },
                        { binding: 3, resource: { buffer: this.projectUniforms } },
                    ],
                });
                this.projectGatherBindGroup = this.device.createBindGroup({
                    layout: this.pipes.projectGatherBgl,
                    entries: [
                        { binding: 0, resource: { buffer: this.splatsBuffer } },
                        { binding: 1, resource: { buffer: this.sorter.valuesA } },
                        { binding: 2, resource: { buffer: this.instanceBuffer } },
                        { binding: 3, resource: { buffer: this.projectUniforms } },
                        // Phase 2b: SH-rest blob. See the non-fused projectBindGroup above.
                        { binding: 4, resource: { buffer: this.shRestBuffer } },
                    ],
                });
                this.projectGatherBindGroups = [this.projectGatherBindGroup];
            }
        }
        // -----------------------------------------------------------------
        // Opacity-radius pre-sort cull. Defaults to ON because production
        // orbit cameras (~0.7×bbox-diag, 60° FOV) cull 33–82% of splats and
        // give a 2.8× fps speedup on bicycle / 1.2× on bonsai (re-validated
        // 2026-05-15 via novel-2-renderer + R4-ADCA benches; see
        // docs/perf/webgpu-10m-profile.md). The earlier "cull collapses on
        // real captures" claim was a too-close-camera bench artifact.
        // Callers can still opt out by passing useCull: false.
        // -----------------------------------------------------------------
        this.useCull = init.useCull ?? true;
        if (this.useCull && this.pager.numPages > 1) {
            throw new Error(`ComputeDecodePipeline: useCull is not supported with multi-page splats ` +
                `(${this.pager.numPages} pages required for capacity ${this.capacity}). ` +
                `Pass useCull=false or reduce capacity to <= ${this.pager.splatsPerPage}.`);
        }
        if (this.useCull) {
            const cullPipes = createCullPipelines(this.device, this.dilation);
            // For the cull path we write Instance records straight into the
            // unsorted scratch buffer (non-fused only — fused-path integration
            // is a follow-up because the fused path skips the scratch buffer
            // entirely). Bench harness uses the non-fused path by default per
            // commit 354cd8e.
            if (this.useFusedProject) {
                throw new Error('useCull is only supported on the non-fused project path');
            }
            this.cull = new CullPipeline({
                device: this.device,
                capacity: this.capacity,
                pipes: cullPipes,
                splatsBuffer: this.splatsBuffer,
                keysBuffer: this.sorter.keysA,
                valuesBuffer: this.sorter.valuesA,
                instBuffer: this.instUnsorted,
                projectUniforms: this.projectUniforms,
            });
        }
        else {
            this.cull = null;
        }
        // -----------------------------------------------------------------
        // Optional WSR pipeline. Allocates the per-pixel accumulator + output
        // buffers (sized to wsrMaxWidth × wsrMaxHeight). The compute kernels
        // own the splatsBuffer read-only; the legacy sorted path stays
        // available on the same ComputeDecodePipeline instance for parity
        // testing.
        // -----------------------------------------------------------------
        this.useWSR = init.useWSR ?? false;
        if (this.useWSR && this.pager.numPages > 1) {
            throw new Error(`ComputeDecodePipeline: useWSR is not supported with multi-page splats ` +
                `(${this.pager.numPages} pages required). Reduce capacity to <= ${this.pager.splatsPerPage}.`);
        }
        if (this.useWSR) {
            this.wsr = new WSRPipeline({
                device: this.device,
                maxWidth: init.wsrMaxWidth ?? 1920,
                maxHeight: init.wsrMaxHeight ?? 1080,
                splatsBuffer: this.splatsBuffer,
            });
        }
        else {
            this.wsr = null;
        }
        // -----------------------------------------------------------------
        // Optional tile-prefix-sum WSR (B8 PR2 KILL recovery). Tile binning +
        // workgroup-shared accumulation eliminates per-pixel atomic
        // contention. Coexists with `useWSR` (the atomic-scatter path) on
        // the same ComputeDecodePipeline; `encode()` prefers the tile path
        // when both are set.
        // -----------------------------------------------------------------
        this.useWSRTile = init.useWSRTile ?? false;
        if (this.useWSRTile && this.pager.numPages > 1) {
            throw new Error(`ComputeDecodePipeline: useWSRTile is not yet supported with multi-page splats ` +
                `(${this.pager.numPages} pages required for capacity ${this.capacity}). ` +
                `Reduce capacity to <= ${this.pager.splatsPerPage} or use the fused project_gather path.`);
        }
        if (this.useWSRTile) {
            this.wsrTile = new WSRTilePipeline({
                device: this.device,
                maxWidth: init.wsrMaxWidth ?? 1920,
                maxHeight: init.wsrMaxHeight ?? 1080,
                splatsBuffer: this.splatsBuffer,
                maxPerTile: init.wsrTileMaxPerTile,
            });
        }
        else {
            this.wsrTile = null;
        }
    }
    /**
     * Phase 2b: rebuild every bind group that captured the SH-rest buffer
     * after the dummy buffer is replaced by a real per-scene buffer in
     * uploadChunk(). The bind group otherwise still points at the destroyed
     * dummy GPUBuffer and the next dispatch crashes.
     */
    _rebuildShRestBindGroups() {
        const INSTANCE_BYTES = FLOATS_PER_INSTANCE * 4;
        if (!this.useFusedProject) {
            // Non-fused path: cs_project binding.
            this.projectBindGroup = this.device.createBindGroup({
                layout: this.pipes.projectBgl,
                entries: [
                    { binding: 0, resource: { buffer: this.splatsBuffer } },
                    { binding: 1, resource: { buffer: this.instUnsorted } },
                    { binding: 2, resource: { buffer: this.sorter.keysA } },
                    { binding: 3, resource: { buffer: this.sorter.valuesA } },
                    { binding: 4, resource: { buffer: this.projectUniforms } },
                    { binding: 5, resource: { buffer: this.shRestBuffer } },
                ],
            });
            return;
        }
        // Fused path: rebuild every project_gather bind group. Keygen bind group
        // does not reference the SH-rest buffer, so it stays valid.
        if (this.pager.numPages > 1) {
            const bgl = this._projectGatherBglPaged;
            const N = this.pager.numPages;
            const pageEntries = (downstream) => [
                ...this.pager.pageBuffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
                ...downstream.map((e) => ({ ...e, binding: e.binding + (N - 1) })),
            ];
            this.projectGatherBindGroups = [];
            for (const ipage of this.instancePages) {
                const idxOffset = ipage.splatStart * 4;
                const idxSize = Math.max(ipage.splatCount * 4, 4);
                const outSize = Math.max(ipage.splatCount * INSTANCE_BYTES, INSTANCE_BYTES);
                this.projectGatherBindGroups.push(this.device.createBindGroup({
                    layout: bgl,
                    entries: pageEntries([
                        { binding: 1, resource: { buffer: this.sorter.valuesA, offset: idxOffset, size: idxSize } },
                        { binding: 2, resource: { buffer: ipage.buffer, offset: 0, size: outSize } },
                        { binding: 3, resource: { buffer: this.projectUniforms } },
                        { binding: 4, resource: { buffer: this.shRestBuffer } },
                    ]),
                }));
            }
            this.projectGatherBindGroup = this.projectGatherBindGroups[0];
        }
        else {
            const bgl = this._projectGatherBglSingle;
            this.projectGatherBindGroup = this.device.createBindGroup({
                layout: bgl,
                entries: [
                    { binding: 0, resource: { buffer: this.splatsBuffer } },
                    { binding: 1, resource: { buffer: this.sorter.valuesA } },
                    { binding: 2, resource: { buffer: this.instanceBuffer } },
                    { binding: 3, resource: { buffer: this.projectUniforms } },
                    { binding: 4, resource: { buffer: this.shRestBuffer } },
                ],
            });
            this.projectGatherBindGroups = [this.projectGatherBindGroup];
        }
    }
    /**
     * Stage one chunk's raw bytes onto the GPU and dispatch the decode shader.
     * The decoded splats land at `[decodedSplats, decodedSplats + chunkCount)`
     * inside `splatsBuffer`.
     *
     * Returns immediately after queueing; the decode runs on the GPU before
     * the next frame's project pass.
     */
    uploadChunk(descriptor, bytes) {
        if (!descriptor.attributeLayout) {
            throw new Error('compute-decode: chunk has no attributeLayout (legacy AoS not supported on GPU path)');
        }
        if (descriptor.splatCount === 0)
            return;
        if (this.decodedSplats + descriptor.splatCount > this.capacity) {
            throw new Error(`compute-decode: capacity exceeded (${this.decodedSplats + descriptor.splatCount} > ${this.capacity})`);
        }
        // Round up to a multiple of 4 for u32-aligned storage reads.
        const padBytes = (bytes.byteLength + 3) & ~3;
        const bytesBuffer = this.device.createBuffer({
            size: Math.max(padBytes, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // writeBuffer requires a 4-byte-aligned source size and offset. Pad with zeros.
        if (padBytes === bytes.byteLength) {
            this.device.queue.writeBuffer(bytesBuffer, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
        }
        else {
            const padded = new Uint8Array(padBytes);
            padded.set(bytes);
            this.device.queue.writeBuffer(bytesBuffer, 0, padded.buffer, 0, padBytes);
        }
        const decodeUniforms = buildDecodeUniforms(this.device, descriptor.attributeLayout, descriptor.splatCount);
        // Stage 6 (sf-154): the destination splats live across N pager pages.
        // For each page sub-range that this chunk overlaps, we issue a
        // separate decode dispatch with the page's GPUBuffer bound (with a
        // dynamic-offset binding) and a per-sub-range chunk_offset that
        // selects the right slice of the source bytes. When numPages == 1
        // this collapses to a single dispatch identical to the pre-Stage-6
        // path.
        const encoder = this.device.createCommandEncoder();
        let srcSplatOffset = 0;
        for (const range of this.pager.pageRanges(this.decodedSplats, descriptor.splatCount)) {
            const dstView = {
                buffer: this.pager.pageBuffers[range.page],
                offset: range.localStart * BYTES_PER_DECODED_SPLAT,
                size: range.localCount * BYTES_PER_DECODED_SPLAT,
            };
            // The decode kernel reads source bytes by index relative to splat 0
            // of the chunk and writes dst_splats[i]. Since we sliced dst with a
            // dynamic-offset binding, the write index `i` for THIS sub-range is
            // also page-local — so we pass chunk_offset = srcSplatOffset to
            // make the shader read source slice [srcSplatOffset..]. Currently
            // the decode kernel uses i = gid.x + chunk_offset for BOTH the source
            // SoA index AND the dst index. We patch by giving each sub-range a
            // freshly built decodeUniforms with splat_count = sub-range count
            // and SoA byteOffset rebased.
            // Chunk may straddle a splats-page boundary. For each sub-range we
            // build a fresh decodeUniforms whose SoA byteOffsets are rebased by
            // srcSplatOffset so the kernel reads the right source slice with
            // chunk_offset = 0. The dst binding is already sliced via dynamic
            // offset to the destination page's local range.
            const subUniforms = buildDecodeUniformsForRange(this.device, descriptor.attributeLayout, range.localCount, srcSplatOffset);
            const subBindGroupSliced = this.device.createBindGroup({
                layout: this.pipes.decodeBgl,
                entries: [
                    { binding: 0, resource: { buffer: bytesBuffer } },
                    { binding: 1, resource: dstView },
                    { binding: 2, resource: { buffer: subUniforms } },
                ],
            });
            dispatchPerSplat(this.device, encoder, this.pipes.decode, subBindGroupSliced, subUniforms, UNIFORM_CHUNK_OFFSET_BYTES.decode, range.localCount);
            srcSplatOffset += range.localCount;
        }
        this.device.queue.submit([encoder.finish()]);
        // Phase 2b: stage the per-splat SH-rest blob (if present) into the
        // pipeline-lifetime sh_rest storage buffer at the correct splat offset.
        // The descriptor's attributeLayout carries `shRest.byteOffset/Length` into
        // the same source `bytes` buffer that `bytesBuffer` already mirrors.
        const layout = descriptor.attributeLayout;
        if (layout.shRest && layout.shDegree && layout.shDegree > 0) {
            const incomingDeg = layout.shDegree;
            const incomingCoef = shCoefCountForDegree(incomingDeg);
            // Allocate the scene-wide SH-rest buffer on first SH-bearing chunk.
            // The buffer is sized for the pipeline's full capacity so subsequent
            // chunks can append without reallocation.
            if (this.shDegree === 0) {
                this.shDegree = incomingDeg;
                this.shCoefCount = incomingCoef;
                const totalBytes = this.capacity * this.shCoefCount * 3 * 4;
                // Re-allocate the per-pipeline buffer (was the 16-byte dummy).
                this.shRestBuffer.destroy();
                this.shRestBuffer = this.device.createBuffer({
                    size: Math.max(totalBytes, 16),
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
                // The bind groups created in the constructor still reference the
                // OLD dummy buffer. Rebuild every bind group that captured it.
                this._rebuildShRestBindGroups();
            }
            else if (this.shDegree !== incomingDeg) {
                throw new Error(`compute-decode: SH degree mismatch across chunks ` +
                    `(have ${this.shDegree}, got ${incomingDeg}). All chunks in a scene must agree.`);
            }
            // Copy this chunk's SH-rest slice into the per-pipeline buffer at the
            // splat offset. The blob is contiguous (splat-major) so a single
            // writeBuffer suffices.
            const splatStrideBytes = this.shCoefCount * 3 * 4;
            const dstByteOffset = this.decodedSplats * splatStrideBytes;
            const srcByteOffset = bytes.byteOffset + layout.shRest.byteOffset;
            // Align to 4-byte boundary for queue.writeBuffer (already u32-aligned
            // because each splat is coefCount * 3 floats and floats are 4 B).
            this.device.queue.writeBuffer(this.shRestBuffer, dstByteOffset, bytes.buffer, srcByteOffset, layout.shRest.byteLength);
        }
        this.chunks.push({
            splatCount: descriptor.splatCount,
            bytesBuffer,
            splatsBuffer: this.pager.pageBuffers[0],
            decodeUniforms,
            // decodeBindGroup field kept for backward-compat — the per-sub-range
            // bind group is created+used inline above and not retained.
            decodeBindGroup: this.device.createBindGroup({
                layout: this.pipes.decodeBgl,
                entries: [
                    { binding: 0, resource: { buffer: bytesBuffer } },
                    { binding: 1, resource: { buffer: this.pager.pageBuffers[0] } },
                    { binding: 2, resource: { buffer: decodeUniforms } },
                ],
            }),
        });
        this.decodedSplats += descriptor.splatCount;
    }
    /** Number of splats decoded so far. */
    get splatCount() {
        return this.decodedSplats;
    }
    /**
     * Record project + sort + gather dispatches for the current frame.
     *
     * @param encoder command encoder owned by the caller (typically the renderer).
     * @param view column-major view matrix.
     * @param viewProj column-major view-projection matrix.
     * @param focal `[focalX, focalY]` in pixels.
     * @param viewport `[width, height]` in pixels.
     * @param cameraPos optional world-space camera position. Required when
     *                  the SH-rest evaluator is active (a non-null SH-rest
     *                  buffer was uploaded AND `shRestEnabled` is true). When
     *                  omitted, the shader's SH-rest branch is force-disabled
     *                  for this frame (DC-only colors, baseline behavior).
     */
    encode(encoder, view, viewProj, focal, viewport, cameraPos) {
        const count = this.decodedSplats;
        if (count === 0)
            return;
        // WSR-Tile path (B8 PR2 KILL recovery). Tile-prefix-sum scatter; one
        // workgroup per tile, accumulators in workgroup-private registers.
        // Mutually preferred over the legacy useWSR atomic-scatter when both
        // flags are set — the tile path is the production-viable WSR.
        if (this.useWSRTile && this.wsrTile) {
            this.encodeWSRTile(encoder, view, viewProj, focal, viewport);
            return;
        }
        // WSR path (PR1 feature flag). Skips keygen/radix/gather entirely and
        // produces the final rgba8unorm-packed frame in `wsr.outputBuffer`. The
        // sorted alpha-blend instanceBuffer path is bypassed; the caller (the
        // renderer or the unit test) reads `wsr.outputBuffer` instead.
        if (this.useWSR && this.wsr) {
            this.encodeWSR(encoder, view, viewProj, focal, viewport);
            return;
        }
        // Phase 2b: SH-rest evaluator only runs when we have data AND the
        // caller passed a camera position AND the runtime toggle is on. Any
        // missing piece collapses to the byte-identical DC-only OFF path.
        const shActive = this.shDegree > 0 && this.shRestEnabled && cameraPos !== undefined;
        // Pack project uniforms (Phase 2b layout): 2 mat4(32) + viewport(2)
        // + focal(2) + (count, chunk_offset, pad, pad)(4) + cam_pos(4) +
        // sh_params(4) = 48 floats / 192 bytes.
        const ab = new ArrayBuffer(192);
        const f32 = new Float32Array(ab);
        const i32 = new Int32Array(ab);
        const u32 = new Uint32Array(ab);
        f32.set(view, 0);
        f32.set(viewProj, 16);
        f32[32] = viewport[0];
        f32[33] = viewport[1];
        f32[34] = focal[0];
        f32[35] = focal[1];
        i32[36] = count;
        // cam_pos at float slot 40..43.
        f32[40] = cameraPos ? cameraPos[0] : 0;
        f32[41] = cameraPos ? cameraPos[1] : 0;
        f32[42] = cameraPos ? cameraPos[2] : 0;
        f32[43] = shActive ? 1.0 : 0.0; // sh_enabled flag in cam_pos.w
        // sh_params at u32 slot 44..47.
        u32[44] = shActive ? (this.shDegree >>> 0) : 0;
        u32[45] = shActive ? (this.shCoefCount >>> 0) : 0;
        u32[46] = 0;
        u32[47] = 0;
        this.device.queue.writeBuffer(this.projectUniforms, 0, ab);
        const wgs = Math.ceil(count / 256);
        if (this.useFusedProject) {
            // Fused path:
            //   1. cs_keygen        — depth-only key + identity index.
            //   2. radix sort       — sorts (key, index) ascending by key.
            //   3. cs_project_gather — re-projects in sorted order, writes
            //                          instanceBuffer[i] directly. No 640 MB
            //                          unsorted-scratch buffer touched.
            // Multi-dispatch unblocks > 16.7 M-splat scenes (LODGE L1/L0).
            // NOTE: `this.sorter.encode` is NOT chunked yet (Stage 5 followup —
            // see tasks/scripts/wgpu-multidispatch-followup.md). Until that
            // lands, scenes whose splat count exceeds the radix-sort dispatch
            // cap will still fail at the sort, even though keygen + gather
            // tolerate the size. The bench-side `dispatchCap` reflects this.
            void wgs;
            dispatchPerSplat(this.device, encoder, this._pagedKeygen ?? this.pipes.keygen, this.keygenBindGroup, this.projectUniforms, UNIFORM_CHUNK_OFFSET_BYTES.project, count);
            this.sorter.encode(encoder, count);
            // Stage 6 (sf-154): dispatch project_gather once per INSTANCE page
            // with chunk_offset = page.splatStart so the kernel's bounds guard
            // (i >= splat_count) still triggers correctly. The bind groups
            // dynamically slice the indices + inst_out buffers to the page's
            // range, so `i = gid.x + chunk_offset` reads/writes the right
            // global slot for THIS page. Splat-count uniform stays at the
            // global `count` so the dispatch caps at the last page's tail.
            const pgPipeline = this._pagedProjectGather ?? this.pipes.projectGather;
            for (let pi = 0; pi < this.projectGatherBindGroups.length; pi++) {
                const page = this.instancePages[pi];
                const pageStart = page.splatStart;
                const pageCount = Math.min(page.splatCount, count - pageStart);
                if (pageCount <= 0)
                    break;
                // Carve this page into <=65535-WG chunks. chunk_offset starts at
                // pageStart so g_indices / g_inst_out reads land in the page's
                // slice (the bind group's dynamic offset has subtracted pageStart
                // worth of bytes, so kernel-side `i - pageStart` indexes from 0).
                // Wait — the bind group offset means kernel sees buffer starting
                // at byte=offset. So kernel's `g_inst_out[i]` with i = gid.x +
                // chunk_offset and chunk_offset = pageStart maps to global slot
                // pageStart + (offset/48) + (gid.x + chunk_offset). That double-
                // counts pageStart. Fix: chunk_offset for the kernel must be 0
                // when the binding is already sliced. But then the bounds guard
                // `i >= splat_count` fires at pageCount, not at global count.
                // So: set splat_count = pageCount and chunk_offset = 0 for each
                // page dispatch.
                const scratch = new Uint32Array(2);
                scratch[0] = pageCount >>> 0; // splat_count slot
                scratch[1] = 0; // chunk_offset slot
                // ProjectUniforms layout: 2×mat4(128) + viewport(8) + focal(8)
                // + splat_count(4) + chunk_offset(4). splat_count at byte 144.
                this.device.queue.writeBuffer(this.projectUniforms, 144, scratch.buffer, 0, 8);
                dispatchPerSplat(this.device, encoder, pgPipeline, this.projectGatherBindGroups[pi], this.projectUniforms, UNIFORM_CHUNK_OFFSET_BYTES.project, pageCount);
            }
            // Restore splat_count = global count after the per-page loop so any
            // downstream code that re-reads the uniforms sees the global value.
            {
                const scratch = new Uint32Array(1);
                scratch[0] = count >>> 0;
                this.device.queue.writeBuffer(this.projectUniforms, 144, scratch.buffer, 0, 4);
            }
            return;
        }
        // Non-fused (legacy) path: cs_project → radix sort → cs_gather.
        // Kept as a fallback / parity reference.
        dispatchPerSplat(this.device, encoder, this.pipes.project, this.projectBindGroup, this.projectUniforms, UNIFORM_CHUNK_OFFSET_BYTES.project, count);
        this.sorter.encode(encoder, count);
        {
            const u = new Uint32Array(8); // 32 bytes
            u[0] = count;
            this.device.queue.writeBuffer(this.gatherUniforms, 0, u.buffer);
            dispatchPerSplat(this.device, encoder, this.pipes.gather, this.gatherBindGroup, this.gatherUniforms, UNIFORM_CHUNK_OFFSET_BYTES.gather, count);
        }
    }
    /**
     * Weighted Sum Rendering encode path (PR1).
     *
     * Records `clear → scatter-accumulate → resolve` into the caller's encoder.
     * Skips the keygen / radix-sort / gather kernels entirely — WSR is order-
     * independent, so the sort cost (30.15 ms / 58 % of frame at 3.62 M splats
     * per the real-scene bench) is reclaimed.
     *
     * The final rgba8unorm-packed frame lands in `this.wsr.outputBuffer`. The
     * renderer is expected to `copyBufferToTexture` from there to the canvas
     * texture; the unit test in `wsr.test.ts` instead reads the buffer back via
     * a staging copy and asserts the bytes are finite-positive non-zero.
     *
     * **Path-choice rationale (per spec § 5 "Two-Pass vs One-Pass"):** the spec
     * recommends Option A (fragment-shader rasterization with ROP additive blend
     * on rgba32float + r32float MRT). We instead use Option B (compute scatter
     * with CAS-loop atomic float-add on `array<atomic<u32>>` buffers) because
     * the `float32-blendable` device feature required by Option A is not in any
     * shipped WebGPU 1.0 implementation as of 2026-05 (Chrome canary only).
     * The B7.1 EXECUTION-LOG measurement (2026-05-15) established that on the
     * laptop 4090 the scatter path is DRAM-write-bound, not atomic-bound, at
     * 10 M splats — the CAS-loop's amortized iteration count is ≤ 2 even under
     * heavy contention. Option B is the portable, measured-equivalent choice
     * for PR1; we re-evaluate if `float32-blendable` ships in stable channels
     * before PR5 flips WSR to default.
     */
    encodeWSR(encoder, view, viewProj, focal, viewport) {
        if (!this.wsr)
            throw new Error('encodeWSR requires useWSR=true');
        const count = this.decodedSplats;
        if (count === 0)
            return;
        this.wsr.encode(encoder, view, viewProj, focal, viewport, count, this.wsrSigma, this.wsrBgColor);
    }
    /**
     * Tile-prefix-sum WSR encode path (B8 PR2 KILL recovery).
     *
     * Records `tile-clear → tile-bin → per-tile accumulate → resolve` into
     * the caller's encoder. The per-pixel accumulators are written from
     * workgroup-private registers — NO per-pixel atomics, no CAS-loop, no
     * float32-blendable dependency.
     *
     * Final rgba8unorm-packed frame lands in `this.wsrTile.outputBuffer`.
     */
    encodeWSRTile(encoder, view, viewProj, focal, viewport) {
        if (!this.wsrTile)
            throw new Error('encodeWSRTile requires useWSRTile=true');
        const count = this.decodedSplats;
        if (count === 0)
            return;
        this.wsrTile.encode(encoder, view, viewProj, focal, viewport, count, this.wsrSigma, this.wsrBgColor);
    }
    /**
     * Cull-enabled production encode. Runs:
     *   1. cs_cull + scan + cs_compact + cs_project_cmpct  (writes keysA /
     *      valuesA / instUnsorted at the first `survivors` slots).
     *   2. Radix sort over `survivors`.
     *   3. cs_gather over `survivors`.
     *
     * Requires the cull's survivor count to have been refreshed via
     * `cull.readSurvivorCount()` after a prior `encodeWithCull` was
     * submitted. On the very first frame, the count is 0 and the sort /
     * gather are skipped — the caller is expected to do a warm-up
     * submit + readback before the timed loop begins.
     *
     * `tau` is the opacity floor (default 1/255). The bench overrides it
     * if 1/255 prunes too many splats on the synthetic scene.
     */
    async encodeWithCull(encoder, view, viewProj, focal, viewport, tau = 1 / 255) {
        if (!this.cull)
            throw new Error('encodeWithCull requires useCull=true');
        const count = this.decodedSplats;
        if (count === 0)
            return;
        // Pack project uniforms — same packing as encode(), but the `splat_count`
        // slot will be overwritten by the host to `survivors` for cs_project_cmpct.
        // Phase 2b: same 192-byte layout as encode(); SH-rest stays disabled on
        // this path (cs_cull / cs_project_cmpct don't read sh_params).
        const ab = new ArrayBuffer(192);
        const f32 = new Float32Array(ab);
        const i32 = new Int32Array(ab);
        f32.set(view, 0);
        f32.set(viewProj, 16);
        f32[32] = viewport[0];
        f32[33] = viewport[1];
        f32[34] = focal[0];
        f32[35] = focal[1];
        i32[36] = this.cull.cachedSurvivors > 0 ? this.cull.cachedSurvivors : count;
        // cam_pos.w = 0 → shader's SH-rest branch short-circuits to DC-only.
        this.device.queue.writeBuffer(this.projectUniforms, 0, ab);
        // 1. Cull + compact + project (project only fires if cachedSurvivors > 0).
        this.cull.encode(encoder, view, viewProj, focal, viewport, count, tau);
        // 2. Radix sort + gather over survivors. On the very first frame,
        //    cachedSurvivors is 0; we skip these and let the readback populate
        //    the count for the next frame.
        const survivors = this.cull.cachedSurvivors;
        if (survivors > 0) {
            this.sorter.encode(encoder, survivors);
            const u = new Uint32Array(8);
            u[0] = survivors;
            this.device.queue.writeBuffer(this.gatherUniforms, 0, u.buffer);
            dispatchPerSplat(this.device, encoder, this.pipes.gather, this.gatherBindGroup, this.gatherUniforms, UNIFORM_CHUNK_OFFSET_BYTES.gather, survivors);
        }
    }
    /**
     * Timestamp-instrumented variant of `encodeWithCull`. Writes 8 timestamps:
     *   [0..1) cull+scan+compact   (combined; cull internals not drilled)
     *   [2..3) project_cmpct
     *   [4..5) sort_full
     *   [6..7) gather
     * Returns the number of timestamps written. Requires `timestamp-query`.
     */
    encodeWithCullTimed(encoder, view, viewProj, focal, viewport, querySet, baseIndex, tau = 1 / 255) {
        if (!this.cull)
            throw new Error('encodeWithCullTimed requires useCull=true');
        const count = this.decodedSplats;
        if (count === 0)
            return 0;
        const ab = new ArrayBuffer(160);
        const f32 = new Float32Array(ab);
        const i32 = new Int32Array(ab);
        f32.set(view, 0);
        f32.set(viewProj, 16);
        f32[32] = viewport[0];
        f32[33] = viewport[1];
        f32[34] = focal[0];
        f32[35] = focal[1];
        i32[36] = this.cull.cachedSurvivors > 0 ? this.cull.cachedSurvivors : count;
        this.device.queue.writeBuffer(this.projectUniforms, 0, ab);
        // Re-pack the cull's own uniforms.
        const cull = this.cull;
        {
            const cb = new ArrayBuffer(cull.cullUniforms.size);
            const cf = new Float32Array(cb);
            const cu = new Uint32Array(cb);
            cf.set(view, 0);
            cf.set(viewProj, 16);
            cf[32] = viewport[0];
            cf[33] = viewport[1];
            cf[34] = focal[0];
            cf[35] = focal[1];
            cu[36] = count;
            cf[37] = tau;
            this.device.queue.writeBuffer(cull.cullUniforms, 0, cb);
        }
        const numScanWgs = Math.ceil(count / 256);
        {
            const u = new Uint32Array(8);
            u[0] = count;
            u[1] = numScanWgs;
            this.device.queue.writeBuffer(cull.scanUniforms, 0, u.buffer);
        }
        {
            const u = new Uint32Array(8);
            u[0] = count;
            this.device.queue.writeBuffer(cull.compactUniforms, 0, u.buffer);
        }
        const wgs = Math.ceil(count / 256);
        // Window 1: cull + scan + compact.
        {
            const pass = encoder.beginComputePass({
                timestampWrites: {
                    querySet,
                    beginningOfPassWriteIndex: baseIndex + 0,
                    endOfPassWriteIndex: baseIndex + 1,
                },
            });
            pass.setPipeline(cull.pipes.cull);
            pass.setBindGroup(0, cull.cullBindGroup);
            pass.dispatchWorkgroups(wgs);
            pass.setPipeline(cull.pipes.scanPerWg);
            pass.setBindGroup(0, cull.scanBindGroup);
            pass.dispatchWorkgroups(numScanWgs);
            pass.setPipeline(cull.pipes.scanBlockSums);
            pass.dispatchWorkgroups(1);
            pass.setPipeline(cull.pipes.scanAddBlockSums);
            pass.dispatchWorkgroups(numScanWgs);
            pass.setPipeline(cull.pipes.compact);
            pass.setBindGroup(0, cull.compactBindGroup);
            pass.dispatchWorkgroups(wgs);
            pass.end();
        }
        // Copy the tail readback (after compact's pass closed).
        const tailOff = (count - 1) * 4;
        encoder.copyBufferToBuffer(cull.prefixBuffer, tailOff, cull.readbackTail, 0, 4);
        encoder.copyBufferToBuffer(cull.flagBuffer, tailOff, cull.readbackTail, 4, 4);
        const survivors = cull.cachedSurvivors;
        if (survivors === 0) {
            // No survivors yet — skip the downstream stages but emit zero-width
            // timestamp windows so the resolveQuerySet copy stays valid.
            for (let k = 2; k < 8; k += 2) {
                const p = encoder.beginComputePass({
                    timestampWrites: {
                        querySet,
                        beginningOfPassWriteIndex: baseIndex + k,
                        endOfPassWriteIndex: baseIndex + k + 1,
                    },
                });
                p.end();
            }
            return 8;
        }
        // Window 2: project_cmpct.
        {
            const pass = encoder.beginComputePass({
                timestampWrites: {
                    querySet,
                    beginningOfPassWriteIndex: baseIndex + 2,
                    endOfPassWriteIndex: baseIndex + 3,
                },
            });
            pass.setPipeline(cull.pipes.projectCmpct);
            pass.setBindGroup(0, cull.projectCmpctBindGroup);
            pass.dispatchWorkgroups(Math.ceil(survivors / 256));
            pass.end();
        }
        // Window 3: sort_full.
        this.sorter.encodeTimed(encoder, survivors, querySet, baseIndex + 4);
        // Window 4: gather.
        {
            const u = new Uint32Array(8);
            u[0] = survivors;
            this.device.queue.writeBuffer(this.gatherUniforms, 0, u.buffer);
            const pass = encoder.beginComputePass({
                timestampWrites: {
                    querySet,
                    beginningOfPassWriteIndex: baseIndex + 6,
                    endOfPassWriteIndex: baseIndex + 7,
                },
            });
            pass.setPipeline(this.pipes.gather);
            pass.setBindGroup(0, this.gatherBindGroup);
            pass.dispatchWorkgroups(Math.ceil(survivors / 256));
            pass.end();
        }
        return 8;
    }
    /**
   * Bench-only variant of `encode` that writes timestamps around each top-
   * level compute pass. Returns the number of timestamps written, so the
   * caller knows how to slice the resolve buffer.
   *
   * Layout, fused path (the only path we ship by default):
   *   [0..1) keygen          [1..2) sort_full          [2..3) project_gather
   * Non-fused fallback:
   *   [0..1) project_unsorted [1..2) sort_full         [2..3) gather
   *
   * Requires the `timestamp-query` feature on the device. The bench owns
   * the GPUQuerySet so we keep the dependency one-way (production code does
   * not import any timestamp plumbing).
   */
    encodeTimed(encoder, view, viewProj, focal, viewport, querySet, baseIndex) {
        const count = this.decodedSplats;
        if (count === 0)
            return 0;
        const ab = new ArrayBuffer(160);
        const f32 = new Float32Array(ab);
        const i32 = new Int32Array(ab);
        f32.set(view, 0);
        f32.set(viewProj, 16);
        f32[32] = viewport[0];
        f32[33] = viewport[1];
        f32[34] = focal[0];
        f32[35] = focal[1];
        i32[36] = count;
        this.device.queue.writeBuffer(this.projectUniforms, 0, ab);
        const wgs = Math.ceil(count / 256);
        if (this.useFusedProject) {
            // keygen
            {
                const pass = encoder.beginComputePass({
                    timestampWrites: {
                        querySet,
                        beginningOfPassWriteIndex: baseIndex + 0,
                        endOfPassWriteIndex: baseIndex + 1,
                    },
                });
                pass.setPipeline(this._pagedKeygen ?? this.pipes.keygen);
                pass.setBindGroup(0, this.keygenBindGroup);
                pass.dispatchWorkgroups(wgs);
                pass.end();
            }
            // sort_full — radix sort orchestrates its own 4 passes; we wrap the
            // *whole* sort in one timestamp window. Drilling into histogram/scan/
            // scatter requires `timestamp-query-inside-passes`, which we'll add
            // in a follow-up if sort_full proves to dominate frame time.
            this.sorter.encodeTimed(encoder, count, querySet, baseIndex + 2);
            // project_gather
            {
                const pass = encoder.beginComputePass({
                    timestampWrites: {
                        querySet,
                        beginningOfPassWriteIndex: baseIndex + 4,
                        endOfPassWriteIndex: baseIndex + 5,
                    },
                });
                pass.setPipeline(this._pagedProjectGather ?? this.pipes.projectGather);
                pass.setBindGroup(0, this.projectGatherBindGroup);
                pass.dispatchWorkgroups(wgs);
                pass.end();
            }
            return 6;
        }
        // Non-fused (legacy) path.
        {
            const pass = encoder.beginComputePass({
                timestampWrites: {
                    querySet,
                    beginningOfPassWriteIndex: baseIndex + 0,
                    endOfPassWriteIndex: baseIndex + 1,
                },
            });
            pass.setPipeline(this.pipes.project);
            pass.setBindGroup(0, this.projectBindGroup);
            pass.dispatchWorkgroups(wgs);
            pass.end();
        }
        this.sorter.encodeTimed(encoder, count, querySet, baseIndex + 2);
        {
            const u = new Uint32Array(8);
            u[0] = count;
            this.device.queue.writeBuffer(this.gatherUniforms, 0, u.buffer);
            const pass = encoder.beginComputePass({
                timestampWrites: {
                    querySet,
                    beginningOfPassWriteIndex: baseIndex + 4,
                    endOfPassWriteIndex: baseIndex + 5,
                },
            });
            pass.setPipeline(this.pipes.gather);
            pass.setBindGroup(0, this.gatherBindGroup);
            pass.dispatchWorkgroups(wgs);
            pass.end();
        }
        return 6;
    }
    /**
     * Bench-only variant that drills the radix sort into per-sub-stage
     * timestamps. Writes 20 timestamps total:
     *   [0..1)   keygen / project
     *   [2..3)   pass0_histogram
     *   [4..5)   pass0_scan_per_wg
     *   [6..7)   pass0_scan_block_sums
     *   [8..9)   pass0_scan_add_block_sums
     *   [10..11) pass0_scatter
     *   [12..13) pass1_full
     *   [14..15) pass2_full
     *   [16..17) pass3_full
     *   [18..19) project_gather / gather
     * Caller must supply a querySet of capacity >= 20.
     */
    encodeTimedDrilled(encoder, view, viewProj, focal, viewport, querySet, baseIndex) {
        const count = this.decodedSplats;
        if (count === 0)
            return 0;
        const ab = new ArrayBuffer(160);
        const f32 = new Float32Array(ab);
        const i32 = new Int32Array(ab);
        f32.set(view, 0);
        f32.set(viewProj, 16);
        f32[32] = viewport[0];
        f32[33] = viewport[1];
        f32[34] = focal[0];
        f32[35] = focal[1];
        i32[36] = count;
        this.device.queue.writeBuffer(this.projectUniforms, 0, ab);
        const wgs = Math.ceil(count / 256);
        const ts = (begin, end) => ({
            timestampWrites: {
                querySet,
                beginningOfPassWriteIndex: begin,
                endOfPassWriteIndex: end,
            },
        });
        if (this.useFusedProject) {
            {
                const pass = encoder.beginComputePass(ts(baseIndex + 0, baseIndex + 1));
                pass.setPipeline(this._pagedKeygen ?? this.pipes.keygen);
                pass.setBindGroup(0, this.keygenBindGroup);
                pass.dispatchWorkgroups(wgs);
                pass.end();
            }
            this.sorter.encodeTimedDrilled(encoder, count, querySet, baseIndex + 2);
            {
                const pass = encoder.beginComputePass(ts(baseIndex + 18, baseIndex + 19));
                pass.setPipeline(this._pagedProjectGather ?? this.pipes.projectGather);
                pass.setBindGroup(0, this.projectGatherBindGroup);
                pass.dispatchWorkgroups(wgs);
                pass.end();
            }
            return 20;
        }
        {
            const pass = encoder.beginComputePass(ts(baseIndex + 0, baseIndex + 1));
            pass.setPipeline(this.pipes.project);
            pass.setBindGroup(0, this.projectBindGroup);
            pass.dispatchWorkgroups(wgs);
            pass.end();
        }
        this.sorter.encodeTimedDrilled(encoder, count, querySet, baseIndex + 2);
        {
            const u = new Uint32Array(8);
            u[0] = count;
            this.device.queue.writeBuffer(this.gatherUniforms, 0, u.buffer);
            const pass = encoder.beginComputePass(ts(baseIndex + 18, baseIndex + 19));
            pass.setPipeline(this.pipes.gather);
            pass.setBindGroup(0, this.gatherBindGroup);
            pass.dispatchWorkgroups(wgs);
            pass.end();
        }
        return 20;
    }
    /**
     * Stage 6: build templated cs_keygen + cs_project_gather pipelines for
     * multi-page splats. The original PROJECT_GATHER_WGSL has a single
     * splats binding (k_splats / g_splats); we rewrite each entry point's
     * WGSL to declare N page bindings + a `read_splats_*(i)` helper, then
     * compile a fresh shader module + pipeline per entry point.
     *
     * Returns the templated keygen and project_gather pipelines plus their
     * matching bind-group layouts (which are also paged: bindings
     * [0, N) are page buffers, downstream bindings are shifted by N-1).
     */
    _buildPagedFusedPipelines(numPages, splatsPerPage) {
        const COMPUTE = GPUShaderStage.COMPUTE;
        // PROJECT_GATHER_WGSL contains BOTH cs_keygen (binding name k_splats)
        // and cs_project_gather (binding name g_splats). We template each binding
        // separately; templateSplatsAccess emits N page bindings for the named
        // binding and rebases the others.
        const dilSrc = applyDilationOverride(PROJECT_GATHER_WGSL, this.dilation);
        // The WGSL file contains BOTH cs_keygen and cs_project_gather; each
        // entry point declares its own binding-0 splats array. If we template
        // the whole file for 'k_splats' (numPages > 1), the rebasing also
        // shifts the g_splats bindings into the same slot as k_keys, causing
        // a 'multiple variables use the same binding' WGSL error.
        //
        // Fix: split the source at the cs_project_gather marker so each
        // templating pass only sees its own entry point's bindings. The
        // common preamble (structs, helpers above cs_keygen) stays in the
        // keygen half; we replicate it into the gather half so the gather
        // module is self-contained.
        const splitMarker = '// cs_project_gather — full projection';
        const markerIdx = dilSrc.indexOf(splitMarker);
        if (markerIdx < 0) {
            throw new Error('cs_project_gather split marker not found in PROJECT_GATHER_WGSL');
        }
        // Common preamble = everything up to the structs section's end (before
        // the first @group(0) binding).
        const firstBindingIdx = dilSrc.indexOf('@group(0) @binding(0)');
        const preamble = dilSrc.slice(0, firstBindingIdx);
        const keygenSrc = preamble + dilSrc.slice(firstBindingIdx, markerIdx);
        const gatherSrc = preamble + dilSrc.slice(markerIdx);
        const tplK = templateSplatsAccess(keygenSrc, 'k_splats', numPages, splatsPerPage);
        const tplG = templateSplatsAccess(gatherSrc, 'g_splats', numPages, splatsPerPage);
        const keygenMod = this.device.createShaderModule({ code: tplK.wgsl });
        const pgMod = this.device.createShaderModule({ code: tplG.wgsl });
        // Keygen BGL: N page bindings (read-only-storage) + 3 downstream bindings
        // (keys, indices, uniforms) shifted by (N-1).
        const keygenEntries = [];
        for (let p = 0; p < numPages; p++) {
            keygenEntries.push({ binding: p, visibility: COMPUTE, buffer: { type: 'read-only-storage' } });
        }
        keygenEntries.push({ binding: numPages, visibility: COMPUTE, buffer: { type: 'storage' } });
        keygenEntries.push({ binding: numPages + 1, visibility: COMPUTE, buffer: { type: 'storage' } });
        keygenEntries.push({ binding: numPages + 2, visibility: COMPUTE, buffer: { type: 'uniform' } });
        const keygenBgl = this.device.createBindGroupLayout({ entries: keygenEntries });
        // Project_gather BGL: N page bindings + (indices, inst_out, uniforms, sh_rest).
        // Phase 2b: SH-rest blob (binding 4 in single-page WGSL) shifts to
        // `numPages + 3` after the templater rebases for multi-page splats.
        const pgEntries = [];
        for (let p = 0; p < numPages; p++) {
            pgEntries.push({ binding: p, visibility: COMPUTE, buffer: { type: 'read-only-storage' } });
        }
        pgEntries.push({ binding: numPages, visibility: COMPUTE, buffer: { type: 'read-only-storage' } });
        pgEntries.push({ binding: numPages + 1, visibility: COMPUTE, buffer: { type: 'storage' } });
        pgEntries.push({ binding: numPages + 2, visibility: COMPUTE, buffer: { type: 'uniform' } });
        pgEntries.push({ binding: numPages + 3, visibility: COMPUTE, buffer: { type: 'read-only-storage' } });
        const projectGatherBgl = this.device.createBindGroupLayout({ entries: pgEntries });
        const keygen = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [keygenBgl] }),
            compute: { module: keygenMod, entryPoint: 'cs_keygen' },
        });
        const projectGather = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [projectGatherBgl] }),
            compute: { module: pgMod, entryPoint: 'cs_project_gather' },
        });
        return { keygen, projectGather, keygenBgl, projectGatherBgl };
    }
    /** Tear down. Idempotent. */
    destroy() {
        for (const c of this.chunks) {
            c.bytesBuffer.destroy();
            c.decodeUniforms.destroy();
        }
        this.chunks.length = 0;
        this.pager.destroy();
        this.instUnsorted?.destroy();
        for (const p of this.instancePages)
            p.buffer.destroy();
        this.projectUniforms.destroy();
        this.shRestBuffer.destroy();
        this.gatherUniforms?.destroy();
        this.sorter.destroy();
        this.cull?.destroy();
        this.wsr?.destroy();
        this.wsrTile?.destroy();
    }
}
export { createRadixSortPipelines, RadixSort } from './radix_sort.js';
export { DECODE_WGSL, RADIX_SORT_WGSL, PROJECT_GATHER_WGSL, CULL_WGSL, WSR_CLEAR_WGSL, WSR_ACCUMULATE_WGSL, WSR_RESOLVE_WGSL, TILE_BIN_WGSL, WSR_TILE_ACCUMULATE_WGSL, } from './shaders.generated.js';
export { CullPipeline, createCullPipelines } from './cull.js';
export { WSRPipeline, createWSRPipelines, WSR_DEFAULT_BG_WEIGHT, WSR_TILE, WSR_WG, } from './wsr.js';
export { WSRTilePipeline, createWSRTilePipelines, WSR_TILE_SIZE, WSR_TILE_BIN_WG, WSR_TILE_CLEAR_WG, WSR_TILE_DEFAULT_MAX_PER_TILE, WSR_TILE_DEFAULT_BG_WEIGHT, } from './wsr_tile.js';
//# sourceMappingURL=index.js.map