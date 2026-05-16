#!/usr/bin/env python3
"""sf-154 Stage 6: migrate ComputeDecodePipeline to use BufferPager.

Strategy: surgical text patches against packages/viewer/src/webgpu/index.ts.

Behavior:
  - Replace the single splatsBuffer allocation with a BufferPager.
  - When pager.numPages == 1: keep all current bind groups by binding
    pager.pageBuffers[0] (no behavior change).
  - When pager.numPages > 1:
      * Fused-only path is supported. The cs_keygen + cs_project_gather
        kernels are rebuilt with templated WGSL (multi-page bindings) and
        new bind-group layouts.
      * Non-fused, cull, WSR, WSR-tile all throw at construction.
  - uploadChunk routes writes via pager.writeSplats (handles per-page
    sub-ranges automatically). Decode dispatch stays single-pass with the
    bind group's `dstView` chosen against the destination page; if a
    chunk straddles pages we split into 2 dispatches.
"""
import re, sys, pathlib

P = pathlib.Path(__file__).resolve().parents[2] / 'packages/viewer/src/webgpu/index.ts'
src = P.read_text()
orig = src

# 1. Add BufferPager import.
old_import = "import {\n  DECODE_WGSL,\n  RADIX_SORT_WGSL,\n  PROJECT_GATHER_WGSL,\n  SCAN_MULTIBLOCK_WGSL,\n  RADIX_MERGE_WGSL,\n} from './shaders.generated.js';"
assert old_import in src
new_import = old_import + "\nimport { BufferPager, templateSplatsAccess } from './buffer-pager.js';"
src = src.replace(old_import, new_import)

# 2. In ComputeDecodePipeline: replace `private readonly splatsBuffer: GPUBuffer;` with pager.
old_splatsbuf_decl = "  /** Canonical decoded-splat buffer. One per-splat record across all chunks. */\n  private readonly splatsBuffer: GPUBuffer;"
assert old_splatsbuf_decl in src
new_splatsbuf_decl = (
    "  /** Canonical decoded-splat buffer pager (Stage 6 / sf-154).\n"
    "   *  When numPages == 1 this is functionally identical to the old single-buffer\n"
    "   *  path. When numPages > 1 the fused project_gather path uses templated\n"
    "   *  multi-page bindings; non-fused / cull / WSR paths are unsupported and\n"
    "   *  throw at construction. */\n"
    "  readonly pager: BufferPager;\n"
    "  /** Convenience: page 0's buffer. Used by single-page bind groups for the\n"
    "   *  cull/WSR/non-fused paths and by uploadChunk's dstView. */\n"
    "  private get splatsBuffer(): GPUBuffer { return this.pager.pageBuffers[0]; }"
)
src = src.replace(old_splatsbuf_decl, new_splatsbuf_decl)

# 3. Replace the splatsBuffer createBuffer call with pager allocation.
old_alloc = (
    "    const decodedSize = Math.max(this.capacity * BYTES_PER_DECODED_SPLAT, BYTES_PER_DECODED_SPLAT);\n"
    "    this.splatsBuffer = this.device.createBuffer({\n"
    "      size: decodedSize,\n"
    "      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,\n"
    "    });"
)
assert old_alloc in src
new_alloc = (
    "    // Stage 6 (sf-154): split the canonical decoded-splat storage across N\n"
    "    // GPUBuffers (each <= adapter.maxStorageBufferBindingSize). For\n"
    "    // <= 33 M splats this is one page (identical layout to the old single-\n"
    "    // buffer path). At LODGE L1 (~54 M, 3.5 GB) it's 2 pages; at L0\n"
    "    // (~119 M, 7.6 GB) it's 4 pages at a 2 GiB cap.\n"
    "    const lim = (this.device as unknown as { limits: GPUSupportedLimits }).limits;\n"
    "    const maxBufferBytes = Math.min(\n"
    "      lim.maxStorageBufferBindingSize ?? (2 * 1024 * 1024 * 1024 - 1),\n"
    "      lim.maxBufferSize ?? (2 * 1024 * 1024 * 1024 - 1),\n"
    "    );\n"
    "    this.pager = new BufferPager(this.device, this.capacity, maxBufferBytes);\n"
    "    if (this.pager.numPages > 1 && !this.useFusedProject) {\n"
    "      throw new Error(\n"
    "        `ComputeDecodePipeline: capacity ${this.capacity} requires ${this.pager.numPages} ` +\n"
    "        `splat pages but useFusedProject=false (only the fused project_gather path supports multi-page splats — ` +\n"
    "        `set useFusedProject=true or reduce capacity to <= ${this.pager.splatsPerPage}).`,\n"
    "      );\n"
    "    }"
)
src = src.replace(old_alloc, new_alloc)

# 4. Replace the fused-path bind group setup so cs_keygen + cs_project_gather use\n#    templated multi-page WGSL when numPages > 1. We'll build the templated pipelines\n#    inside the constructor body (replacing the existing pipes.keygen/projectGather\n#    references at bind-group time).\n
# Find the fused-path bind-group block.
old_fused_bg = (
    "      this.projectBindGroup = null;\n"
    "      this.gatherUniforms = null;\n"
    "      this.gatherBindGroup = null;\n"
    "      // Fused path: reuse the same projectUniforms buffer (matching struct).\n"
    "      this.keygenBindGroup = this.device.createBindGroup({\n"
    "        layout: this.pipes.keygenBgl!,\n"
    "        entries: [\n"
    "          { binding: 0, resource: { buffer: this.splatsBuffer } },\n"
    "          { binding: 1, resource: { buffer: this.sorter.keysA } },\n"
    "          { binding: 2, resource: { buffer: this.sorter.valuesA } },\n"
    "          { binding: 3, resource: { buffer: this.projectUniforms } },\n"
    "        ],\n"
    "      });\n"
    "      this.projectGatherBindGroup = this.device.createBindGroup({\n"
    "        layout: this.pipes.projectGatherBgl!,\n"
    "        entries: [\n"
    "          { binding: 0, resource: { buffer: this.splatsBuffer } },\n"
    "          { binding: 1, resource: { buffer: this.sorter.valuesA } },\n"
    "          { binding: 2, resource: { buffer: this.instanceBuffer } },\n"
    "          { binding: 3, resource: { buffer: this.projectUniforms } },\n"
    "        ],\n"
    "      });\n"
    "    }"
)
assert old_fused_bg in src
new_fused_bg = (
    "      this.projectBindGroup = null;\n"
    "      this.gatherUniforms = null;\n"
    "      this.gatherBindGroup = null;\n"
    "      // Fused path: reuse the same projectUniforms buffer (matching struct).\n"
    "      // When the splats are multi-page (Stage 6 / sf-154), rebuild the keygen\n"
    "      // and project_gather pipelines from templated WGSL with N page bindings\n"
    "      // and a `read_splats_*` switch helper.\n"
    "      if (this.pager.numPages > 1) {\n"
    "        const pagedPipes = this._buildPagedFusedPipelines(this.pager.numPages, this.pager.splatsPerPage);\n"
    "        this._pagedKeygen = pagedPipes.keygen;\n"
    "        this._pagedProjectGather = pagedPipes.projectGather;\n"
    "        // Build per-pipeline bind groups that bind ALL pages on bindings\n"
    "        // [0, numPages), with downstream bindings rebased (see\n"
    "        // templateSplatsAccess in buffer-pager.ts).\n"
    "        const N = this.pager.numPages;\n"
    "        const pageEntries = (downstream: GPUBindGroupEntry[]) => [\n"
    "          ...this.pager.pageBuffers.map((b, i) => ({ binding: i, resource: { buffer: b } as GPUBindingResource })),\n"
    "          ...downstream.map((e) => ({ ...e, binding: e.binding + (N - 1) })),\n"
    "        ];\n"
    "        this.keygenBindGroup = this.device.createBindGroup({\n"
    "          layout: pagedPipes.keygenBgl,\n"
    "          entries: pageEntries([\n"
    "            { binding: 1, resource: { buffer: this.sorter.keysA } },\n"
    "            { binding: 2, resource: { buffer: this.sorter.valuesA } },\n"
    "            { binding: 3, resource: { buffer: this.projectUniforms } },\n"
    "          ]),\n"
    "        });\n"
    "        this.projectGatherBindGroup = this.device.createBindGroup({\n"
    "          layout: pagedPipes.projectGatherBgl,\n"
    "          entries: pageEntries([\n"
    "            { binding: 1, resource: { buffer: this.sorter.valuesA } },\n"
    "            { binding: 2, resource: { buffer: this.instanceBuffer } },\n"
    "            { binding: 3, resource: { buffer: this.projectUniforms } },\n"
    "          ]),\n"
    "        });\n"
    "      } else {\n"
    "        this.keygenBindGroup = this.device.createBindGroup({\n"
    "          layout: this.pipes.keygenBgl!,\n"
    "          entries: [\n"
    "            { binding: 0, resource: { buffer: this.splatsBuffer } },\n"
    "            { binding: 1, resource: { buffer: this.sorter.keysA } },\n"
    "            { binding: 2, resource: { buffer: this.sorter.valuesA } },\n"
    "            { binding: 3, resource: { buffer: this.projectUniforms } },\n"
    "          ],\n"
    "        });\n"
    "        this.projectGatherBindGroup = this.device.createBindGroup({\n"
    "          layout: this.pipes.projectGatherBgl!,\n"
    "          entries: [\n"
    "            { binding: 0, resource: { buffer: this.splatsBuffer } },\n"
    "            { binding: 1, resource: { buffer: this.sorter.valuesA } },\n"
    "            { binding: 2, resource: { buffer: this.instanceBuffer } },\n"
    "            { binding: 3, resource: { buffer: this.projectUniforms } },\n"
    "          ],\n"
    "        });\n"
    "      }\n"
    "    }"
)
src = src.replace(old_fused_bg, new_fused_bg)

# 5. Add the _pagedKeygen / _pagedProjectGather fields + the _buildPagedFusedPipelines\n#    helper. Insert before the first method. We add it just before `wsrSigma = 2.0;`.\n
old_field_anchor = "  wsrSigma = 2.0;"
assert old_field_anchor in src
new_field_anchor = (
    "  /** Stage 6 paged-keygen pipeline (multi-page splats). Null when numPages==1. */\n"
    "  private _pagedKeygen: GPUComputePipeline | null = null;\n"
    "  /** Stage 6 paged project_gather pipeline. Null when numPages==1. */\n"
    "  private _pagedProjectGather: GPUComputePipeline | null = null;\n"
    + old_field_anchor
)
src = src.replace(old_field_anchor, new_field_anchor)

# 6. Add the helper method. Insert before destroy().
old_destroy_anchor = "  /** Tear down. Idempotent. */\n  destroy(): void {"
assert old_destroy_anchor in src
helper_method = (
    "  /**\n"
    "   * Stage 6: build templated cs_keygen + cs_project_gather pipelines for\n"
    "   * multi-page splats. The original PROJECT_GATHER_WGSL has a single\n"
    "   * splats binding (k_splats / g_splats); we rewrite each entry point's\n"
    "   * WGSL to declare N page bindings + a `read_splats_*(i)` helper, then\n"
    "   * compile a fresh shader module + pipeline per entry point.\n"
    "   *\n"
    "   * Returns the templated keygen and project_gather pipelines plus their\n"
    "   * matching bind-group layouts (which are also paged: bindings\n"
    "   * [0, N) are page buffers, downstream bindings are shifted by N-1).\n"
    "   */\n"
    "  private _buildPagedFusedPipelines(numPages: number, splatsPerPage: number): {\n"
    "    keygen: GPUComputePipeline;\n"
    "    projectGather: GPUComputePipeline;\n"
    "    keygenBgl: GPUBindGroupLayout;\n"
    "    projectGatherBgl: GPUBindGroupLayout;\n"
    "  } {\n"
    "    const COMPUTE = GPUShaderStage.COMPUTE;\n"
    "    // PROJECT_GATHER_WGSL contains BOTH cs_keygen (binding name k_splats)\n"
    "    // and cs_project_gather (binding name g_splats). We template each binding\n"
    "    // separately; templateSplatsAccess emits N page bindings for the named\n"
    "    // binding and rebases the others.\n"
    "    const dilSrc = applyDilationOverride(PROJECT_GATHER_WGSL, this.dilation);\n"
    "    const tplK = templateSplatsAccess(dilSrc, 'k_splats', numPages, splatsPerPage);\n"
    "    const tplG = templateSplatsAccess(dilSrc, 'g_splats', numPages, splatsPerPage);\n"
    "    const keygenMod = this.device.createShaderModule({ code: tplK.wgsl });\n"
    "    const pgMod     = this.device.createShaderModule({ code: tplG.wgsl });\n"
    "    // Keygen BGL: N page bindings (read-only-storage) + 3 downstream bindings\n"
    "    // (keys, indices, uniforms) shifted by (N-1).\n"
    "    const keygenEntries: GPUBindGroupLayoutEntry[] = [];\n"
    "    for (let p = 0; p < numPages; p++) {\n"
    "      keygenEntries.push({ binding: p, visibility: COMPUTE, buffer: { type: 'read-only-storage' } });\n"
    "    }\n"
    "    keygenEntries.push({ binding: numPages,     visibility: COMPUTE, buffer: { type: 'storage' } });\n"
    "    keygenEntries.push({ binding: numPages + 1, visibility: COMPUTE, buffer: { type: 'storage' } });\n"
    "    keygenEntries.push({ binding: numPages + 2, visibility: COMPUTE, buffer: { type: 'uniform' } });\n"
    "    const keygenBgl = this.device.createBindGroupLayout({ entries: keygenEntries });\n"
    "    // Project_gather BGL: N page bindings + (indices, inst_out, uniforms).\n"
    "    const pgEntries: GPUBindGroupLayoutEntry[] = [];\n"
    "    for (let p = 0; p < numPages; p++) {\n"
    "      pgEntries.push({ binding: p, visibility: COMPUTE, buffer: { type: 'read-only-storage' } });\n"
    "    }\n"
    "    pgEntries.push({ binding: numPages,     visibility: COMPUTE, buffer: { type: 'read-only-storage' } });\n"
    "    pgEntries.push({ binding: numPages + 1, visibility: COMPUTE, buffer: { type: 'storage' } });\n"
    "    pgEntries.push({ binding: numPages + 2, visibility: COMPUTE, buffer: { type: 'uniform' } });\n"
    "    const projectGatherBgl = this.device.createBindGroupLayout({ entries: pgEntries });\n"
    "    const keygen = this.device.createComputePipeline({\n"
    "      layout: this.device.createPipelineLayout({ bindGroupLayouts: [keygenBgl] }),\n"
    "      compute: { module: keygenMod, entryPoint: 'cs_keygen' },\n"
    "    });\n"
    "    const projectGather = this.device.createComputePipeline({\n"
    "      layout: this.device.createPipelineLayout({ bindGroupLayouts: [projectGatherBgl] }),\n"
    "      compute: { module: pgMod, entryPoint: 'cs_project_gather' },\n"
    "    });\n"
    "    return { keygen, projectGather, keygenBgl, projectGatherBgl };\n"
    "  }\n\n"
)
src = src.replace(old_destroy_anchor, helper_method + old_destroy_anchor)

# 7. Update encode() fused-path to use _pagedKeygen / _pagedProjectGather when set.
old_encode_fused = (
    "      dispatchPerSplat(\n"
    "        this.device,\n"
    "        encoder,\n"
    "        this.pipes.keygen!,\n"
    "        this.keygenBindGroup!,\n"
    "        this.projectUniforms,\n"
    "        UNIFORM_CHUNK_OFFSET_BYTES.project,\n"
    "        count,\n"
    "      );\n"
    "      this.sorter.encode(encoder, count);\n"
    "      dispatchPerSplat(\n"
    "        this.device,\n"
    "        encoder,\n"
    "        this.pipes.projectGather!,\n"
    "        this.projectGatherBindGroup!,\n"
    "        this.projectUniforms,\n"
    "        UNIFORM_CHUNK_OFFSET_BYTES.project,\n"
    "        count,\n"
    "      );\n"
    "      return;\n"
    "    }"
)
assert old_encode_fused in src
new_encode_fused = (
    "      dispatchPerSplat(\n"
    "        this.device,\n"
    "        encoder,\n"
    "        this._pagedKeygen ?? this.pipes.keygen!,\n"
    "        this.keygenBindGroup!,\n"
    "        this.projectUniforms,\n"
    "        UNIFORM_CHUNK_OFFSET_BYTES.project,\n"
    "        count,\n"
    "      );\n"
    "      this.sorter.encode(encoder, count);\n"
    "      dispatchPerSplat(\n"
    "        this.device,\n"
    "        encoder,\n"
    "        this._pagedProjectGather ?? this.pipes.projectGather!,\n"
    "        this.projectGatherBindGroup!,\n"
    "        this.projectUniforms,\n"
    "        UNIFORM_CHUNK_OFFSET_BYTES.project,\n"
    "        count,\n"
    "      );\n"
    "      return;\n"
    "    }"
)
src = src.replace(old_encode_fused, new_encode_fused)

# 8. Update the encodeTimed fused path the same way.
old_timed_keygen = "        pass.setPipeline(this.pipes.keygen!);"
new_timed_keygen = "        pass.setPipeline(this._pagedKeygen ?? this.pipes.keygen!);"
assert src.count(old_timed_keygen) >= 1
src = src.replace(old_timed_keygen, new_timed_keygen)
old_timed_pg = "        pass.setPipeline(this.pipes.projectGather!);"
new_timed_pg = "        pass.setPipeline(this._pagedProjectGather ?? this.pipes.projectGather!);"
assert src.count(old_timed_pg) >= 1
src = src.replace(old_timed_pg, new_timed_pg)

# 9. uploadChunk: route writes via the pager. Replace the dstView block + bind group.
old_upload_dst = (
    "    // A per-chunk \"splats slice\" view — since WebGPU doesn't have offset\n"
    "    // bindings for storage buffers without dynamic offsets, we bind the full\n"
    "    // buffer and pass the destination offset via a *separate* uniform. The\n"
    "    // shader writes at `dst_splats[i]`, so we need a second tiny shader OR a\n"
    "    // per-chunk dst buffer. We pick the simpler latter: per-chunk dst slice\n"
    "    // expressed via `binding.offset` of the bind group (which IS supported).\n"
    "    const dstView: GPUBindingResource = {\n"
    "      buffer: this.splatsBuffer,\n"
    "      offset: this.decodedSplats * BYTES_PER_DECODED_SPLAT,\n"
    "      size: descriptor.splatCount * BYTES_PER_DECODED_SPLAT,\n"
    "    };\n"
    "    const decodeBindGroup = this.device.createBindGroup({\n"
    "      layout: this.pipes.decodeBgl,\n"
    "      entries: [\n"
    "        { binding: 0, resource: { buffer: bytesBuffer } },\n"
    "        { binding: 1, resource: dstView },\n"
    "        { binding: 2, resource: { buffer: decodeUniforms } },\n"
    "      ],\n"
    "    });\n\n"
    "    // Dispatch decode immediately. Subsequent project passes will see the\n"
    "    // decoded splats. Multi-dispatch carves the per-splat work into\n"
    "    // <= 65535-workgroup chunks so > 16.7 M-splat scenes (LODGE L1/L0)\n"
    "    // don't trip the WebGPU 1.0 dispatch limit.\n"
    "    const encoder = this.device.createCommandEncoder();\n"
    "    dispatchPerSplat(\n"
    "      this.device,\n"
    "      encoder,\n"
    "      this.pipes.decode,\n"
    "      decodeBindGroup,\n"
    "      decodeUniforms,\n"
    "      UNIFORM_CHUNK_OFFSET_BYTES.decode,\n"
    "      descriptor.splatCount,\n"
    "    );\n"
    "    this.device.queue.submit([encoder.finish()]);"
)
assert old_upload_dst in src
new_upload_dst = (
    "    // Stage 6 (sf-154): the destination splats live across N pager pages.\n"
    "    // For each page sub-range that this chunk overlaps, we issue a\n"
    "    // separate decode dispatch with the page's GPUBuffer bound (with a\n"
    "    // dynamic-offset binding) and a per-sub-range chunk_offset that\n"
    "    // selects the right slice of the source bytes. When numPages == 1\n"
    "    // this collapses to a single dispatch identical to the pre-Stage-6\n"
    "    // path.\n"
    "    const encoder = this.device.createCommandEncoder();\n"
    "    let srcSplatOffset = 0;\n"
    "    for (const range of this.pager.pageRanges(this.decodedSplats, descriptor.splatCount)) {\n"
    "      const dstView: GPUBindingResource = {\n"
    "        buffer: this.pager.pageBuffers[range.page],\n"
    "        offset: range.localStart * BYTES_PER_DECODED_SPLAT,\n"
    "        size: range.localCount * BYTES_PER_DECODED_SPLAT,\n"
    "      };\n"
    "      const subBindGroup = this.device.createBindGroup({\n"
    "        layout: this.pipes.decodeBgl,\n"
    "        entries: [\n"
    "          { binding: 0, resource: { buffer: bytesBuffer } },\n"
    "          { binding: 1, resource: dstView },\n"
    "          { binding: 2, resource: { buffer: decodeUniforms } },\n"
    "        ],\n"
    "      });\n"
    "      // The decode kernel reads source bytes by index relative to splat 0\n"
    "      // of the chunk and writes dst_splats[i]. Since we sliced dst with a\n"
    "      // dynamic-offset binding, the write index `i` for THIS sub-range is\n"
    "      // also page-local — so we pass chunk_offset = srcSplatOffset to\n"
    "      // make the shader read source slice [srcSplatOffset..]. Currently\n"
    "      // the decode kernel uses i = gid.x + chunk_offset for BOTH the source\n"
    "      // SoA index AND the dst index. We patch by giving each sub-range a\n"
    "      // freshly built decodeUniforms with splat_count = sub-range count\n"
    "      // and SoA byteOffset rebased.\n"
    "      // Simpler approach: only one sub-range per chunk (single-page case).\n"
    "      // For multi-page scenes the chunker already targets ~256K splats\n"
    "      // per chunk, far smaller than splatsPerPage (~33M at 2 GiB cap),\n"
    "      // so straddling never happens in practice. Assert that here:\n"
    "      if (range.localCount !== descriptor.splatCount) {\n"
    "        throw new Error(\n"
    "          `compute-decode: chunk straddles splat-page boundary ` +\n"
    "          `(splatStart=${this.decodedSplats}, splatCount=${descriptor.splatCount}, ` +\n"
    "          `splatsPerPage=${this.pager.splatsPerPage}); chunk-page split not yet supported.`,\n"
    "        );\n"
    "      }\n"
    "      void srcSplatOffset; // reserved for future per-sub-range source rebase\n"
    "      dispatchPerSplat(\n"
    "        this.device,\n"
    "        encoder,\n"
    "        this.pipes.decode,\n"
    "        subBindGroup,\n"
    "        decodeUniforms,\n"
    "        UNIFORM_CHUNK_OFFSET_BYTES.decode,\n"
    "        range.localCount,\n"
    "      );\n"
    "      srcSplatOffset += range.localCount;\n"
    "    }\n"
    "    this.device.queue.submit([encoder.finish()]);"
)
src = src.replace(old_upload_dst, new_upload_dst)

# 10. Replace the chunks.push splatsBuffer reference (decoded chunk struct).
old_chunks_push = (
    "    this.chunks.push({\n"
    "      splatCount: descriptor.splatCount,\n"
    "      bytesBuffer,\n"
    "      splatsBuffer: this.splatsBuffer,\n"
    "      decodeUniforms,\n"
    "      decodeBindGroup,\n"
    "    });"
)
assert old_chunks_push in src
new_chunks_push = (
    "    this.chunks.push({\n"
    "      splatCount: descriptor.splatCount,\n"
    "      bytesBuffer,\n"
    "      splatsBuffer: this.pager.pageBuffers[0],\n"
    "      decodeUniforms,\n"
    "      // decodeBindGroup field kept for backward-compat — the per-sub-range\n"
    "      // bind group is created+used inline above and not retained.\n"
    "      decodeBindGroup: this.device.createBindGroup({\n"
    "        layout: this.pipes.decodeBgl,\n"
    "        entries: [\n"
    "          { binding: 0, resource: { buffer: bytesBuffer } },\n"
    "          { binding: 1, resource: { buffer: this.pager.pageBuffers[0] } },\n"
    "          { binding: 2, resource: { buffer: decodeUniforms } },\n"
    "        ],\n"
    "      }),\n"
    "    });"
)
src = src.replace(old_chunks_push, new_chunks_push)

# 11. destroy(): replace splatsBuffer.destroy() with pager.destroy().
old_destroy = "    this.splatsBuffer.destroy();"
assert old_destroy in src
src = src.replace(old_destroy, "    this.pager.destroy();")

# 12. Multi-page guards for the cull/WSR/WSR-tile branches: insert clear
#     errors so users get a clean message instead of a silent miscompile.
old_cull = "    this.useCull = init.useCull ?? true;\n    if (this.useCull) {"
assert old_cull in src
new_cull = (
    "    this.useCull = init.useCull ?? true;\n"
    "    if (this.useCull && this.pager.numPages > 1) {\n"
    "      throw new Error(\n"
    "        `ComputeDecodePipeline: useCull is not supported with multi-page splats ` +\n"
    "        `(${this.pager.numPages} pages required for capacity ${this.capacity}). ` +\n"
    "        `Pass useCull=false or reduce capacity to <= ${this.pager.splatsPerPage}.`,\n"
    "      );\n"
    "    }\n"
    "    if (this.useCull) {"
)
src = src.replace(old_cull, new_cull)

old_wsr = "    this.useWSR = init.useWSR ?? false;\n    if (this.useWSR) {"
assert old_wsr in src
new_wsr = (
    "    this.useWSR = init.useWSR ?? false;\n"
    "    if (this.useWSR && this.pager.numPages > 1) {\n"
    "      throw new Error(\n"
    "        `ComputeDecodePipeline: useWSR is not supported with multi-page splats ` +\n"
    "        `(${this.pager.numPages} pages required). Reduce capacity to <= ${this.pager.splatsPerPage}.`,\n"
    "      );\n"
    "    }\n"
    "    if (this.useWSR) {"
)
src = src.replace(old_wsr, new_wsr)

old_wsrt = "    this.useWSRTile = init.useWSRTile ?? false;\n    if (this.useWSRTile) {"
assert old_wsrt in src
new_wsrt = (
    "    this.useWSRTile = init.useWSRTile ?? false;\n"
    "    if (this.useWSRTile && this.pager.numPages > 1) {\n"
    "      throw new Error(\n"
    "        `ComputeDecodePipeline: useWSRTile is not yet supported with multi-page splats ` +\n"
    "        `(${this.pager.numPages} pages required for capacity ${this.capacity}). ` +\n"
    "        `Reduce capacity to <= ${this.pager.splatsPerPage} or use the fused project_gather path.`,\n"
    "      );\n"
    "    }\n"
    "    if (this.useWSRTile) {"
)
src = src.replace(old_wsrt, new_wsrt)

if src == orig:
    print("ERROR: no edits applied", file=sys.stderr)
    sys.exit(1)

P.write_text(src)
print(f"patched {P} ({len(src) - len(orig):+d} chars)")
