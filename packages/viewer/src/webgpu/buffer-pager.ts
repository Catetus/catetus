// SPDX-License-Identifier: Apache-2.0
/**
 * Buffer pager for the canonical decoded-splat buffer (Stage 6, sf-154).
 *
 * Background: the WebGPU 1.0 spec caps `maxStorageBufferBindingSize` at
 * `2^31 - 1` bytes (~2 GiB). At 64 B per `DecodedSplat`, that's a hard
 * ceiling of ~33 M splats per single storage binding. Real LODGE scenes
 * (Sweet Corals L0 = 119 M splats × 64 B = 7.62 GB; L1 = 54 M × 64 B =
 * 3.46 GB) exceed that. To render them end-to-end we split the decoded
 * splats across N contiguous "pages", each ≤ adapter.maxBufferSize.
 *
 * Two access patterns are supported:
 *
 *   1. **Per-dispatch contiguous slice** — the per-splat compute kernels
 *      (`cs_decode`, `cs_project`, `cs_keygen`, `cs_cull`, `cs_lod_blend`,
 *      `cs_lod_alpha_reset`, `cs_tile_bin`) all run with `i = gid.x +
 *      chunk_offset` where `i` is monotonic across the whole scene. We
 *      choose a per-page chunk that lies entirely within one page, bind
 *      that page as the splats binding, and dispatch with the slice's
 *      local `chunk_offset`. The shader sees `splats[local_i]` and the
 *      multi-dispatch wrapper handles the per-page chunking.
 *
 *   2. **Random read access** — `cs_project_gather`, `cs_wsr_accumulate`,
 *      `cs_wsr_tile_accumulate` all index `splats[idx]` with `idx` coming
 *      from a sorted-indices or per-tile splat-list buffer. These kernels
 *      get a templated WGSL prelude that emits N storage bindings + a
 *      `read_splats(i: u32) -> DecodedSplat` helper that switches on
 *      `i / SPLATS_PER_PAGE` and loads from the right page. The kernel
 *      body uses `read_splats(idx)` instead of `splats[idx]`.
 *
 * For the read_write `cs_lod_blend` / `cs_lod_alpha_reset` kernels we
 * always pick a per-dispatch slice that lies within one page (pattern 1),
 * so no random write access is needed.
 */

/** Bytes per canonical decoded splat. Mirrors `BYTES_PER_DECODED_SPLAT`. */
const BYTES_PER_DECODED_SPLAT = 64;

/**
 * Compute the page layout for a given total splat count and per-buffer cap.
 *
 * Returns one page per ≤ pageSize splats. Splat-index ranges are half-open:
 * `[pages[k].splatStart, pages[k].splatStart + pages[k].splatCount)`.
 * The last page may be shorter than `splatsPerPage`.
 *
 * @param totalSplats max splat capacity across all pages
 * @param maxBufferBytes per-binding byte cap (typically `device.limits.maxStorageBufferBindingSize`)
 */
export interface PageLayout {
  /** Number of pages. */
  numPages: number;
  /** Splats per page (the last page may have fewer). */
  splatsPerPage: number;
  /** Per-page byte size of the storage buffer. */
  bytesPerPage: number;
  /** Per-page descriptors. */
  pages: { splatStart: number; splatCount: number; byteSize: number }[];
}

export function computePageLayout(totalSplats: number, maxBufferBytes: number): PageLayout {
  if (totalSplats <= 0) {
    return { numPages: 1, splatsPerPage: 1, bytesPerPage: BYTES_PER_DECODED_SPLAT, pages: [{ splatStart: 0, splatCount: 0, byteSize: BYTES_PER_DECODED_SPLAT }] };
  }
  // Round splats-per-page DOWN to a multiple of 256 so per-splat dispatch
  // workgroups don't straddle page boundaries (workgroup_size = 256 in all
  // per-splat kernels).
  let splatsPerPage = Math.floor(maxBufferBytes / BYTES_PER_DECODED_SPLAT);
  splatsPerPage = Math.floor(splatsPerPage / 256) * 256;
  if (splatsPerPage <= 0) {
    throw new Error(`computePageLayout: maxBufferBytes ${maxBufferBytes} too small to hold even one workgroup of splats`);
  }
  const numPages = Math.ceil(totalSplats / splatsPerPage);
  const bytesPerPage = splatsPerPage * BYTES_PER_DECODED_SPLAT;
  const pages: { splatStart: number; splatCount: number; byteSize: number }[] = [];
  for (let i = 0; i < numPages; i++) {
    const start = i * splatsPerPage;
    const count = Math.min(splatsPerPage, totalSplats - start);
    pages.push({ splatStart: start, splatCount: count, byteSize: count * BYTES_PER_DECODED_SPLAT });
  }
  return { numPages, splatsPerPage, bytesPerPage, pages };
}

/**
 * BufferPager — owns N storage buffers and presents per-page slice helpers.
 *
 * Each page is allocated as a separate `GPUBuffer` of size `bytesPerPage`
 * (the last page may be smaller). The pager exposes:
 *
 *   - `pageBuffers`            — direct array of page GPUBuffers for binding.
 *   - `splatToPage(splatIdx)`  — returns `{ page, localSplat }` for a global splat index.
 *   - `pageRanges()`           — iterate per-page splat ranges.
 *   - `writeChunk(splatStart, bytes)` — write a contiguous slice of decoded
 *     splat bytes into the right page (helper for cs_decode's dst writes).
 *   - `pickPageForRange(splatStart, splatCount)` — assert the range lies
 *     within one page; return that page index. Throws if it straddles.
 */
export class BufferPager {
  readonly device: GPUDevice;
  readonly capacity: number;
  readonly layout: PageLayout;
  readonly pageBuffers: GPUBuffer[];

  constructor(device: GPUDevice, capacity: number, maxBufferBytes: number) {
    this.device = device;
    this.capacity = capacity;
    this.layout = computePageLayout(capacity, maxBufferBytes);
    this.pageBuffers = this.layout.pages.map((p) =>
      device.createBuffer({
        size: Math.max(p.byteSize, BYTES_PER_DECODED_SPLAT),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    );
  }

  /** Number of pages allocated. */
  get numPages(): number { return this.layout.numPages; }

  /** Splats per page (last page may have fewer). */
  get splatsPerPage(): number { return this.layout.splatsPerPage; }

  /** Map a global splat index → page + local splat index inside that page. */
  splatToPage(splatIdx: number): { page: number; localSplat: number } {
    const page = Math.floor(splatIdx / this.layout.splatsPerPage);
    const localSplat = splatIdx - page * this.layout.splatsPerPage;
    return { page, localSplat };
  }

  /**
   * Pick the single page that contains `[splatStart, splatStart + splatCount)`.
   * Throws if the range straddles a page boundary.
   */
  pickPageForRange(splatStart: number, splatCount: number): number {
    if (splatCount === 0) return 0;
    const a = this.splatToPage(splatStart);
    const last = splatStart + splatCount - 1;
    const b = this.splatToPage(last);
    if (a.page !== b.page) {
      throw new Error(
        `BufferPager.pickPageForRange: range [${splatStart}, ${splatStart + splatCount}) ` +
        `straddles pages ${a.page}…${b.page} (splatsPerPage=${this.layout.splatsPerPage})`,
      );
    }
    return a.page;
  }

  /**
   * Iterate the per-page sub-ranges of `[splatStart, splatStart + splatCount)`.
   * Yields `{ page, localStart, localCount }` for each page touched.
   */
  *pageRanges(splatStart: number, splatCount: number): Generator<{ page: number; localStart: number; localCount: number }> {
    if (splatCount === 0) return;
    let cursor = splatStart;
    const end = splatStart + splatCount;
    while (cursor < end) {
      const a = this.splatToPage(cursor);
      const pageEnd = (a.page + 1) * this.layout.splatsPerPage;
      const segEnd = Math.min(end, pageEnd);
      const localStart = a.localSplat;
      const localCount = segEnd - cursor;
      yield { page: a.page, localStart, localCount };
      cursor = segEnd;
    }
  }

  /**
   * Write a contiguous byte buffer of decoded splats starting at global
   * splat index `splatStart`. Splits across pages as needed.
   *
   * `bytes` must be `splatCount * BYTES_PER_DECODED_SPLAT` long.
   */
  writeSplats(splatStart: number, splatCount: number, bytes: Uint8Array): void {
    if (splatCount * BYTES_PER_DECODED_SPLAT !== bytes.byteLength) {
      throw new Error(`BufferPager.writeSplats: bytes.byteLength ${bytes.byteLength} != ${splatCount * BYTES_PER_DECODED_SPLAT}`);
    }
    let srcOffset = 0;
    for (const { page, localStart, localCount } of this.pageRanges(splatStart, splatCount)) {
      const dstByteOffset = localStart * BYTES_PER_DECODED_SPLAT;
      const chunkBytes = localCount * BYTES_PER_DECODED_SPLAT;
      this.device.queue.writeBuffer(
        this.pageBuffers[page],
        dstByteOffset,
        bytes.buffer,
        bytes.byteOffset + srcOffset,
        chunkBytes,
      );
      srcOffset += chunkBytes;
    }
  }

  /** Tear down. Idempotent. */
  destroy(): void {
    for (const b of this.pageBuffers) b.destroy();
    this.pageBuffers.length = 0;
  }
}

/**
 * WGSL templating for random-read-access kernels.
 *
 * Replaces the canonical single-binding declaration of the splats array:
 *
 *   @group(0) @binding(0) var<storage, read> {NAME} : array<DecodedSplat>;
 *
 * with N storage bindings (one per page) and a `read_splats_{NAME}(i: u32)`
 * helper function that switches on `page = i / SPLATS_PER_PAGE` and loads
 * from the appropriate page binding.
 *
 * The kernel body must be hand-edited to call `read_splats_{NAME}(i)`
 * instead of `{NAME}[i]` — that text substitution is done as a separate
 * regex pass below to keep the templating focused on the bindings.
 *
 * Bindings other than the splats array are renumbered: the first binding
 * after the splats pages takes binding `(numPages)`, the next takes
 * `(numPages + 1)`, etc. The templating function returns the rewritten
 * WGSL plus a host-side mapping of original→new binding numbers so the
 * TS-side `createBindGroupLayout` / `createBindGroup` calls can be updated
 * in lockstep.
 *
 * @param wgsl              source WGSL with a single splats binding.
 * @param splatsBindingName variable name of the splats binding (e.g. `g_splats`,
 *                          `splats`, `k_splats`).
 * @param numPages          number of pages.
 * @param splatsPerPage     splats per page (must match the BufferPager).
 *
 * Returns `{ wgsl, splatsBindings, otherBindingsRebased }`. The caller is
 * responsible for rebasing its bind-group-layout entries to the returned
 * binding numbers.
 */
export interface TemplateResult {
  wgsl: string;
  /** New binding indices for the splats pages (length = numPages). */
  splatsBindings: number[];
  /** Map original binding index → new binding index for non-splats bindings. */
  rebasedBindings: Map<number, number>;
}

export function templateSplatsAccess(
  wgsl: string,
  splatsBindingName: string,
  numPages: number,
  splatsPerPage: number,
): TemplateResult {
  // 1. Find the original splats binding declaration. Format:
  //      @group(0) @binding(N) var<storage, read[_write]> NAME : array<DecodedSplat>;
  const re = new RegExp(
    String.raw`@group\(0\)\s+@binding\((\d+)\)\s+var<storage,\s*read(?:_write)?>\s+` +
    splatsBindingName + String.raw`\s*:\s*array<DecodedSplat>\s*;`,
  );
  const m = wgsl.match(re);
  if (!m) {
    throw new Error(`templateSplatsAccess: could not find splats binding "${splatsBindingName}"`);
  }
  const originalBinding = parseInt(m[1], 10);

  // 2. Collect all OTHER bindings to rebase. The new layout puts the N
  //    splats bindings at [originalBinding, originalBinding + N), and
  //    every binding originally >= originalBinding + 1 shifts up by N - 1.
  const allBindings: number[] = [];
  const allRe = /@group\(0\)\s+@binding\((\d+)\)/g;
  let am: RegExpExecArray | null;
  while ((am = allRe.exec(wgsl)) !== null) {
    allBindings.push(parseInt(am[1], 10));
  }
  const rebasedBindings = new Map<number, number>();
  for (const b of allBindings) {
    if (b === originalBinding) continue;
    if (b > originalBinding) {
      rebasedBindings.set(b, b + (numPages - 1));
    } else {
      rebasedBindings.set(b, b);
    }
  }

  // 3. Build the new splats bindings + read_splats helper.
  const splatsBindings: number[] = [];
  const pageDecls: string[] = [];
  for (let p = 0; p < numPages; p++) {
    const b = originalBinding + p;
    splatsBindings.push(b);
    pageDecls.push(
      `@group(0) @binding(${b}) var<storage, read> ${splatsBindingName}_p${p} : array<DecodedSplat>;`,
    );
  }
  const helperName = `read_splats_${splatsBindingName}`;
  const cases: string[] = [];
  for (let p = 0; p < numPages; p++) {
    cases.push(`    case ${p}u: { return ${splatsBindingName}_p${p}[off]; }`);
  }
  const helper = `
const SPLATS_PER_PAGE_${splatsBindingName} : u32 = ${splatsPerPage}u;
fn ${helperName}(i: u32) -> DecodedSplat {
  let page = i / SPLATS_PER_PAGE_${splatsBindingName};
  let off  = i - page * SPLATS_PER_PAGE_${splatsBindingName};
  switch(page) {
${cases.join('\n')}
    default: { return ${splatsBindingName}_p0[0u]; }
  }
}
`;

  // 4. Replace the original splats binding line with the new pages + helper.
  let newWgsl = wgsl.replace(re, pageDecls.join('\n') + helper);

  // 5. Rebase the other binding numbers in the WGSL source.
  //    Walk in reverse so we don't double-rebase shifted indices.
  const sortedKeys = Array.from(rebasedBindings.keys()).sort((a, b) => b - a);
  for (const oldB of sortedKeys) {
    const newB = rebasedBindings.get(oldB)!;
    if (oldB === newB) continue;
    const reB = new RegExp(String.raw`(@group\(0\)\s+@binding\()` + oldB + String.raw`(\))`, 'g');
    // To avoid catching bindings that happened to hit oldB during prior shifts,
    // we tag with a sentinel then untag afterwards.
    newWgsl = newWgsl.replace(reB, `$1__SF_TMP_${oldB}_TO_${newB}__$2`);
  }
  for (const oldB of sortedKeys) {
    const newB = rebasedBindings.get(oldB)!;
    if (oldB === newB) continue;
    newWgsl = newWgsl.replaceAll(`__SF_TMP_${oldB}_TO_${newB}__`, String(newB));
  }

  // 6. Replace `NAME[i]` indexing with `helperName(i)` calls.
  //    We need to be careful: the WGSL still has `NAME_p0[off]` etc inside the
  //    helper body. Use a regex that requires NAME NOT followed by `_p`.
  const indexRe = new RegExp(
    String.raw`\b` + splatsBindingName + String.raw`\s*\[\s*([^\[\]]+?)\s*\]`,
    'g',
  );
  newWgsl = newWgsl.replace(indexRe, (_match, idxExpr) => {
    // Skip matches inside the helper block (the helper's `_p0[off]` form
    // doesn't trigger this regex anyway because of the `_p0` suffix, but
    // be defensive).
    return `${helperName}(${idxExpr})`;
  });

  return { wgsl: newWgsl, splatsBindings, rebasedBindings };
}
