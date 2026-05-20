/**
 * Format dispatch: takes a bag of files (or URL+bytes) and returns a SplatScene.
 *
 * Detection order:
 *   1. Extension on the primary file. (.ply / .splat / .sog / .glb)
 *   2. Magic-byte sniff if the extension is unknown.
 *
 * Sidecar handling for .glb:
 *   - If a `.shpal` is in the file set, attach it as `<glbname>.shpal`.
 *   - Otherwise the SF-GLB loader auto-fetches the sibling from `baseUrl`.
 */
import { isLikely3DGSPly, loadPly } from './ply.js';
import { loadSplat } from './splat.js';
import { loadSog } from './sog.js';
import { loadSfGlb } from './sf-glb.js';
import { type SplatScene } from '../splat-scene.js';

export interface NamedBytes { name: string; bytes: Uint8Array; }

/** Load from a bag of dropped/picked files. Picks the splat file and bundles
 *  any sibling sidecars. */
export async function loadFromFiles(files: NamedBytes[]): Promise<SplatScene> {
  if (files.length === 0) throw new Error('no files supplied');
  // Pick the primary splat file (anything that isn't .shpal/.v5tail).
  const primary = files.find((f) => !isSidecarName(f.name)) ?? files[0];
  const sidecars: Record<string, Uint8Array> = {};
  for (const f of files) {
    if (f === primary) continue;
    sidecars[f.name] = f.bytes;
  }
  return loadByName(primary, sidecars);
}

/** Load from a single (url-resolved) name+bytes pair. `baseUrl` enables sidecar
 *  auto-fetch for GLBs that reference an external `.shpal`. */
export async function loadFromUrl(
  url: string,
  bytes: Uint8Array,
): Promise<SplatScene> {
  const fname = url.split('/').pop()!.split('?')[0];
  return loadByName({ name: fname, bytes }, {}, url);
}

async function loadByName(
  primary: NamedBytes,
  sidecarsByName: Record<string, Uint8Array>,
  baseUrl?: string,
): Promise<SplatScene> {
  const ext = detectFormat(primary);
  switch (ext) {
    case 'ply':
      return loadPly(primary.bytes, primary.name);
    case 'splat':
      return loadSplat(primary.bytes, primary.name);
    case 'sog': {
      const sidecars: Record<string, Uint8Array> = {};
      for (const [name, bytes] of Object.entries(sidecarsByName)) {
        sidecars[name] = bytes;
        const base = name.split('/').pop()!;
        if (base !== name) sidecars[base] = bytes;
      }
      return loadSog(primary.bytes, primary.name, { sidecars, baseUrl });
    }
    case 'glb': {
      // Build a sidecar lookup keyed by the basename the GLB's extension
      // references. The GLB uses just the filename (no path), so we map every
      // dropped .shpal under its own basename.
      const sidecars: Record<string, Uint8Array> = {};
      for (const [name, bytes] of Object.entries(sidecarsByName)) {
        sidecars[name] = bytes;
        // Also strip directories so the GLB's `uri: "scene.glb.shpal"` matches.
        const base = name.split('/').pop()!;
        if (base !== name) sidecars[base] = bytes;
      }
      return loadSfGlb(primary.bytes, primary.name, { sidecars, baseUrl });
    }
    default:
      throw new Error(`dispatch: unknown format for "${primary.name}"`);
  }
}

function isSidecarName(name: string): boolean {
  return /\.(shpal|v5tail|bin)$/i.test(name);
}

function detectFormat(f: NamedBytes): 'ply' | 'splat' | 'sog' | 'glb' | null {
  const lower = f.name.toLowerCase();
  if (lower.endsWith('.ply')) return 'ply';
  if (lower.endsWith('.splat')) return 'splat';
  if (lower.endsWith('.sog')) return 'sog';
  if (lower.endsWith('.glb')) return 'glb';

  // Magic-byte fallback.
  const b = f.bytes;
  // GLB: "glTF" (0x46546C67 LE).
  if (b.length >= 4 && b[0] === 0x67 && b[1] === 0x6C && b[2] === 0x54 && b[3] === 0x46) return 'glb';
  // ZIP (SOG): "PK\x03\x04".
  if (b.length >= 4 && b[0] === 0x50 && b[1] === 0x4B && b[2] === 0x03 && b[3] === 0x04) return 'sog';
  // PLY: "ply\n".
  if (b.length >= 4 && b[0] === 0x70 && b[1] === 0x6C && b[2] === 0x79) {
    return isLikely3DGSPly(b) ? 'ply' : 'ply'; // either way the PLY parser will run.
  }
  // .splat heuristic: 32-byte multiples + plausible float positions in the first record.
  if (b.length >= 32 && b.length % 32 === 0) {
    const dv = new DataView(b.buffer, b.byteOffset, 32);
    const x = dv.getFloat32(0, true);
    if (Number.isFinite(x) && Math.abs(x) < 1e6) return 'splat';
  }
  return null;
}
