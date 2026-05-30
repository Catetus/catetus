// Bundled corpus data — minimal slice used by list_scenes / get_scene / predict_quality.
// Real, full payloads live in src/resources/assets/ owned by MCP-RESOURCES agent.
// We keep a tiny inlined copy here so the server-core can ship standalone.

export interface SceneRecord {
  id: string;
  name: string;
  class: string;
  license: string;
  splatCount: number;
  bytesBaseline: number;
  shDegree: number;
  hash: string;
  summary: string;
  leaderboardRef?: string;
  bbox?: { min: [number, number, number]; max: [number, number, number] };
  perPreset?: Record<string, { bytesOut: number; ratio: number; psnr?: number; meanDeltaE94?: number; mlScore: number | null }>;
  leaderboardRow?: {
    sog_mb: number;
    sf_mb: number;
    t21r_mb: number;
    v52_mb: number;
    sog_psnr: number;
    sf_psnr: number;
    t21r_psnr: number;
    v52_psnr: number;
    delta_v52_minus_sog: number;
    delta_v52_minus_sf: number;
  };
  fixtureUri?: string;
}

// Canonical-11 (Inria 3DGS leaderboard corpus). Numbers per Appendix A of ARCHITECTURE.md.
export const CANONICAL_11_SCENES: SceneRecord[] = [
  mk("bonsai", "indoor", 1244819, 230_400_000, 3, "Indoor bonsai plant; benchmark workhorse."),
  mk("counter", "indoor", 1175687, 218_300_000, 3, "Kitchen counter, Mip-NeRF 360 source."),
  mk("kitchen", "indoor", 1788243, 332_100_000, 3, "Kitchen scene with reflective surfaces."),
  mk("room", "indoor", 1548960, 287_500_000, 3, "Indoor room with furniture."),
  mk("garden", "outdoor", 1722000, 319_900_000, 3, "Outdoor garden, well-lit."),
  mk("stump", "outdoor", 1124060, 208_700_000, 3, "Forest stump close-up."),
  mk("bicycle", "outdoor", 1620010, 300_900_000, 3, "Outdoor bicycle hero scene."),
  mk("flowers", "outdoor", 1410012, 261_900_000, 3, "Outdoor flower bed."),
  mk("treehill", "outdoor", 1450000, 269_300_000, 3, "Tree-on-hill outdoor scene."),
  mk("playroom", "indoor", 1900000, 352_900_000, 3, "Deep Blending playroom — V5.2 best-case (+34.6 dB)."),
  mk("drjohnson", "indoor", 1830000, 339_900_000, 3, "Deep Blending Dr Johnson scene."),
];

// SplatBench-v0 minimal stub. Real catalog comes from SplatForge/benches/reports/splatbench-v0.json.
export const SPLATBENCH_V0_SCENES: SceneRecord[] = [
  mk("sb-bonsai", "indoor", 1244819, 230_400_000, 3, "SplatBench bonsai."),
  mk("sb-garden", "outdoor", 1722000, 319_900_000, 3, "SplatBench garden."),
  mk("sb-room", "indoor", 1548960, 287_500_000, 3, "SplatBench room."),
];

function mk(
  id: string,
  cls: string,
  splats: number,
  bytes: number,
  sh: number,
  summary: string,
): SceneRecord {
  return {
    id,
    name: id,
    class: cls,
    license: "custom",
    splatCount: splats,
    bytesBaseline: bytes,
    shDegree: sh,
    hash: `blake3:${id.padEnd(64, "0")}`,
    summary,
    leaderboardRef: `catetus://bench/canonical-11#${id}`,
    fixtureUri: `catetus://scene/${id}`,
  };
}
