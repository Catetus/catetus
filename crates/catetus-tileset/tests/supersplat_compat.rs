//! Verifies our `LodMeta` types deserialize a real SuperSplat `lod-meta.json`
//! fragment and that our emitter produces a manifest in the same schema family.
//!
//! NOTE: SuperSplat's real `lod-meta.json` stores `lods` as a JSON *object*
//! keyed by `"0".."6"` with `{file, offset, count}` entries (verified against
//! the Koriyama Castle manifest at
//! d28zzqy0iyovbz.cloudfront.net/b11e45d1/v1/lod-meta.json). Our `LodMeta`
//! type uses a `lods: Vec<{file,count}>` array — the same information in a
//! flatter shape. The fragment below uses the array shape our emitter writes;
//! a full byte-for-byte SuperSplat transcoder (object-keyed `lods` + per-LOD
//! `offset` into a shared tile file) is a roadmap item (STATUS.md §Interop).

use catetus_core::ir::{Color, Splat, SplatScene};
use catetus_tileset::{
    plan_tileset, write_tileset, LodMeta, LodMetaNode, OctreeConfig, SfTileCodec, TilesetConfig,
};

const SAMPLE: &str = r#"{
  "lodLevels": 7,
  "environment": "env/meta.json",
  "filenames": [
    "2_0/meta.json",
    "0_0/meta.json",
    "1_0/meta.json"
  ],
  "tree": {
    "bound": { "min": [-254.9435, -26.79508, -405.4218],
               "max": [224.0617, 42.23647, 181.5972] },
    "children": [
      {
        "bound": { "min": [-254.9435, -26.79508, -405.4218],
                   "max": [-15.44089, 42.23647, -111.9123] },
        "lods": [
          { "file": 0, "count": 7456 },
          { "file": 1, "count": 31960 },
          { "file": 2, "count": 124648 }
        ],
        "children": []
      }
    ]
  }
}"#;

#[test]
fn deserializes_lod_meta_in_supersplat_shape() {
    let meta: LodMeta = serde_json::from_str(SAMPLE).expect("parse sample");
    assert_eq!(meta.lod_levels, 7);
    assert_eq!(meta.environment.as_deref(), Some("env/meta.json"));
    assert_eq!(meta.filenames.len(), 3);
    assert!(meta.tree.lods.is_empty()); // interior root, no LODs
    assert_eq!(meta.tree.children.len(), 1);
    let child = &meta.tree.children[0];
    assert_eq!(child.lods.len(), 3);
    assert_eq!(child.lods[0].file, 0);
    assert_eq!(child.lods[2].count, 124648);
}

#[test]
fn our_emitter_output_reparses_as_lod_meta() {
    let mut scene = SplatScene::new();
    for x in 0..20 {
        for y in 0..20 {
            for z in 0..20 {
                scene.splats.push(Splat {
                    position: [x as f32, y as f32, z as f32],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.5, 0.5, 0.5],
                    opacity: 0.6,
                    color: Color::Rgb([0.5, 0.5, 0.5]),
                });
            }
        }
    }
    let cfg = TilesetConfig {
        octree: OctreeConfig { max_depth: 4, max_splats_per_leaf: 512 },
        lods_per_node: 3,
        coarsest_target: 100,
        proxy_cap: 20_000,
    };
    let plan = plan_tileset(&scene, &cfg).unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_tileset(&plan, &SfTileCodec, dir.path()).unwrap();

    let txt = std::fs::read_to_string(dir.path().join("lod-meta.json")).unwrap();
    let meta: LodMeta = serde_json::from_str(&txt).unwrap();
    assert!(meta.lod_levels >= 2);
    assert!(!meta.tree.children.is_empty());
    fn check(n: &LodMetaNode) {
        if !n.lods.is_empty() {
            assert_eq!(n.lods.len(), 3);
            for w in n.lods.windows(2) {
                assert!(w[1].count >= w[0].count);
            }
        }
        for c in &n.children {
            check(c);
        }
    }
    check(&meta.tree);
}
