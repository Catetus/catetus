//! Bit-exact equivalence tests for the hoisted-offset binary decoder.
//!
//! Strategy: build a synthetic PLY in memory with each field-ordering
//! variant we want to support, decode it with `read_ply_bytes`, then
//! independently decode the same bytes via a deliberately-naive reference
//! parser (per-scalar `read_f32_le`, no offset hoisting). Assert that the
//! resulting `SplatScene`s are exactly equal.
//!
//! These tests cover every required correctness property of the new fast
//! path that isn't already covered by `ply_roundtrip.rs`:
//!   * Multiple Inria-flavour property orderings (DC interleaved with
//!     f_rest vs separate, normals present vs absent, opacity before/after
//!     scale, etc.).
//!   * Large vertex counts that cross the parallel-decode threshold
//!     (PARALLEL_THRESHOLD = 256 * 1024 splats).
//!   * Malformed headers (truncated body, missing required field).

use catetus_core::{Color, Splat, SplatScene};
use catetus_ply::{read_ply_bytes, PlyError};

#[derive(Clone, Copy)]
struct PropSpec {
    name: &'static str,
    /// Synthetic value generator: `(splat_index) -> raw_value_on_disk`.
    /// We use deterministic noise so each row has distinct bytes.
    gen: fn(usize) -> f32,
}

/// Build a synthetic binary-LE PLY with the requested property ordering.
fn build_ply(n: usize, props: &[PropSpec]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64 + props.len() * 32 + n * props.len() * 4);
    buf.extend_from_slice(b"ply\n");
    buf.extend_from_slice(b"format binary_little_endian 1.0\n");
    buf.extend_from_slice(format!("element vertex {n}\n").as_bytes());
    for p in props {
        buf.extend_from_slice(format!("property float {}\n", p.name).as_bytes());
    }
    buf.extend_from_slice(b"end_header\n");
    for i in 0..n {
        for p in props {
            buf.extend_from_slice(&(p.gen)(i).to_le_bytes());
        }
    }
    buf
}

/// Reference decoder: scan the body left-to-right one f32 at a time, build
/// a name→value map per row, then synthesise the IR. Deliberately slow but
/// trivially-correct.
fn decode_reference(bytes: &[u8], props: &[PropSpec], n: usize) -> SplatScene {
    // Find end_header — we synthesised this so we know it's there.
    let header_end = find_end_header(bytes);
    let body = &bytes[header_end..];
    let stride = props.len() * 4;

    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        let row = &body[i * stride..i * stride + stride];
        let mut row_vals: Vec<(&str, f32)> = Vec::with_capacity(props.len());
        for (k, p) in props.iter().enumerate() {
            let b: [u8; 4] = row[k * 4..k * 4 + 4].try_into().unwrap();
            row_vals.push((p.name, f32::from_le_bytes(b)));
        }
        let get = |name: &str| row_vals.iter().find(|(n, _)| *n == name).map(|(_, v)| *v);
        let pos = [get("x").unwrap(), get("y").unwrap(), get("z").unwrap()];
        let rw = get("rot_0").unwrap();
        let rx = get("rot_1").unwrap();
        let ry = get("rot_2").unwrap();
        let rz = get("rot_3").unwrap();
        let rot = normalize_quat([rx, ry, rz, rw]);
        let scale = [
            get("scale_0").unwrap().exp(),
            get("scale_1").unwrap().exp(),
            get("scale_2").unwrap().exp(),
        ];
        let opacity = sigmoid(get("opacity").unwrap());
        let dc = [
            get("f_dc_0").unwrap(),
            get("f_dc_1").unwrap(),
            get("f_dc_2").unwrap(),
        ];
        let f_rest: Vec<(usize, f32)> = props
            .iter()
            .enumerate()
            .filter_map(|(k, p)| {
                p.name
                    .strip_prefix("f_rest_")
                    .and_then(|s| s.parse::<usize>().ok())
                    .map(|_idx| {
                        let b: [u8; 4] = row[k * 4..k * 4 + 4].try_into().unwrap();
                        (k, f32::from_le_bytes(b))
                    })
            })
            .collect();
        let color = if f_rest.is_empty() {
            Color::Rgb(dc)
        } else {
            let rest_per_channel = f_rest.len() / 3;
            let total = rest_per_channel + 1;
            let degree = match total {
                1 => 0,
                4 => 1,
                9 => 2,
                16 => 3,
                _ => 0,
            };
            let mut coeffs = Vec::with_capacity(3 * total);
            coeffs.extend_from_slice(&dc);
            // f_rest appears in property-declaration order — preserve that.
            for (_, v) in &f_rest {
                coeffs.push(*v);
            }
            Color::Sh { degree, coeffs }
        };
        splats.push(Splat {
            position: pos,
            rotation: rot,
            scale,
            opacity,
            color,
        });
    }
    SplatScene {
        splats,
        coordinate_system: Default::default(),
        semantic_labels: None,
        temporal_mode: catetus_core::TemporalMode::Static,
        lods: None,
        codecgs: None,
    }
}

fn find_end_header(bytes: &[u8]) -> usize {
    let needle = b"end_header\n";
    let mut i = 0;
    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            return i + needle.len();
        }
        i += 1;
    }
    panic!("no end_header in synthetic ply");
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n == 0.0 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

// ---------- value generators (deterministic, distinct per row) ----------

fn gen_x(i: usize) -> f32 {
    (i as f32) * 0.001
}
fn gen_y(i: usize) -> f32 {
    (i as f32) * -0.002 + 1.0
}
fn gen_z(i: usize) -> f32 {
    (i as f32).sqrt() * 0.5
}
fn gen_norm(i: usize) -> f32 {
    if i % 2 == 0 {
        0.0
    } else {
        1.0
    }
}
fn gen_scale_0(i: usize) -> f32 {
    -2.0 + (i as f32) * 1e-6
}
fn gen_scale_1(i: usize) -> f32 {
    -1.5 + (i as f32) * 1e-6
}
fn gen_scale_2(i: usize) -> f32 {
    -1.0 + (i as f32) * 1e-6
}
fn gen_rot_0(i: usize) -> f32 {
    // w
    1.0 - (i as f32) * 1e-7
}
fn gen_rot_1(i: usize) -> f32 {
    (i as f32) * 1e-5
}
fn gen_rot_2(i: usize) -> f32 {
    (i as f32) * -1e-5
}
fn gen_rot_3(i: usize) -> f32 {
    (i as f32) * 2e-5
}
fn gen_opacity(i: usize) -> f32 {
    // logit; sigmoid will map into (0, 1)
    -3.0 + (i as f32) * 1e-4
}
fn gen_dc_0(_i: usize) -> f32 {
    0.1
}
fn gen_dc_1(_i: usize) -> f32 {
    0.2
}
fn gen_dc_2(_i: usize) -> f32 {
    0.3
}
fn gen_f_rest_k(k: usize) -> fn(usize) -> f32 {
    // Cheap synth: encode k into the value so each f_rest_k channel is
    // distinct and deterministic per row.
    match k {
        0 => |i| (i as f32) * 1e-3,
        1 => |i| (i as f32) * 2e-3,
        2 => |i| (i as f32) * 3e-3,
        3 => |i| (i as f32) * 4e-3,
        4 => |i| (i as f32) * 5e-3,
        5 => |i| (i as f32) * 6e-3,
        6 => |i| (i as f32) * 7e-3,
        7 => |i| (i as f32) * 8e-3,
        8 => |i| (i as f32) * 9e-3,
        _ => |i| (i as f32) * 0.01,
    }
}

// ---------- ordering variants ----------

fn ordering_minimal() -> Vec<PropSpec> {
    vec![
        PropSpec {
            name: "x",
            gen: gen_x,
        },
        PropSpec {
            name: "y",
            gen: gen_y,
        },
        PropSpec {
            name: "z",
            gen: gen_z,
        },
        PropSpec {
            name: "scale_0",
            gen: gen_scale_0,
        },
        PropSpec {
            name: "scale_1",
            gen: gen_scale_1,
        },
        PropSpec {
            name: "scale_2",
            gen: gen_scale_2,
        },
        PropSpec {
            name: "rot_0",
            gen: gen_rot_0,
        },
        PropSpec {
            name: "rot_1",
            gen: gen_rot_1,
        },
        PropSpec {
            name: "rot_2",
            gen: gen_rot_2,
        },
        PropSpec {
            name: "rot_3",
            gen: gen_rot_3,
        },
        PropSpec {
            name: "opacity",
            gen: gen_opacity,
        },
        PropSpec {
            name: "f_dc_0",
            gen: gen_dc_0,
        },
        PropSpec {
            name: "f_dc_1",
            gen: gen_dc_1,
        },
        PropSpec {
            name: "f_dc_2",
            gen: gen_dc_2,
        },
    ]
}

/// Inria-style ordering used by gsplat / nerfstudio: normals between
/// position and DC, f_rest after DC, opacity+scale+rot at the end.
fn ordering_inria_degree1() -> Vec<PropSpec> {
    let mut v = vec![
        PropSpec {
            name: "x",
            gen: gen_x,
        },
        PropSpec {
            name: "y",
            gen: gen_y,
        },
        PropSpec {
            name: "z",
            gen: gen_z,
        },
        PropSpec {
            name: "nx",
            gen: gen_norm,
        },
        PropSpec {
            name: "ny",
            gen: gen_norm,
        },
        PropSpec {
            name: "nz",
            gen: gen_norm,
        },
        PropSpec {
            name: "f_dc_0",
            gen: gen_dc_0,
        },
        PropSpec {
            name: "f_dc_1",
            gen: gen_dc_1,
        },
        PropSpec {
            name: "f_dc_2",
            gen: gen_dc_2,
        },
    ];
    // SH degree 1 = 3 DC + 9 rest (per-channel: 1+3 = 4 = (1+1)^2). So 9 f_rest.
    for k in 0..9 {
        let name: &'static str = Box::leak(format!("f_rest_{k}").into_boxed_str());
        v.push(PropSpec {
            name,
            gen: gen_f_rest_k(k),
        });
    }
    v.push(PropSpec {
        name: "opacity",
        gen: gen_opacity,
    });
    v.push(PropSpec {
        name: "scale_0",
        gen: gen_scale_0,
    });
    v.push(PropSpec {
        name: "scale_1",
        gen: gen_scale_1,
    });
    v.push(PropSpec {
        name: "scale_2",
        gen: gen_scale_2,
    });
    v.push(PropSpec {
        name: "rot_0",
        gen: gen_rot_0,
    });
    v.push(PropSpec {
        name: "rot_1",
        gen: gen_rot_1,
    });
    v.push(PropSpec {
        name: "rot_2",
        gen: gen_rot_2,
    });
    v.push(PropSpec {
        name: "rot_3",
        gen: gen_rot_3,
    });
    v
}

/// Inria-style ordering with f_rest INTERLEAVED with f_dc: some splat
/// pipelines emit f_dc_0, f_rest_0, f_rest_1, f_rest_2, f_dc_1, f_rest_3, ...
/// This is unusual but valid PLY. We just need to handle whatever order the
/// header declares.
fn ordering_shuffled_dc_rest_degree1() -> Vec<PropSpec> {
    vec![
        PropSpec {
            name: "opacity",
            gen: gen_opacity,
        },
        PropSpec {
            name: "x",
            gen: gen_x,
        },
        PropSpec {
            name: "f_rest_0",
            gen: gen_f_rest_k(0),
        },
        PropSpec {
            name: "y",
            gen: gen_y,
        },
        PropSpec {
            name: "f_dc_0",
            gen: gen_dc_0,
        },
        PropSpec {
            name: "rot_0",
            gen: gen_rot_0,
        },
        PropSpec {
            name: "f_rest_1",
            gen: gen_f_rest_k(1),
        },
        PropSpec {
            name: "z",
            gen: gen_z,
        },
        PropSpec {
            name: "f_dc_1",
            gen: gen_dc_1,
        },
        PropSpec {
            name: "rot_1",
            gen: gen_rot_1,
        },
        PropSpec {
            name: "f_rest_2",
            gen: gen_f_rest_k(2),
        },
        PropSpec {
            name: "scale_0",
            gen: gen_scale_0,
        },
        PropSpec {
            name: "f_dc_2",
            gen: gen_dc_2,
        },
        PropSpec {
            name: "rot_2",
            gen: gen_rot_2,
        },
        PropSpec {
            name: "f_rest_3",
            gen: gen_f_rest_k(3),
        },
        PropSpec {
            name: "scale_1",
            gen: gen_scale_1,
        },
        PropSpec {
            name: "f_rest_4",
            gen: gen_f_rest_k(4),
        },
        PropSpec {
            name: "rot_3",
            gen: gen_rot_3,
        },
        PropSpec {
            name: "f_rest_5",
            gen: gen_f_rest_k(5),
        },
        PropSpec {
            name: "scale_2",
            gen: gen_scale_2,
        },
        PropSpec {
            name: "f_rest_6",
            gen: gen_f_rest_k(6),
        },
        PropSpec {
            name: "f_rest_7",
            gen: gen_f_rest_k(7),
        },
        PropSpec {
            name: "f_rest_8",
            gen: gen_f_rest_k(8),
        },
    ]
}

fn assert_scenes_equal(a: &SplatScene, b: &SplatScene) {
    assert_eq!(a.splats.len(), b.splats.len(), "splat count");
    for (i, (sa, sb)) in a.splats.iter().zip(b.splats.iter()).enumerate() {
        assert_eq!(sa.position, sb.position, "position mismatch at i={i}");
        assert_eq!(sa.rotation, sb.rotation, "rotation mismatch at i={i}");
        assert_eq!(sa.scale, sb.scale, "scale mismatch at i={i}");
        assert_eq!(sa.opacity, sb.opacity, "opacity mismatch at i={i}");
        match (&sa.color, &sb.color) {
            (Color::Rgb(x), Color::Rgb(y)) => assert_eq!(x, y, "rgb mismatch at i={i}"),
            (
                Color::Sh {
                    degree: da,
                    coeffs: ca,
                },
                Color::Sh {
                    degree: db,
                    coeffs: cb,
                },
            ) => {
                assert_eq!(da, db, "sh degree at i={i}");
                assert_eq!(ca, cb, "sh coeffs at i={i}");
            }
            _ => panic!("color kind mismatch at i={i}"),
        }
    }
}

#[test]
fn minimal_ordering_matches_reference() {
    let props = ordering_minimal();
    let bytes = build_ply(64, &props);
    let fast = read_ply_bytes(&bytes).unwrap();
    let slow = decode_reference(&bytes, &props, 64);
    assert_scenes_equal(&fast, &slow);
}

#[test]
fn inria_degree1_matches_reference() {
    let props = ordering_inria_degree1();
    let bytes = build_ply(128, &props);
    let fast = read_ply_bytes(&bytes).unwrap();
    let slow = decode_reference(&bytes, &props, 128);
    assert_scenes_equal(&fast, &slow);
}

#[test]
fn shuffled_dc_rest_matches_reference() {
    let props = ordering_shuffled_dc_rest_degree1();
    let bytes = build_ply(32, &props);
    let fast = read_ply_bytes(&bytes).unwrap();
    let slow = decode_reference(&bytes, &props, 32);
    assert_scenes_equal(&fast, &slow);
}

#[test]
fn large_count_crosses_parallel_threshold() {
    // PARALLEL_THRESHOLD inside the crate is 256 * 1024; pick a count that
    // exercises both the single-threaded prologue *and* multi-shard
    // partitioning. Use the minimal ordering to keep the body small.
    let props = ordering_minimal();
    let n = 300_000;
    let bytes = build_ply(n, &props);
    let fast = read_ply_bytes(&bytes).unwrap();
    let slow = decode_reference(&bytes, &props, n);
    assert_scenes_equal(&fast, &slow);
}

#[test]
fn truncated_body_rejected() {
    let props = ordering_minimal();
    let bytes = build_ply(16, &props);
    // Chop the last vertex record.
    let stride = props.len() * 4;
    let truncated = &bytes[..bytes.len() - stride / 2];
    let err = read_ply_bytes(truncated).unwrap_err();
    assert!(matches!(err, PlyError::TruncatedPayload), "got {err:?}");
}

#[test]
fn missing_required_field_rejected() {
    // Drop "f_dc_0" entirely.
    let props: Vec<PropSpec> = ordering_minimal()
        .into_iter()
        .filter(|p| p.name != "f_dc_0")
        .collect();
    let bytes = build_ply(4, &props);
    let err = read_ply_bytes(&bytes).unwrap_err();
    assert!(
        matches!(err, PlyError::MissingRequiredField(ref s) if s.contains("f_dc_0")),
        "got {err:?}"
    );
}
