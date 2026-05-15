//! `splatforge-usd-fixtures` — deterministic generator for the
//! `ParticleField3DGaussianSplat` conformance fixture corpus.
//!
//! Usage:
//!     splatforge-usd-fixtures <out_dir>
//!
//! Writes eight fixtures to `<out_dir>`:
//!
//!   01_valid_minimal.usda          one splat, identity quat
//!   02_valid_particle_field.usda   three splats with non-identity quats
//!   03_valid_dense.usda            64-splat grid; exercises array path
//!   04_valid_with_sh.usda          adds `custom float[] splatforge:shCoefficients` (degree 3)
//!   05_valid_minimal.usdc          binary form of fixture 01 (round-trip path)
//!   06_invalid_no_orientations.usda removes the `orientations` attribute
//!   07_invalid_opacity_out_of_range.usda  one opacity = 1.5 (out of [0,1])
//!   08_invalid_count_mismatch.usda one of the per-splat arrays has wrong length
//!
//! The generator is byte-deterministic: same input always produces the
//! same bytes. The negative fixtures are produced by mutating canonical
//! USDA output from `splatforge_usd::render_usda`, not by hand-crafting
//! strings, so the validator always operates on realistic USDA shapes.

use std::fs;
use std::path::{Path, PathBuf};

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_usd::{render_usda, write_usda, write_usdc, UsdWriteOpts};

fn deterministic_scene(n: usize, with_sh: bool) -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..n {
        let f = i as f32;
        let color = if with_sh {
            // degree 3 → 16 bands × 3 channels = 48 coefficients per splat.
            let mut coeffs = Vec::with_capacity(48);
            coeffs.push((f * 0.1).fract().abs());
            coeffs.push(0.2);
            coeffs.push(0.3);
            for j in 0..45 {
                coeffs.push((f + j as f32) * 0.001);
            }
            Color::Sh { degree: 3, coeffs }
        } else {
            Color::Rgb([
                ((f * 0.1).fract().abs()).clamp(0.0, 1.0),
                0.2,
                0.3,
            ])
        };
        scene.splats.push(Splat {
            position: [f, f * 0.5, -f],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            opacity: 0.5,
            color,
        });
    }
    scene
}

fn dense_grid_scene() -> SplatScene {
    let mut scene = SplatScene::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                scene.splats.push(Splat {
                    position: [x as f32, y as f32, z as f32],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.25, 0.25, 0.25],
                    opacity: 0.5,
                    color: Color::Rgb([
                        x as f32 / 4.0,
                        y as f32 / 4.0,
                        z as f32 / 4.0,
                    ]),
                });
            }
        }
    }
    scene
}

fn three_splat_scene() -> SplatScene {
    let mut scene = SplatScene::new();
    // Use the same three splats from splatforge-usd/fixtures/particle_field.usda
    // but with normalised quaternions (the fixture file uses an un-normalised
    // first quat (0.9, 0.1, 0.2, 0.3); we normalise to keep QUATS_NORMALIZED
    // passing on the valid fixture).
    let q0 = normalise([0.9, 0.1, 0.2, 0.3]);
    let q1 = [1.0, 0.0, 0.0, 0.0];
    // (cos π/4, sin π/4, 0, 0); we use a value close to FRAC_1_SQRT_2.
    // Allow the approx-constant lint because the literal is intentionally
    // chosen to match the textual fixture exactly, not the math constant.
    #[allow(clippy::approx_constant)]
    let raw_h: f32 = 0.7071;
    let q2 = normalise([raw_h, raw_h, 0.0, 0.0]);
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.5, -1.0], [2.0, 1.0, -2.0]];
    let scales = [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.25, 0.5, 0.25]];
    let opacities = [0.5, 0.75, 1.0];
    let colors = [
        [0.1, 0.2, 0.3],
        [0.5, 0.5, 0.5],
        [0.9, 0.7, 0.1],
    ];
    // USD authoring quat is (w,x,y,z); IR is (x,y,z,w).
    let quats = [q0, q1, q2];
    for i in 0..3 {
        let q = quats[i];
        scene.splats.push(Splat {
            position: positions[i],
            rotation: [q[1], q[2], q[3], q[0]],
            scale: scales[i],
            opacity: opacities[i],
            color: Color::Rgb(colors[i]),
        });
    }
    scene
}

fn normalise(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn write_negative_usda(path: &Path, scene_factory: fn() -> SplatScene, mutate: impl FnOnce(&str) -> String) -> std::io::Result<()> {
    let scene = scene_factory();
    let baseline = render_usda(&scene, &UsdWriteOpts::default());
    let mutated = mutate(&baseline);
    fs::write(path, mutated)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("fixtures"));
    fs::create_dir_all(&out_dir).expect("create out_dir");

    // 01: minimal valid USDA.
    {
        let scene = deterministic_scene(1, false);
        let p = out_dir.join("01_valid_minimal.usda");
        write_usda(&scene, &p, &UsdWriteOpts::default()).expect("write 01");
    }

    // 02: 3-splat particle field with non-identity quaternions.
    {
        let scene = three_splat_scene();
        let p = out_dir.join("02_valid_particle_field.usda");
        write_usda(&scene, &p, &UsdWriteOpts::default()).expect("write 02");
    }

    // 03: dense 4×4×4 grid.
    {
        let scene = dense_grid_scene();
        let p = out_dir.join("03_valid_dense.usda");
        write_usda(&scene, &p, &UsdWriteOpts::default()).expect("write 03");
    }

    // 04: SH coefficients (degree 3).
    {
        let scene = deterministic_scene(2, true);
        let p = out_dir.join("04_valid_with_sh.usda");
        write_usda(&scene, &p, &UsdWriteOpts::default()).expect("write 04");
    }

    // 05: USDC binary form of fixture 01.
    {
        let scene = deterministic_scene(1, false);
        let p = out_dir.join("05_valid_minimal.usdc");
        write_usdc(&scene, &p, &UsdWriteOpts::default()).expect("write 05");
    }

    // 06: invalid — orientations attribute removed.
    write_negative_usda(
        &out_dir.join("06_invalid_no_orientations.usda"),
        || deterministic_scene(3, false),
        |raw| {
            // Drop the entire `quatf[] orientations = [...]` line.
            raw.lines()
                .filter(|l| !l.contains("quatf[] orientations"))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"
        },
    )
    .expect("write 06");

    // 07: invalid — first opacity bumped to 1.5 (out of [0,1]).
    write_negative_usda(
        &out_dir.join("07_invalid_opacity_out_of_range.usda"),
        || deterministic_scene(3, false),
        |raw| {
            // Replace the first opacity value `0.5` with `1.5` in the
            // opacities line.
            let mut out = String::with_capacity(raw.len());
            let mut replaced = false;
            for line in raw.lines() {
                if !replaced && line.contains("float[] opacities") {
                    out.push_str(&line.replacen("0.5", "1.5", 1));
                    replaced = true;
                } else {
                    out.push_str(line);
                }
                out.push('\n');
            }
            assert!(replaced, "could not find opacities line to mutate");
            out
        },
    )
    .expect("write 07");

    // 08: invalid — opacities length doesn't match points length.
    write_negative_usda(
        &out_dir.join("08_invalid_count_mismatch.usda"),
        || deterministic_scene(3, false),
        |raw| {
            // Truncate the opacities array from 3 entries to 2 by dropping
            // its trailing element.
            let mut out = String::with_capacity(raw.len());
            for line in raw.lines() {
                if line.contains("float[] opacities") {
                    // Find the array body and drop the last comma-separated
                    // element.
                    let lb = line.find('[').expect("`[`");
                    let rb = line.rfind(']').expect("`]`");
                    let body = &line[lb + 1..rb];
                    let parts: Vec<&str> = body.split(',').collect();
                    assert!(parts.len() > 1, "need >=2 entries to truncate");
                    let truncated = parts[..parts.len() - 1].join(",");
                    out.push_str(&line[..lb + 1]);
                    out.push_str(&truncated);
                    out.push_str(&line[rb..]);
                } else {
                    out.push_str(line);
                }
                out.push('\n');
            }
            out
        },
    )
    .expect("write 08");

    println!("wrote 8 fixtures to {}", out_dir.display());
}
