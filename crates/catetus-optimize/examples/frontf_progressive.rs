//! FRONT-F: progressive-layer measurement (attribute domain, 100% local).
//!
//! Proves the V5.2 base+residual pair is a genuine progressive-streaming
//! format:
//!   - base recon (the clean base GLB, read via `read_glb`) = fast first-paint
//!     layer, a standalone valid scene.
//!   - the on-disk `.v5tail` sidecar = refinement layer, decoded via the
//!     SHIPPING decoder (`catetus_gltf::v5_tail::decode_v5tail_bytes`) and
//!     applied ONTO the base to sharpen in place.
//!
//! Measures per-attribute reconstruction error vs the ground-truth PLY for
//! (base) and (base + decoded residual) over the residual-covered splats, in
//! the SAME raw 3DGS space the codec encodes (log-scale, logit-opacity) — the
//! identical math `build_v5_tail_sidecar` used to compute the residual.
//!
//! The preset Morton-sorts splats, so the GLB recon is NOT row-aligned to GT.
//! We recover, for each residual-covered recon row `s`, its GT row by matching
//! `recon_pos[s] + decoded.pos[kk]` (≈ the true GT position, since the residual
//! IS `gt − recon` to dequant precision) against a spatial hash of GT positions.
//! Exact in practice (positions are unique); the harness reports the match rate
//! + decoded-vs-true residual agreement so the measurement is self-checking.
//!
//! NOTE: attribute-domain MSE / PSNR only. Render-domain dB needs the gsplat
//! rasterizer (GPU) and is an explicit follow-up, out of scope here.
//!
//! Run:
//!   cargo run -p catetus-optimize --release --example frontf_progressive -- \
//!       <GT.ply> <base.glb> <full.glb.v5tail>

use catetus_core::Color;
use catetus_gltf::read_glb;
use catetus_gltf::v5_tail::decode_v5tail_bytes;
use catetus_ply::read_ply;
use std::collections::HashMap;
use std::path::Path;

#[inline]
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln()
}
#[inline]
fn ln_scale(s: f32) -> f32 {
    s.max(f32::MIN_POSITIVE).ln()
}

fn dc_shr(c: &Color) -> (Vec<f32>, Vec<f32>) {
    match c {
        Color::Sh { coeffs, .. } if coeffs.len() >= 3 => {
            let dc = coeffs[0..3].to_vec();
            let shr = if coeffs.len() > 3 { coeffs[3..].to_vec() } else { vec![] };
            (dc, shr)
        }
        Color::Rgb(rgb) => (rgb.to_vec(), vec![]),
        _ => (vec![0.0, 0.0, 0.0], vec![]),
    }
}

#[derive(Default)]
struct Acc {
    se_base: f64,
    se_res: f64,
    n: usize,
    peak: f64,
}
impl Acc {
    fn add(&mut self, gt: f32, base: f32, applied: f32) {
        self.se_base += ((base - gt) as f64).powi(2);
        self.se_res += ((applied - gt) as f64).powi(2);
        self.n += 1;
        self.peak = self.peak.max((gt as f64).abs());
    }
    fn mse_base(&self) -> f64 { self.se_base / self.n.max(1) as f64 }
    fn mse_res(&self) -> f64 { self.se_res / self.n.max(1) as f64 }
}

fn psnr(peak: f64, mse: f64) -> f64 {
    if mse <= 0.0 { return f64::INFINITY; }
    10.0 * (peak * peak / mse).log10()
}

fn cell_key(p: [f32; 3], step: f32) -> (i64, i64, i64) {
    (
        (p[0] / step).floor() as i64,
        (p[1] / step).floor() as i64,
        (p[2] / step).floor() as i64,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gt = read_ply(Path::new(&args[1])).expect("read GT ply");
    let recon = read_glb(Path::new(&args[2])).expect("parse base glb");
    let payload = std::fs::read(Path::new(&args[3])).expect("read .v5tail");
    let decoded = decode_v5tail_bytes(&payload).expect("decode v5tail");

    let n = recon.splats.len();
    let k = decoded.sel_idx.len();
    let coefs = decoded.header.sh_rest_coefs as usize;

    let step = 0.05f32;
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (i, s) in gt.splats.iter().enumerate() {
        grid.entry(cell_key(s.position, step)).or_default().push(i as u32);
    }
    let nearest_gt = |target: [f32; 3]| -> Option<usize> {
        let kc = cell_key(target, step);
        let mut best = None;
        let mut best_d = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = grid.get(&(kc.0 + dx, kc.1 + dy, kc.2 + dz)) {
                        for &gi in v {
                            let g = &gt.splats[gi as usize];
                            let d = ((g.position[0] - target[0]) as f64).powi(2)
                                + ((g.position[1] - target[1]) as f64).powi(2)
                                + ((g.position[2] - target[2]) as f64).powi(2);
                            if d < best_d {
                                best_d = d;
                                best = Some(gi as usize);
                            }
                        }
                    }
                }
            }
        }
        best
    };

    let mut origin = vec![u32::MAX; k];
    let mut matched = 0usize;
    let mut resid_check = 0.0f64;
    let mut resid_n = 0usize;
    for (kk, &s) in decoded.sel_idx.iter().enumerate() {
        let s = s as usize;
        let target = [
            recon.splats[s].position[0] + decoded.pos[kk * 3],
            recon.splats[s].position[1] + decoded.pos[kk * 3 + 1],
            recon.splats[s].position[2] + decoded.pos[kk * 3 + 2],
        ];
        if let Some(gi) = nearest_gt(target) {
            origin[kk] = gi as u32;
            matched += 1;
            let g = &gt.splats[gi];
            for c in 0..3 {
                let true_res = g.position[c] - recon.splats[s].position[c];
                resid_check += (true_res - decoded.pos[kk * 3 + c]).abs() as f64;
                resid_n += 1;
            }
        }
    }
    resid_check /= resid_n.max(1) as f64;

    println!("scene splats:       {n}");
    println!("residual-covered K: {k}  (top {:.2}%)", 100.0 * k as f64 / n as f64);
    println!("matched K to GT:    {matched} / {k}");
    println!("sh_rest_coefs:      {coefs}");
    println!("sidecar bytes:      {}", payload.len());
    println!("decoded-vs-true position residual mean|err|: {resid_check:.6e} (codec dequant error; small => faithful match)");
    println!();

    let mut a_pos = Acc::default();
    let mut a_rot = Acc::default();
    let mut a_opa = Acc::default();
    let mut a_sca = Acc::default();
    let mut a_dc = Acc::default();
    let mut a_shr = Acc::default();

    for (kk, &s) in decoded.sel_idx.iter().enumerate() {
        if origin[kk] == u32::MAX {
            continue;
        }
        let s = s as usize;
        let r = &recon.splats[s];
        let g = &gt.splats[origin[kk] as usize];
        for c in 0..3 {
            a_pos.add(g.position[c], r.position[c], r.position[c] + decoded.pos[kk * 3 + c]);
        }
        for c in 0..4 {
            a_rot.add(g.rotation[c], r.rotation[c], r.rotation[c] + decoded.rot[kk * 4 + c]);
        }
        {
            let base = logit(r.opacity);
            a_opa.add(logit(g.opacity), base, base + decoded.opa[kk]);
        }
        for c in 0..3 {
            let base = ln_scale(r.scale[c]);
            a_sca.add(ln_scale(g.scale[c]), base, base + decoded.sca[kk * 3 + c]);
        }
        let (r_dc, r_shr) = dc_shr(&r.color);
        let (g_dc, g_shr) = dc_shr(&g.color);
        for c in 0..3 {
            let base = *r_dc.get(c).unwrap_or(&0.0);
            a_dc.add(*g_dc.get(c).unwrap_or(&0.0), base, base + decoded.dc[kk * 3 + c]);
        }
        let want = coefs * 3;
        for c in 0..want {
            let base = *r_shr.get(c).unwrap_or(&0.0);
            a_shr.add(*g_shr.get(c).unwrap_or(&0.0), base, base + decoded.shr[kk * want + c]);
        }
    }

    let rows: [(&str, usize, &Acc); 6] = [
        ("position", 3, &a_pos),
        ("rotation", 4, &a_rot),
        ("opacity*", 1, &a_opa),
        ("scale*", 3, &a_sca),
        ("color_dc", 3, &a_dc),
        ("sh_rest", coefs * 3, &a_shr),
    ];
    println!(
        "{:>9} {:>4} {:>14} {:>14} {:>9} {:>11} {:>11} {:>9}",
        "attr", "dim", "mse_base", "mse_base+res", "drop_x", "psnr_base", "psnr_b+res", "dpsnr"
    );
    let (mut tb, mut tr, mut tn, mut gp) = (0.0f64, 0.0f64, 0usize, 0.0f64);
    for (name, dim, acc) in rows {
        let mb = acc.mse_base();
        let mr = acc.mse_res();
        let pb = psnr(acc.peak, mb);
        let pr = psnr(acc.peak, mr);
        println!(
            "{:>9} {:>4} {:>14.6e} {:>14.6e} {:>9.2} {:>11.3} {:>11.3} {:>9.3}",
            name, dim, mb, mr, mb / mr.max(1e-30), pb, pr, pr - pb
        );
        tb += acc.se_base; tr += acc.se_res; tn += acc.n; gp = gp.max(acc.peak);
    }
    let omb = tb / tn as f64;
    let omr = tr / tn as f64;
    println!();
    println!(
        "OVERALL (all attrs concatenated)  mse_base={:.6e}  mse_base+res={:.6e}  drop_x={:.2}",
        omb, omr, omb / omr.max(1e-30)
    );
    println!(
        "OVERALL  psnr_base={:.3} dB  psnr_base+res={:.3} dB  dpsnr=+{:.3} dB  (peak={:.4})",
        psnr(gp, omb), psnr(gp, omr), psnr(gp, omr) - psnr(gp, omb), gp
    );
    println!();
    println!("* opacity in logit space, scale in ln space (the codec's residual space).");
    println!("NOTE: attribute-domain PSNR only. Render-domain dB is a GPU follow-up (out of scope).");
}
