//! FRONT-F alignment probe: is the GLB-readback recon row-aligned to GT,
//! or permuted? Determines whether `decoded.sel_idx` is a shared index.
use catetus_core::Color;
use catetus_gltf::read_glb;
use catetus_ply::read_ply;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gt = read_ply(Path::new(&args[1])).expect("read gt ply");
    let recon = read_glb(Path::new(&args[2])).expect("parse glb");

    println!("gt splats:    {}", gt.splats.len());
    println!("recon splats: {}", recon.splats.len());

    let n = gt.splats.len().min(recon.splats.len());
    let m = n.min(5000);
    let mut aligned_err = 0.0f64;
    for i in 0..m {
        for c in 0..3 {
            aligned_err += (recon.splats[i].position[c] - gt.splats[i].position[c]).abs() as f64;
        }
    }
    aligned_err /= (m * 3) as f64;
    println!("aligned mean |dpos| (first {m} rows): {aligned_err:.6e}");
    for i in [0usize, 1, n / 2] {
        println!(
            "row {i}: gt.pos={:?}  recon.pos={:?}",
            gt.splats[i].position, recon.splats[i].position
        );
    }
    let recon_sh = matches!(recon.splats[0].color, Color::Sh { .. });
    let gt_sh = matches!(gt.splats[0].color, Color::Sh { .. });
    println!("gt[0] is Sh: {gt_sh}   recon[0] is Sh: {recon_sh}");
    if let Color::Sh { coeffs, .. } = &recon.splats[0].color {
        println!("recon[0] sh coeffs len: {}", coeffs.len());
    }
}
