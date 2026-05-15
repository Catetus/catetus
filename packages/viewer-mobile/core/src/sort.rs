//! Depth sort for back-to-front alpha compositing.
//!
//! For the skeleton we use a plain CPU `sort_unstable_by`. The follow-up
//! compute-shader radix sort (`packages/viewer/src/webgpu/radix_sort.wgsl`)
//! will replace this on-GPU, but the CPU oracle stays as the fixture for
//! kernel correctness tests.

use crate::vertex::SplatVertex;

/// Sort `verts` back-to-front (largest view depth first) relative to `view`.
///
/// `view` is the column-major world-to-view matrix from `Camera::view()`.
/// Returns the indices in sorted order (we don't move the vertex bytes — the
/// renderer uploads an index buffer instead).
pub fn sort_by_depth(verts: &[SplatVertex], view: &[f32; 16]) -> Vec<u32> {
    // View-space depth for point `p` under column-major view matrix `V` is
    // `V[2]*p.x + V[6]*p.y + V[10]*p.z + V[14]` (the third row of `V`).
    let v0 = view[2];
    let v1 = view[6];
    let v2 = view[10];
    let v3 = view[14];

    let mut depths: Vec<(u32, f32)> = verts
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let z = v0 * v.position[0] + v1 * v.position[1] + v2 * v.position[2] + v3;
            (i as u32, z)
        })
        .collect();
    // Back-to-front: the most-negative (farthest) z comes first because in a
    // right-handed view space the camera looks down `-Z`.
    depths.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    depths.into_iter().map(|(i, _)| i).collect()
}
