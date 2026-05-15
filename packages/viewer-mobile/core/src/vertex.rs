//! On-the-wire splat vertex consumed by the mobile renderers.
//!
//! The layout is `#[repr(C)]` so Metal vertex buffers and OpenGL ES VBOs can
//! point straight at it. Order matches the WebGPU layout in
//! `packages/viewer/src/webgpu/decode.wgsl` so the kernels port 1:1.

/// One Gaussian splat in render-ready form.
///
/// Fields are packed tightly: 3 floats position, 4 floats rotation
/// (quaternion x,y,z,w), 3 floats scale, 1 float opacity, 3 floats RGB color
/// = 14 floats = 56 bytes. The renderers stride this directly.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplatVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Quaternion `(x, y, z, w)`.
    pub rotation: [f32; 4],
    /// Linear per-axis scale (already exp() applied if the source was log).
    pub scale: [f32; 3],
    /// Linear opacity, `[0, 1]`.
    pub opacity: f32,
    /// DC RGB color, linear sRGB.
    pub color: [f32; 3],
}

impl SplatVertex {
    /// Byte stride, useful for `MTLVertexBufferLayout.stride` and `glVertexAttribPointer`.
    pub const STRIDE: usize = std::mem::size_of::<Self>();
}

/// Compile-time sanity check that the vertex stays 56 bytes (no padding).
#[allow(dead_code)]
const _: () = {
    assert!(SplatVertex::STRIDE == 56);
};
