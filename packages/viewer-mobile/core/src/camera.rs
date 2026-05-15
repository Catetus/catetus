//! Camera + view/projection matrix builder.
//!
//! Mirrors `packages/viewer/src/renderer/math.ts` so the WebGPU and mobile
//! viewers agree on the right-handed look-at and depth-`[0, 1]` perspective.

/// A pinhole camera pose. All vectors are world-space; angles are radians.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Eye position.
    pub position: [f32; 3],
    /// Look-at target.
    pub target: [f32; 3],
    /// World up axis (typically `[0, 1, 0]`).
    pub up: [f32; 3],
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Width / height.
    pub aspect: f32,
    /// Near clip distance.
    pub near: f32,
    /// Far clip distance.
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 3.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y: std::f32::consts::FRAC_PI_3, // 60 deg
            aspect: 1.0,
            near: 0.05,
            far: 1000.0,
        }
    }
}

impl Camera {
    /// View · projection. Column-major 16-float buffer, ready for `MTLBuffer`
    /// or `glUniformMatrix4fv(transpose = false)`.
    pub fn view_proj(&self) -> [f32; 16] {
        crate::math::mul_mat4(&self.proj(), &self.view())
    }

    /// World-to-view (right-handed `lookAt`, `-Z` looks at target).
    pub fn view(&self) -> [f32; 16] {
        crate::math::look_at(self.position, self.target, self.up)
    }

    /// Right-handed perspective with depth in `[0, 1]`.
    pub fn proj(&self) -> [f32; 16] {
        crate::math::perspective(self.fov_y, self.aspect, self.near, self.far)
    }
}
