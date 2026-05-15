//! C ABI exposed to Swift (via `module.modulemap`) and Kotlin (via JNI shims).
//!
//! Conventions:
//! * Functions return `0` on success, negative on failure (see [`SfmvStatus`]).
//! * Buffers are owned by Rust; callers receive an opaque [`SfmvBuffer`]
//!   handle and read pointer + len via [`sfmv_buffer_data`] / [`sfmv_buffer_len`].
//! * The caller MUST call [`sfmv_buffer_free`] exactly once per handle.
//!
//! These are the only entry points cbindgen exports; everything else is
//! Rust-internal.

use std::ffi::c_void;
use std::os::raw::{c_float, c_int, c_uchar};
use std::slice;

use crate::decode::{decode_glb_bytes, DecodeError};
use crate::sort::sort_by_depth;
use crate::vertex::SplatVertex;

/// Status codes returned by the FFI surface.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfmvStatus {
    /// Operation succeeded.
    Ok = 0,
    /// `bytes`/`len` did not parse as a valid `.glb` with `KHR_gaussian_splatting`.
    DecodeFailed = -1,
    /// Asset was syntactically valid but had no splats.
    Empty = -2,
    /// Caller passed a null pointer where one was forbidden.
    NullPointer = -3,
}

/// Opaque vertex buffer handle. Internally a boxed `Vec<SplatVertex>`.
#[repr(C)]
pub struct SfmvBuffer {
    _opaque: [u8; 0],
}

struct OwnedBuffer {
    verts: Vec<SplatVertex>,
}

/// Decode a `.glb` blob.
///
/// On success, `*out` receives a new buffer handle that the caller must free
/// with [`sfmv_buffer_free`]. On failure `*out` is set to null and the
/// returned status indicates the reason.
///
/// # Safety
/// `bytes` must be readable for `len` bytes and `out` must be a writable
/// `*mut *mut SfmvBuffer`.
#[no_mangle]
pub unsafe extern "C" fn sfmv_decode_glb(
    bytes: *const c_uchar,
    len: usize,
    out: *mut *mut SfmvBuffer,
) -> SfmvStatus {
    if bytes.is_null() || out.is_null() {
        return SfmvStatus::NullPointer;
    }
    let slice = slice::from_raw_parts(bytes, len);
    match decode_glb_bytes(slice) {
        Ok(verts) => {
            let boxed = Box::new(OwnedBuffer { verts });
            *out = Box::into_raw(boxed) as *mut SfmvBuffer;
            SfmvStatus::Ok
        }
        Err(DecodeError::Empty) => {
            *out = std::ptr::null_mut();
            SfmvStatus::Empty
        }
        Err(_) => {
            *out = std::ptr::null_mut();
            SfmvStatus::DecodeFailed
        }
    }
}

/// Pointer to the raw [`SplatVertex`] array inside the buffer (read-only).
///
/// # Safety
/// `buf` must be a live handle returned by [`sfmv_decode_glb`].
#[no_mangle]
pub unsafe extern "C" fn sfmv_buffer_data(buf: *const SfmvBuffer) -> *const c_void {
    if buf.is_null() {
        return std::ptr::null();
    }
    let b = &*(buf as *const OwnedBuffer);
    b.verts.as_ptr() as *const c_void
}

/// Number of [`SplatVertex`] entries in the buffer.
///
/// # Safety
/// `buf` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn sfmv_buffer_len(buf: *const SfmvBuffer) -> usize {
    if buf.is_null() {
        return 0;
    }
    let b = &*(buf as *const OwnedBuffer);
    b.verts.len()
}

/// Byte stride of one vertex. Useful for setting up vertex descriptors.
#[no_mangle]
pub extern "C" fn sfmv_vertex_stride() -> usize {
    SplatVertex::STRIDE
}

/// Free a buffer returned by [`sfmv_decode_glb`].
///
/// # Safety
/// `buf` must have come from [`sfmv_decode_glb`] and not yet been freed.
#[no_mangle]
pub unsafe extern "C" fn sfmv_buffer_free(buf: *mut SfmvBuffer) {
    if buf.is_null() {
        return;
    }
    drop(Box::from_raw(buf as *mut OwnedBuffer));
}

/// Sort the buffer by view-space depth and write `count` `u32` indices to
/// `out_indices` (caller-allocated; must be at least `sfmv_buffer_len`
/// entries).
///
/// `view_col_major` points to 16 f32 in column-major order (matching what
/// `Camera::view()` returns).
///
/// # Safety
/// All pointers must be valid for the specified element counts.
#[no_mangle]
pub unsafe extern "C" fn sfmv_sort_by_depth(
    buf: *const SfmvBuffer,
    view_col_major: *const c_float,
    out_indices: *mut u32,
) -> c_int {
    if buf.is_null() || view_col_major.is_null() || out_indices.is_null() {
        return SfmvStatus::NullPointer as c_int;
    }
    let b = &*(buf as *const OwnedBuffer);
    let view_slice = slice::from_raw_parts(view_col_major, 16);
    let mut view = [0.0_f32; 16];
    view.copy_from_slice(view_slice);
    let idx = sort_by_depth(&b.verts, &view);
    std::ptr::copy_nonoverlapping(idx.as_ptr(), out_indices, idx.len());
    SfmvStatus::Ok as c_int
}
