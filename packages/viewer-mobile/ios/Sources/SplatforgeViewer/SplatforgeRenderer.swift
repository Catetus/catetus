// SplatforgeRenderer — point-sprite renderer with orbit camera + Rust depth sort.
//
// Pipeline per frame:
// 1. (one-time at load) `sfmv_decode_glb` → vertex buffer of `SplatVertex`.
// 2. Build view-proj from the orbit camera state (azimuth / elevation /
//    distance / target / fov). The matrices are computed in Swift because the
//    Rust side keeps its own column-major convention; the renderer needs the
//    matrix in MTLBuffer-ready layout per frame anyway.
// 3. Call `sfmv_sort_by_depth` to produce a fresh back-to-front index buffer.
//    The CPU sort is the oracle path; the GPU radix-sort kernel is parity-
//    tested but not yet hooked into the live frame loop (see
//    `docs/perf/ios-viewer-status.md` for the wiring plan).
// 4. Draw one instanced quad per splat with `SplatPointSprite.metal`, indexed
//    via the sort buffer (instance fetches via vertex stage `instance_id`).
//
// The compute-shader sort + decode + 2D-cov projection paths now live in
// `SplatforgeShaders` (RadixSort.metal, SplatDecode.metal, ProjectGather.metal,
// HistogramSubgroup.metal, ScanMultiblock.metal). The frame loop here still
// uses Phase-1 point-sprite + CPU sort; the kernel-parity tests guarantee the
// GPU paths produce identical output when they're swapped in.

import Foundation
import MetalKit
import SplatforgeViewerC
import SplatforgeShaders
import simd

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Camera spherical-coords state shared by all platforms.
private struct OrbitState {
    /// Azimuth around world Y, radians.
    var azimuth: Float = 0
    /// Elevation off equator, radians. Clamped to `[-π/2 + ε, π/2 - ε]`.
    var elevation: Float = 0.25
    /// Distance from target, world units.
    var distance: Float = 3.0
    /// Look-at target, typically the splat-cloud centroid.
    var target: SIMD3<Float> = .zero
    /// FOV in radians.
    var fov: Float = .pi / 3

    /// Eye position derived from spherical coords.
    var eye: SIMD3<Float> {
        let ce = cos(elevation), se = sin(elevation)
        let ca = cos(azimuth),   sa = sin(azimuth)
        let r = distance
        return target + SIMD3<Float>(r * ce * sa, r * se, r * ce * ca)
    }
}

/// Swift never sees the inside of `SfmvBuffer` — the C header forward-declares
/// it as an opaque struct. We carry the handle as `OpaquePointer?` and cast
/// when calling FFI fns.
public final class SplatforgeRenderer: NSObject, MTKViewDelegate {
    private var device: MTLDevice?
    private var queue: MTLCommandQueue?
    private var pipeline: MTLRenderPipelineState?
    private var depthState: MTLDepthStencilState?
    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?
    private var indexScratch: [UInt32] = []
    private var splatCount: Int = 0
    private var bufferHandle: OpaquePointer?
    private var aspect: Float = 1.0
    private var orbit = OrbitState()

    deinit {
        if let h = bufferHandle {
            sfmv_buffer_free(h)
        }
    }

    func attach(to view: MTKView) {
        guard let device = view.device else { return }
        self.device = device
        self.queue = device.makeCommandQueue()
        self.pipeline = makePipeline(device: device, pixelFormat: view.colorPixelFormat)
        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = false // splats blend back-to-front; no Z-write
        self.depthState = device.makeDepthStencilState(descriptor: dd)
    }

    /// Read the `.glb` from disk, call into Rust, upload to a Metal buffer.
    func load(url: URL) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.loadInBackground(url: url)
        }
    }

    private func loadInBackground(url: URL) {
        guard let data = try? Data(contentsOf: url) else { return }
        var handle: OpaquePointer?
        let status = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> SfmvStatus in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return SfmvStatusNullPointer
            }
            return sfmv_decode_glb(base, raw.count, &handle)
        }
        guard status == SfmvStatusOk, let h = handle else { return }
        let count = sfmv_buffer_len(h)
        let stride = sfmv_vertex_stride()
        guard let src = sfmv_buffer_data(h) else { sfmv_buffer_free(h); return }

        // Compute centroid for the orbit target so an arbitrary asset shows up
        // centered in the viewport instead of half-off-screen.
        var centroid = SIMD3<Float>(0, 0, 0)
        var radius: Float = 0
        if count > 0 {
            let typed = src.bindMemory(to: Float.self, capacity: count * stride / MemoryLayout<Float>.size)
            // SplatVertex layout: position (3f), rotation (4f), scale (3f), opacity (1f), color (3f) = 14 floats / 56 bytes.
            let floatsPerVertex = stride / MemoryLayout<Float>.size
            for i in 0..<count {
                let base = i * floatsPerVertex
                centroid += SIMD3<Float>(typed[base], typed[base + 1], typed[base + 2])
            }
            centroid /= Float(count)
            for i in 0..<count {
                let base = i * floatsPerVertex
                let p = SIMD3<Float>(typed[base], typed[base + 1], typed[base + 2]) - centroid
                radius = max(radius, simd_length(p))
            }
        }

        DispatchQueue.main.async { [weak self] in
            guard let self = self, let device = self.device else { sfmv_buffer_free(h); return }
            self.vertexBuffer = device.makeBuffer(bytes: src,
                                                  length: count * stride,
                                                  options: .storageModeShared)
            self.indexBuffer = device.makeBuffer(length: max(count, 1) * MemoryLayout<UInt32>.size,
                                                 options: .storageModeShared)
            self.indexScratch = [UInt32](repeating: 0, count: count)
            self.splatCount = count
            self.bufferHandle = h
            self.orbit.target = centroid
            // Frame the asset: pull the camera back so the whole cloud is in
            // the frustum at the default FOV.
            if radius > 0 {
                self.orbit.distance = radius / tan(self.orbit.fov * 0.5) * 1.6
            }
        }
    }

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspect = Float(size.width / max(size.height, 1))
    }

    public func draw(in view: MTKView) {
        guard let queue = queue,
              let pipeline = pipeline,
              let descriptor = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(depthState)
        if let vb = vertexBuffer, let ib = indexBuffer, splatCount > 0, let h = bufferHandle {
            // CPU depth sort via Rust FFI. Hot path; for the bonsai-7k asset
            // this is ~150 µs on an iPhone 15 Pro (single-threaded
            // `sort_unstable_by`). The GPU radix-sort port replaces this when
            // the project pass lands.
            let view = lookAtMatrix()
            indexScratch.withUnsafeMutableBufferPointer { idxPtr in
                view.withUnsafeBufferPointer { vp in
                    _ = sfmv_sort_by_depth(h, vp.baseAddress, idxPtr.baseAddress)
                }
            }
            indexScratch.withUnsafeBufferPointer { sp in
                memcpy(ib.contents(), sp.baseAddress, splatCount * MemoryLayout<UInt32>.size)
                return
            }
            enc.setVertexBuffer(vb, offset: 0, index: 0)
            var vp = viewProjMatrix()
            enc.setVertexBytes(&vp, length: MemoryLayout<simd_float4x4>.size, index: 1)
            enc.setVertexBuffer(ib, offset: 0, index: 2)
            enc.drawPrimitives(type: .triangleStrip,
                               vertexStart: 0,
                               vertexCount: 4,
                               instanceCount: splatCount)
        }
        enc.endEncoding()
        if let drawable = view.currentDrawable {
            cmd.present(drawable)
        }
        cmd.commit()
    }

    // MARK: - Gesture handlers

#if canImport(UIKit)
    private var panStart: CGPoint = .zero
    private var startAz: Float = 0
    private var startEl: Float = 0
    private var startDist: Float = 0

    @objc func handlePan(_ g: UIPanGestureRecognizer) {
        switch g.state {
        case .began:
            startAz = orbit.azimuth
            startEl = orbit.elevation
        case .changed:
            let t = g.translation(in: g.view)
            // 800 px ≈ a full sweep
            let dx = Float(t.x) / 800.0 * .pi
            let dy = Float(t.y) / 800.0 * .pi
            orbit.azimuth = startAz - dx
            orbit.elevation = clamp(startEl + dy, -1.55, 1.55)
        default: break
        }
    }

    @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
        switch g.state {
        case .began: startDist = orbit.distance
        case .changed:
            orbit.distance = clamp(startDist / Float(g.scale), 0.01, 1000)
        default: break
        }
    }
#elseif canImport(AppKit)
    private var startAz: Float = 0
    private var startEl: Float = 0
    private var startDist: Float = 0

    @objc func handleMacPan(_ g: NSPanGestureRecognizer) {
        switch g.state {
        case .began:
            startAz = orbit.azimuth
            startEl = orbit.elevation
        case .changed:
            let t = g.translation(in: g.view)
            let dx = Float(t.x) / 800.0 * .pi
            let dy = Float(t.y) / 800.0 * .pi
            orbit.azimuth = startAz - dx
            orbit.elevation = clamp(startEl - dy, -1.55, 1.55)
        default: break
        }
    }

    @objc func handleMacMagnify(_ g: NSMagnificationGestureRecognizer) {
        switch g.state {
        case .began: startDist = orbit.distance
        case .changed:
            orbit.distance = clamp(startDist / Float(1 + g.magnification), 0.01, 1000)
        default: break
        }
    }
#endif

    // MARK: - helpers

    private func clamp(_ x: Float, _ lo: Float, _ hi: Float) -> Float {
        return max(lo, min(hi, x))
    }

    private func makePipeline(device: MTLDevice, pixelFormat: MTLPixelFormat) -> MTLRenderPipelineState? {
        // The Shaders/ folder ships as a resource bundle on the
        // `SplatforgeShaders` target; that's where the compiled .metallib
        // lives at runtime.
        let library: MTLLibrary?
        do {
            library = try device.makeDefaultLibrary(bundle: SplatforgeShaders.bundle)
        } catch {
            library = nil
        }
        guard let lib = library,
              let vfn = lib.makeFunction(name: "splat_point_vertex"),
              let ffn = lib.makeFunction(name: "splat_point_fragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = pixelFormat
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = .depth32Float
        return try? device.makeRenderPipelineState(descriptor: desc)
    }

    /// Column-major view matrix, lookAt(eye, target, +Y), matching the
    /// `Camera::view()` convention in the Rust core's `math::look_at`.
    private func lookAtMatrix() -> [Float] {
        let eye = orbit.eye
        let target = orbit.target
        let up = SIMD3<Float>(0, 1, 0)
        let f = simd_normalize(target - eye)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        var m = [Float](repeating: 0, count: 16)
        m[0] = s.x;  m[4] = s.y;  m[8]  = s.z;  m[12] = -simd_dot(s, eye)
        m[1] = u.x;  m[5] = u.y;  m[9]  = u.z;  m[13] = -simd_dot(u, eye)
        m[2] = -f.x; m[6] = -f.y; m[10] = -f.z; m[14] =  simd_dot(f, eye)
        m[15] = 1
        return m
    }

    private func viewProjMatrix() -> simd_float4x4 {
        let near: Float = 0.05
        let far: Float = 1000
        let f = 1.0 / tan(orbit.fov * 0.5)
        let nf = 1.0 / (near - far)
        let proj = simd_float4x4(
            simd_float4(f / aspect, 0, 0, 0),
            simd_float4(0, f, 0, 0),
            simd_float4(0, 0, (far + near) * nf, -1),
            simd_float4(0, 0, 2 * far * near * nf, 0)
        )
        let v = lookAtMatrix()
        let view = simd_float4x4(
            simd_float4(v[0], v[1], v[2], v[3]),
            simd_float4(v[4], v[5], v[6], v[7]),
            simd_float4(v[8], v[9], v[10], v[11]),
            simd_float4(v[12], v[13], v[14], v[15])
        )
        return proj * view
    }
}
