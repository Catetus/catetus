// SplatforgeRenderer — minimal point-sprite renderer.
//
// Pipeline:
// 1. `sfmv_decode_glb` → vertex buffer of `SplatVertex` (56-byte stride).
// 2. Build view-proj from a free-orbit camera (touch / drag in a follow-up).
// 3. Draw one instanced quad per splat with `SplatPointSprite.metal`.
//
// The compute-shader sort + 2D-cov path lives in `Shaders/RadixSort.metal` /
// `Shaders/ProjectCovariance.metal` — both STUBS pending the WGSL→MSL port.

import Foundation
import MetalKit
import SplatforgeViewerC
import simd

/// Swift never sees the inside of `SfmvBuffer` — the C header forward-declares
/// it as an opaque struct. We carry the handle as `OpaquePointer?` and cast
/// when calling FFI fns.
final class SplatforgeRenderer: NSObject, MTKViewDelegate {
    private var device: MTLDevice?
    private var queue: MTLCommandQueue?
    private var pipeline: MTLRenderPipelineState?
    private var depthState: MTLDepthStencilState?
    private var vertexBuffer: MTLBuffer?
    private var splatCount: Int = 0
    private var bufferHandle: OpaquePointer?
    private var aspect: Float = 1.0

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
        DispatchQueue.main.async { [weak self] in
            guard let self = self, let device = self.device else { sfmv_buffer_free(h); return }
            self.vertexBuffer = device.makeBuffer(bytes: src,
                                                  length: count * stride,
                                                  options: .storageModeShared)
            self.splatCount = count
            self.bufferHandle = h
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspect = Float(size.width / max(size.height, 1))
    }

    func draw(in view: MTKView) {
        guard let queue = queue,
              let pipeline = pipeline,
              let descriptor = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(depthState)
        if let vb = vertexBuffer, splatCount > 0 {
            enc.setVertexBuffer(vb, offset: 0, index: 0)
            var vp = identityViewProj(aspect: aspect)
            enc.setVertexBytes(&vp, length: MemoryLayout<simd_float4x4>.size, index: 1)
            // Phase-1: draw each splat as a single point sprite (4-vert triangle strip,
            // instanced). The MSL is in `SplatPointSprite.metal`.
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

    // MARK: - helpers

    private func makePipeline(device: MTLDevice, pixelFormat: MTLPixelFormat) -> MTLRenderPipelineState? {
        // The Shaders/ folder ships as a resource bundle: `Bundle.module` is
        // synthesised by SwiftPM whenever `resources:` is set on the target.
        let library: MTLLibrary?
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
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

    private func identityViewProj(aspect: Float) -> simd_float4x4 {
        // Trivial fixed camera: eye=(0,0,3), looking down -Z. Replaced by the
        // gesture-driven camera in a follow-up.
        let fov: Float = .pi / 3
        let f = 1 / tan(fov * 0.5)
        let n: Float = 0.05
        let fp: Float = 1000
        let proj = simd_float4x4(
            simd_float4(f / aspect, 0, 0, 0),
            simd_float4(0, f, 0, 0),
            simd_float4(0, 0, (fp + n) / (n - fp), -1),
            simd_float4(0, 0, 2 * fp * n / (n - fp), 0)
        )
        let view = simd_float4x4(
            simd_float4(1, 0, 0, 0),
            simd_float4(0, 1, 0, 0),
            simd_float4(0, 0, 1, 0),
            simd_float4(0, 0, -3, 1)
        )
        return proj * view
    }
}
