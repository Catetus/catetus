// QATPlyDecoder.swift — Swift facade over the Metal compute kernels
// declared in Shaders/QATPlyDequant.metal. The decoder takes a parsed
// QAT-PLY header + raw body bytes and returns dequantized fp32 buffers,
// one per quantized_field declared in the header.
//
// SPDX-License-Identifier: MIT

import Foundation
import Metal

public enum QATPlyDecoderError: Error {
    case noMetalDevice
    case shaderLibraryMissing
    case pipelineFailed(String)
    case bufferAllocationFailed
    case commandBufferFailed(Error?)
    case propertyTableMismatch(String)
    case scaleCountMismatch
}

/// Output of a single field's dequantization, sized N×C.
public struct QATPlyDecodedField {
    public let name: String
    public let rows: Int
    public let channels: Int
    /// Row-major fp32 values of length rows*channels.
    public let values: [Float]
}

public final class QATPlyDecoder {

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelineInt8: MTLComputePipelineState
    private let pipelineInt4: MTLComputePipelineState

    /// Convenience init using the system default Metal device.
    public convenience init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw QATPlyDecoderError.noMetalDevice
        }
        try self.init(device: dev)
    }

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw QATPlyDecoderError.pipelineFailed("makeCommandQueue")
        }
        self.queue = queue

        // Locate the .metallib that SwiftPM emits for the Shaders resource.
        let library = try Self.loadLibrary(device: device)

        guard let int8Fn = library.makeFunction(name: "qat_dequant_int8"),
              let int4Fn = library.makeFunction(name: "qat_dequant_int4_packed") else {
            throw QATPlyDecoderError.shaderLibraryMissing
        }
        self.pipelineInt8 = try device.makeComputePipelineState(function: int8Fn)
        self.pipelineInt4 = try device.makeComputePipelineState(function: int4Fn)
    }

    private static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        // SwiftPM compiles bundled .metal files into a default.metallib
        // inside the module bundle. On Apple platforms `Bundle.module`
        // exists; on plain macOS swift test, the test bundle resolves to
        // the package bundle too.
        let bundle = Bundle.module
        if let url = bundle.url(forResource: "default", withExtension: "metallib"),
           let lib = try? device.makeLibrary(URL: url) {
            return lib
        }
        // Fallback: device.makeDefaultLibrary(bundle:) — works when the
        // bundle's Resources directly contain the metallib.
        if let lib = try? device.makeDefaultLibrary(bundle: bundle) {
            return lib
        }
        // Last-resort fallback: compile from source at runtime. This
        // keeps `swift test` green on machines where the SwiftPM Metal
        // pipeline didn't pre-compile the .metallib (e.g. some macOS
        // CLI Swift versions).
        let metalSrc = try Self.readEmbeddedShaderSource(bundle: bundle)
        return try device.makeLibrary(source: metalSrc, options: nil)
    }

    private static func readEmbeddedShaderSource(bundle: Bundle) throws -> String {
        if let url = bundle.url(forResource: "QATPlyDequant", withExtension: "metal"),
           let s = try? String(contentsOf: url, encoding: .utf8) {
            return s
        }
        throw QATPlyDecoderError.shaderLibraryMissing
    }

    // MARK: - High-level API

    /// Decode every quantized_field declared in the header. Returns one
    /// QATPlyDecodedField per field, in declaration order.
    public func decodeAll(
        header: QATPlyHeader,
        headerBytes: Data,
        body: Data
    ) throws -> [QATPlyDecodedField] {
        var out: [QATPlyDecodedField] = []
        let rowSize = header.vertexProperties.reduce(0) { $0 + $1.size }
        var propOff: [String: Int] = [:]
        var o = 0
        for p in header.vertexProperties { propOff[p.name] = o; o += p.size }

        for f in header.fields {
            let dec = try decodeOne(
                field: f,
                anchorCount: header.anchorCount,
                rowSize: rowSize,
                propertyOffsets: propOff,
                headerBytes: headerBytes,
                body: body
            )
            out.append(dec)
        }
        return out
    }

    /// Decode a single field given the parsed property table and raw body.
    public func decodeOne(
        field: QATPlyField,
        anchorCount: Int,
        rowSize: Int,
        propertyOffsets: [String: Int],
        headerBytes: Data,
        body: Data
    ) throws -> QATPlyDecodedField {
        let N = anchorCount
        let C = field.channels

        switch field.dtype {
        case .int8:
            // Pull C int8 columns.
            var q = [Int8](repeating: 0, count: N * C)
            for ci in 0..<C {
                let col = "\(field.name)_q_\(ci)"
                guard let off = propertyOffsets[col] else {
                    throw QATPlyDecoderError.propertyTableMismatch(col)
                }
                for r in 0..<N {
                    q[r * C + ci] = Int8(bitPattern: body[r * rowSize + off])
                }
            }
            // Decode scales from header.
            let b64Str = headerBytes.subdata(in: field.scaleB64Offset..<(field.scaleB64Offset + field.scaleB64Length))
            guard let b64 = String(data: b64Str, encoding: .ascii) else {
                throw QATPlyDecoderError.scaleCountMismatch
            }
            let scales = try QATPlyHeaderParser.base64DecodeFloats(b64)
            if scales.count != C { throw QATPlyDecoderError.scaleCountMismatch }
            let values = try dispatchInt8(q: q, scale: scales, nRows: N, nChannels: C)
            return QATPlyDecodedField(name: field.name, rows: N, channels: C, values: values)

        case .int4:
            let B = (C + 1) / 2
            var packed = [UInt8](repeating: 0, count: N * B)
            for bi in 0..<B {
                let col = "\(field.name)_q_\(bi)"
                guard let off = propertyOffsets[col] else {
                    throw QATPlyDecoderError.propertyTableMismatch(col)
                }
                for r in 0..<N {
                    packed[r * B + bi] = body[r * rowSize + off]
                }
            }
            // Per-anchor scale lives in a separate fp32 column.
            let scaleCol = "\(field.name)_scale"
            guard let scaleOff = propertyOffsets[scaleCol] else {
                throw QATPlyDecoderError.propertyTableMismatch(scaleCol)
            }
            var scales = [Float](repeating: 0, count: N)
            for r in 0..<N {
                let base = r * rowSize + scaleOff
                let raw =
                    UInt32(body[base + 0])
                  | (UInt32(body[base + 1]) << 8)
                  | (UInt32(body[base + 2]) << 16)
                  | (UInt32(body[base + 3]) << 24)
                scales[r] = Float(bitPattern: raw)
            }
            let values = try dispatchInt4(packed: packed, scale: scales, nRows: N, nChannels: C)
            return QATPlyDecodedField(name: field.name, rows: N, channels: C, values: values)
        }
    }

    // MARK: - Metal dispatch

    private func dispatchInt8(q: [Int8], scale: [Float], nRows: Int, nChannels: Int) throws -> [Float] {
        let total = nRows * nChannels
        guard
            let qBuf  = device.makeBuffer(bytes: q,  length: total, options: .storageModeShared),
            let sBuf  = device.makeBuffer(bytes: scale, length: nChannels * MemoryLayout<Float>.size, options: .storageModeShared),
            let oBuf  = device.makeBuffer(length: total * MemoryLayout<Float>.size, options: .storageModeShared),
            let cmd   = queue.makeCommandBuffer(),
            let enc   = cmd.makeComputeCommandEncoder()
        else {
            throw QATPlyDecoderError.bufferAllocationFailed
        }
        enc.setComputePipelineState(pipelineInt8)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(sBuf, offset: 0, index: 1)
        enc.setBuffer(oBuf, offset: 0, index: 2)
        var dims = SIMD2<UInt32>(UInt32(nRows), UInt32(nChannels))
        enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)

        let tgw = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: nRows, height: nChannels, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgw)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { throw QATPlyDecoderError.commandBufferFailed(err) }

        var out = [Float](repeating: 0, count: total)
        let ptr = oBuf.contents().bindMemory(to: Float.self, capacity: total)
        for i in 0..<total { out[i] = ptr[i] }
        return out
    }

    private func dispatchInt4(packed: [UInt8], scale: [Float], nRows: Int, nChannels: Int) throws -> [Float] {
        let B = (nChannels + 1) / 2
        let total = nRows * nChannels
        guard
            let pBuf = device.makeBuffer(bytes: packed, length: nRows * B, options: .storageModeShared),
            let sBuf = device.makeBuffer(bytes: scale,  length: nRows * MemoryLayout<Float>.size, options: .storageModeShared),
            let oBuf = device.makeBuffer(length: total * MemoryLayout<Float>.size, options: .storageModeShared),
            let cmd  = queue.makeCommandBuffer(),
            let enc  = cmd.makeComputeCommandEncoder()
        else {
            throw QATPlyDecoderError.bufferAllocationFailed
        }
        enc.setComputePipelineState(pipelineInt4)
        enc.setBuffer(pBuf, offset: 0, index: 0)
        enc.setBuffer(sBuf, offset: 0, index: 1)
        enc.setBuffer(oBuf, offset: 0, index: 2)
        var dims = SIMD2<UInt32>(UInt32(nRows), UInt32(nChannels))
        enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)

        let tgw = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: nRows, height: nChannels, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgw)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { throw QATPlyDecoderError.commandBufferFailed(err) }

        var out = [Float](repeating: 0, count: total)
        let ptr = oBuf.contents().bindMemory(to: Float.self, capacity: total)
        for i in 0..<total { out[i] = ptr[i] }
        return out
    }
}
