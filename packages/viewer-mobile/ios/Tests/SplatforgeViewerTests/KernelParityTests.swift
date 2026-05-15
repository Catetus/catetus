// KernelParityTests.swift
//
// Runtime parity tests for the MSL compute kernels in
// `Sources/SplatforgeViewer/Shaders/`. Each test compiles the relevant .metal
// source against the local Metal device, dispatches it over a fixture input,
// and asserts the GPU output matches a Swift CPU oracle.
//
// The oracle for the radix sort mirrors `packages/viewer-mobile/core/src/sort.rs`
// (back-to-front depth ordering ⇔ ascending u32 key after the bit-flip the
// project kernel applies).
//
// The tests are gated on `MTLCreateSystemDefaultDevice()` being non-nil; on a
// CI runner without a GPU they are SKIPPED rather than failed.

import XCTest
import Metal
import SplatforgeShaders

final class KernelParityTests: XCTestCase {

    // MARK: - Fixtures (8 scenes spanning the input distributions we expect
    // from real splat scenes: tiny, dense, nearly-sorted, reverse-sorted,
    // heavy-duplicates, sparse, large-uniform, small-edge).
    private static let fixtures: [(name: String, keys: [UInt32])] = [
        ("tiny",            (0..<16).map      { _ in UInt32.random(in: 0...UInt32.max) }),
        ("dense_random",    (0..<4096).map    { _ in UInt32.random(in: 0...UInt32.max) }),
        ("nearly_sorted",   (0..<2048).map    { UInt32($0 * 7 &+ UInt32.random(in: 0...3)) }),
        ("reverse_sorted",  (0..<2048).reversed().map { UInt32($0) }),
        ("heavy_duplicates",(0..<4096).map    { _ in UInt32.random(in: 0...15) }),
        ("sparse",          (0..<512).map     { _ in UInt32.random(in: 0...UInt32.max) }),
        ("large_uniform",   Array(repeating: UInt32(0xdeadbeef), count: 4096)),
        ("small_edge",      [0, UInt32.max, 1, UInt32.max - 1, 2, 3, 0, UInt32.max]),
    ]

    // ------------------------------------------------------------------
    // Radix sort parity: GPU result equals stable Swift sort on the key
    // sequence, and the carried indices match the input's stable permutation.
    // ------------------------------------------------------------------
    func testRadixSortParityAcrossFixtures() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available on this host.")
        }
        let library = try Self.compileLibrary(named: "RadixSort", device: device)
        let histPipe = try device.makeComputePipelineState(function: library.makeFunction(name: "cs_histogram")!)
        let scanPipe = try device.makeComputePipelineState(function: library.makeFunction(name: "cs_scan")!)
        let scatPipe = try device.makeComputePipelineState(function: library.makeFunction(name: "cs_scatter")!)

        guard let queue = device.makeCommandQueue() else {
            return XCTFail("makeCommandQueue failed")
        }

        for fx in Self.fixtures {
            try runRadixSortFixture(name: fx.name,
                                    keys: fx.keys,
                                    device: device,
                                    queue: queue,
                                    histPipe: histPipe,
                                    scanPipe: scanPipe,
                                    scatPipe: scatPipe)
        }
    }

    // MARK: - One-fixture runner
    private func runRadixSortFixture(name: String,
                                     keys keysIn: [UInt32],
                                     device: MTLDevice,
                                     queue: MTLCommandQueue,
                                     histPipe: MTLComputePipelineState,
                                     scanPipe: MTLComputePipelineState,
                                     scatPipe: MTLComputePipelineState) throws {
        let count   = keysIn.count
        let wgSize  = 256
        let radix   = 16
        let numWGs  = (count + wgSize - 1) / wgSize
        let valuesIn: [UInt32] = (0..<UInt32(count)).map { $0 }

        // Allocate ping-pong buffers.
        let keysA   = try makeBuffer(device, keysIn)
        let keysB   = try makeBuffer(device, Array(repeating: UInt32(0), count: count))
        let valsA   = try makeBuffer(device, valuesIn)
        let valsB   = try makeBuffer(device, Array(repeating: UInt32(0), count: count))
        let hist    = try makeBuffer(device, Array(repeating: UInt32(0), count: numWGs * radix))

        var inK = keysA, inV = valsA, outK = keysB, outV = valsB

        // 8 passes of 4-bit radix.
        for pass in 0..<8 {
            let shift = UInt32(pass * 4)
            var unif = RadixUniforms(count: UInt32(count),
                                     bit_shift: shift,
                                     num_wgs: UInt32(numWGs),
                                     _pad: 0)
            let unifBuf = device.makeBuffer(bytes: &unif,
                                            length: MemoryLayout<RadixUniforms>.stride,
                                            options: .storageModeShared)!

            // Zero the histogram between passes.
            memset(hist.contents(), 0, numWGs * radix * MemoryLayout<UInt32>.size)

            let cmd = queue.makeCommandBuffer()!

            do {
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(histPipe)
                enc.setBuffer(inK,     offset: 0, index: 0)
                enc.setBuffer(hist,    offset: 0, index: 4)
                enc.setBuffer(unifBuf, offset: 0, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: numWGs, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: wgSize, height: 1, depth: 1))
                enc.endEncoding()
            }
            do {
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(scanPipe)
                enc.setBuffer(hist,    offset: 0, index: 4)
                enc.setBuffer(unifBuf, offset: 0, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: wgSize, height: 1, depth: 1))
                enc.endEncoding()
            }
            do {
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(scatPipe)
                enc.setBuffer(inK,     offset: 0, index: 0)
                enc.setBuffer(inV,     offset: 0, index: 1)
                enc.setBuffer(outK,    offset: 0, index: 2)
                enc.setBuffer(outV,    offset: 0, index: 3)
                enc.setBuffer(hist,    offset: 0, index: 4)
                enc.setBuffer(unifBuf, offset: 0, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: numWGs, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: wgSize, height: 1, depth: 1))
                enc.endEncoding()
            }

            cmd.commit()
            cmd.waitUntilCompleted()

            // Swap.
            swap(&inK, &outK)
            swap(&inV, &outV)
        }

        // After 8 passes, `inK` / `inV` hold the final sorted output (swap on
        // pass 7 left them in the source-side buffers).
        let gpuKeys   = Array(UnsafeBufferPointer(start: inK.contents().assumingMemoryBound(to: UInt32.self), count: count))
        let gpuValues = Array(UnsafeBufferPointer(start: inV.contents().assumingMemoryBound(to: UInt32.self), count: count))

        // Oracle: stable Swift sort by key ASC, with the original index as
        // tiebreaker (matches LSD radix's stable-by-original-order semantics).
        let oracle = keysIn.enumerated()
            .sorted { lhs, rhs in
                if lhs.element != rhs.element { return lhs.element < rhs.element }
                return lhs.offset < rhs.offset
            }

        let oracleKeys   = oracle.map { $0.element }
        let oracleValues = oracle.map { UInt32($0.offset) }

        XCTAssertEqual(gpuKeys, oracleKeys,
                       "fixture=\(name): GPU radix-sorted keys mismatch CPU oracle")
        XCTAssertEqual(gpuValues, oracleValues,
                       "fixture=\(name): GPU carried payload mismatch CPU oracle")
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    private func makeBuffer(_ device: MTLDevice, _ data: [UInt32]) throws -> MTLBuffer {
        let byteLen = data.count * MemoryLayout<UInt32>.size
        guard let buf = device.makeBuffer(length: max(byteLen, 16), options: .storageModeShared) else {
            throw NSError(domain: "ParityTests", code: 1)
        }
        if !data.isEmpty {
            memcpy(buf.contents(), data, byteLen)
        }
        return buf
    }

    /// Resolve and compile a .metal source by name. We compile from source at
    /// runtime so the test does not depend on `metallib` build steps; the
    /// .metal files are copied into the SplatforgeViewer resource bundle by
    /// `Package.swift`'s `.process("Shaders")` rule.
    private static func compileLibrary(named name: String, device: MTLDevice) throws -> MTLLibrary {
        // The .metal sources live in the `SplatforgeShaders` resource bundle.
        guard let src = SplatforgeShaders.source(forKernel: name) else {
            throw NSError(domain: "ParityTests", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "missing \(name).metal in SplatforgeShaders bundle"])
        }
        return try device.makeLibrary(source: src, options: nil)
    }

    private struct RadixUniforms {
        var count: UInt32
        var bit_shift: UInt32
        var num_wgs: UInt32
        var _pad: UInt32
    }
}
