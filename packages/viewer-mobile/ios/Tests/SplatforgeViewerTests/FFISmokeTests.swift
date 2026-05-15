// FFISmokeTests.swift
//
// End-to-end smoke test that exercises the Rust FFI surface exported by the
// `SplatforgeViewerCore.xcframework` binary target. This is the first test in
// the iOS package that actually links against the Rust staticlib — if the
// XCFramework is missing a slice for the host platform, this test will fail
// at runtime symbol lookup, surfacing the breakage immediately.
//
// We don't ship a real .glb fixture (the bonsai-7k asset is ~22 MB and lives
// outside the repo). Instead we use the workspace's `splatforge-gltf` writer
// indirectly: the Swift side simply pokes the FFI with garbage bytes and
// asserts the error paths return the documented status codes. That's enough
// to prove the symbol is wired and ABI-compatible.

import XCTest
import SplatforgeViewerC

final class FFISmokeTests: XCTestCase {

    /// `sfmv_vertex_stride()` returns the 56-byte SplatVertex stride. This is
    /// the cheapest possible call into the staticlib — if it returns anything
    /// other than 56 the C ABI is wrong (struct padding, packed vs not, etc.).
    func testVertexStrideMatches() {
        XCTAssertEqual(sfmv_vertex_stride(), 56,
                       "SplatVertex must be 56 bytes (3+4+3+1+3 floats)")
    }

    /// Decoding null pointer must yield the documented error code, not crash.
    func testDecodeNullReturnsNullPointerStatus() {
        var handle: OpaquePointer?
        let status = sfmv_decode_glb(nil, 0, &handle)
        XCTAssertEqual(status, SfmvStatusNullPointer)
        XCTAssertNil(handle)
    }

    /// Decoding garbage bytes returns `DecodeFailed`, not a crash, and writes
    /// NULL to the out pointer.
    func testDecodeGarbageReturnsDecodeFailed() {
        // "not a glb" — would also fail magic check. Use a non-empty slice so
        // we exercise the slice → glb parser path, not the null guard.
        var bytes: [UInt8] = Array("not a glb".utf8)
        var handle: OpaquePointer?
        let status = bytes.withUnsafeBufferPointer { buf -> SfmvStatus in
            return sfmv_decode_glb(buf.baseAddress, buf.count, &handle)
        }
        XCTAssertEqual(status, SfmvStatusDecodeFailed)
        XCTAssertNil(handle)
    }

    /// `sfmv_buffer_len` and `sfmv_buffer_data` must tolerate a NULL handle
    /// (return 0 / NULL respectively, never crash).
    func testBufferAccessorsTolerateNull() {
        XCTAssertEqual(sfmv_buffer_len(nil), 0)
        XCTAssertNil(sfmv_buffer_data(nil))
    }
}
