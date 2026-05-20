// QATPlyDecoderTests.swift — XCTest suite that runs the Metal decoder
// against the same 10 conformance fixtures as the C/Python references
// and asserts byte-exact equality with the JSON expectations.
//
// SPDX-License-Identifier: MIT

import XCTest
import Metal
@testable import CatetusQATKernel

private struct ConformanceFile: Decodable {
    let version: Int
    let cases: [Case]
    struct Case: Decodable {
        let name: String
        let n_anchors: Int
        let fields: [String: Field]
    }
    struct Field: Decodable {
        let shape: [Int]
        let expected_fp32_b64: String
    }
}

final class QATPlyDecoderTests: XCTestCase {

    private func loadConformance() throws -> ConformanceFile {
        let bundle = Bundle.module
        guard let url = bundle.url(forResource: "conformance",
                                   withExtension: "json",
                                   subdirectory: "Fixtures") else {
            XCTFail("conformance.json not found in test bundle")
            fatalError()
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ConformanceFile.self, from: data)
    }

    private func fixtureData(_ name: String) throws -> Data {
        guard let url = Bundle.module.url(forResource: name,
                                          withExtension: nil,
                                          subdirectory: "Fixtures") else {
            XCTFail("fixture \(name) not found")
            fatalError()
        }
        return try Data(contentsOf: url)
    }

    func testHeaderParserBasic() throws {
        let data = try fixtureData("case01_int8_1x1.ply")
        let hdr = try QATPlyHeaderParser.parse(data)
        XCTAssertEqual(hdr.anchorCount, 1)
        XCTAssertGreaterThan(hdr.fields.count, 0)
        XCTAssertEqual(hdr.fields.first?.dtype, .int8)
    }

    func testBase64Decode() throws {
        // "AACoQQ==" -> [21.0]
        let v = try QATPlyHeaderParser.base64DecodeFloats("AACoQQ==")
        XCTAssertEqual(v, [21.0])
    }

    func testFullConformance() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("no Metal device on this machine")
        }
        let decoder = try QATPlyDecoder()
        let conf = try loadConformance()
        XCTAssertEqual(conf.cases.count, 10, "expected 10 conformance cases")

        for c in conf.cases {
            let data = try fixtureData(c.name)
            guard let bodyOff = QATPlyHeaderParser.findEndHeader(data) else {
                XCTFail("\(c.name): no end_header")
                continue
            }
            let headerBytes = data.subdata(in: 0..<bodyOff)
            let body        = data.subdata(in: bodyOff..<data.count)
            let hdr         = try QATPlyHeaderParser.parse(data)
            XCTAssertEqual(hdr.anchorCount, c.n_anchors, "\(c.name): anchor count")

            let decoded = try decoder.decodeAll(header: hdr,
                                                 headerBytes: headerBytes,
                                                 body: body)
            // Verify every declared field matches expectations byte-for-byte.
            for f in decoded {
                guard let exp = c.fields[f.name] else { continue }
                let expVals = try QATPlyHeaderParser.base64DecodeFloats(exp.expected_fp32_b64)
                XCTAssertEqual(f.values.count, expVals.count,
                               "\(c.name)/\(f.name): length")
                f.values.withUnsafeBufferPointer { aBuf in
                    expVals.withUnsafeBufferPointer { eBuf in
                        let aBytes = UnsafeRawBufferPointer(aBuf)
                        let eBytes = UnsafeRawBufferPointer(eBuf)
                        XCTAssertEqual(aBytes.count, eBytes.count)
                        for i in 0..<aBytes.count where aBytes[i] != eBytes[i] {
                            XCTFail("\(c.name)/\(f.name): byte \(i) mismatch \(aBytes[i]) vs \(eBytes[i])")
                            return
                        }
                    }
                }
            }
        }
    }
}
