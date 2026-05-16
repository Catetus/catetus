// QATPlyHeaderParser.swift — pure-Swift parser for the
// `comment quantized_field ...` lines in a QAT-PLY v1 header. Logic
// matches the C reference parser at apps/codec/qat-ply-c/qat_ply_decode.c
// — same fields, same defaults, same error behaviour.
//
// SPDX-License-Identifier: MIT

import Foundation

public enum QATPlyDtype: Int, Sendable {
    case int8 = 8
    case int4 = 4
}

public enum QATPlyScaleKind: Int, Sendable {
    case perChannel = 0
    case perAnchor  = 1
}

public struct QATPlyField: Sendable, Equatable {
    public let name: String
    public let dtype: QATPlyDtype
    public let channels: Int
    public let scaleKind: QATPlyScaleKind
    public let packedPerByte: Int
    /// Byte offset of the scale_b64=... payload inside the original header.
    public let scaleB64Offset: Int
    public let scaleB64Length: Int
}

public struct QATPlyHeader: Sendable {
    public let fields: [QATPlyField]
    /// Number of anchors (`element vertex N`).
    public let anchorCount: Int
    /// Properties of the vertex element in declaration order.
    public let vertexProperties: [QATPlyProperty]
    /// Byte offset of the body (first byte after `end_header\n`).
    public let bodyOffset: Int
}

public struct QATPlyProperty: Sendable, Equatable {
    public let name: String
    public let type: String
    public let size: Int
}

public enum QATPlyError: Error, Equatable {
    case missingEndHeader
    case badHeader(String)
    case badBase64
    case channelMismatch
    case missingProperty(String)
}

public enum QATPlyHeaderParser {

    private static let typeSizes: [String: Int] = [
        "char": 1, "uchar": 1, "short": 2, "ushort": 2,
        "int": 4, "uint": 4, "float": 4, "double": 8, "float32": 4,
    ]

    /// Parse a full PLY file's header (everything up to and including
    /// `end_header\n`). Caller owns `data` — we do not retain it.
    public static func parse(_ data: Data) throws -> QATPlyHeader {
        guard let bodyOffset = findEndHeader(data) else {
            throw QATPlyError.missingEndHeader
        }
        let headerData = data.subdata(in: 0..<bodyOffset)
        guard let headerStr = String(data: headerData, encoding: .ascii) else {
            throw QATPlyError.badHeader("header is not ASCII")
        }

        var fields: [QATPlyField] = []
        var props: [QATPlyProperty] = []
        var anchorCount = 0
        var inVertex = false

        var scan = headerStr.startIndex
        var lineStartByte = 0
        while scan < headerStr.endIndex {
            let lineEnd = headerStr[scan...].firstIndex(of: "\n") ?? headerStr.endIndex
            let line = headerStr[scan..<lineEnd]
            let lineByteLen = line.utf8.count
            let lineStr = String(line)

            if lineStr.hasPrefix("element vertex ") {
                let nStr = String(lineStr.dropFirst("element vertex ".count))
                anchorCount = Int(nStr.trimmingCharacters(in: .whitespaces)) ?? 0
                inVertex = true
            } else if lineStr.hasPrefix("element ") {
                inVertex = false
            } else if inVertex && lineStr.hasPrefix("property ") {
                let parts = lineStr.dropFirst("property ".count)
                    .split(separator: " ", maxSplits: 2, omittingEmptySubsequences: true)
                if parts.count >= 2 {
                    let type = String(parts[0])
                    let name = String(parts[1])
                    guard let sz = typeSizes[type] else {
                        throw QATPlyError.badHeader("unknown property type \(type)")
                    }
                    props.append(QATPlyProperty(name: name, type: type, size: sz))
                }
            } else if lineStr.hasPrefix("comment quantized_field ") {
                let field = try parseQuantizedFieldLine(
                    lineStr,
                    lineStartByte: lineStartByte
                )
                fields.append(field)
            }
            // Advance: line + the '\n' (1 byte).
            lineStartByte += lineByteLen + 1
            scan = lineEnd
            if scan < headerStr.endIndex {
                scan = headerStr.index(after: scan)
            }
        }

        return QATPlyHeader(
            fields: fields,
            anchorCount: anchorCount,
            vertexProperties: props,
            bodyOffset: bodyOffset
        )
    }

    /// Decode a base64 string into a [Float]. Matches
    /// `qat_ply_base64_decode_fp32` semantics (LE fp32, no whitespace).
    public static func base64DecodeFloats(_ b64: String) throws -> [Float] {
        guard let raw = Data(base64Encoded: b64) else {
            throw QATPlyError.badBase64
        }
        guard raw.count % 4 == 0 else { throw QATPlyError.badBase64 }
        var out = [Float](repeating: 0, count: raw.count / 4)
        _ = out.withUnsafeMutableBytes { mutPtr in
            raw.copyBytes(to: mutPtr)
        }
        return out
    }

    /// Find byte offset of the first byte *after* "\nend_header\n".
    public static func findEndHeader(_ data: Data) -> Int? {
        let needle: [UInt8] = Array("\nend_header\n".utf8)
        return data.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> Int? in
            guard let base = buf.baseAddress else { return nil }
            let bytes = base.assumingMemoryBound(to: UInt8.self)
            let n = buf.count
            if n < needle.count { return nil }
            for i in 0...(n - needle.count) {
                var match = true
                for j in 0..<needle.count where bytes[i + j] != needle[j] {
                    match = false; break
                }
                if match { return i + needle.count }
            }
            return nil
        }
    }

    // MARK: - quantized_field line parser

    private static func parseQuantizedFieldLine(
        _ line: String,
        lineStartByte: Int
    ) throws -> QATPlyField {
        // Format:
        //   comment quantized_field <NAME> int<BITS> channels=<C> [packed_per_byte=2] [scale_kind=per_anchor] [scale_b64=...]
        let payload = String(line.dropFirst("comment quantized_field ".count))
        let toks = payload.split(separator: " ", omittingEmptySubsequences: true)
        guard toks.count >= 3 else {
            throw QATPlyError.badHeader("quantized_field too short: \(line)")
        }
        let name = String(toks[0])
        let typeTok = String(toks[1])
        guard typeTok.hasPrefix("int"),
              let bits = Int(typeTok.dropFirst(3))
        else {
            throw QATPlyError.badHeader("bad type \(typeTok)")
        }
        guard let dtype = QATPlyDtype(rawValue: bits) else {
            throw QATPlyError.badHeader("unsupported dtype int\(bits)")
        }

        var channels = 0
        var packedPerByte = (dtype == .int4) ? 2 : 0
        var scaleKind: QATPlyScaleKind = .perChannel
        var scaleB64Offset = 0
        var scaleB64Length = 0

        let prefix = "comment quantized_field \(name) \(typeTok) "
        var cursorByte = lineStartByte + prefix.utf8.count

        for tok in toks.dropFirst(2) {
            let tokenStr = String(tok)
            let tokenByteLen = tokenStr.utf8.count

            if tokenStr.hasPrefix("channels=") {
                channels = Int(tokenStr.dropFirst("channels=".count)) ?? 0
            } else if tokenStr.hasPrefix("packed_per_byte=") {
                packedPerByte = Int(tokenStr.dropFirst("packed_per_byte=".count)) ?? 2
            } else if tokenStr.hasPrefix("scale_kind=") {
                let v = String(tokenStr.dropFirst("scale_kind=".count))
                scaleKind = (v == "per_anchor") ? .perAnchor : .perChannel
            } else if tokenStr.hasPrefix("scale_b64=") {
                let valBytes = tokenByteLen - "scale_b64=".count
                scaleB64Offset = cursorByte + "scale_b64=".count
                scaleB64Length = valBytes
            }
            cursorByte += tokenByteLen + 1  // token + space separator
        }

        if channels <= 0 {
            throw QATPlyError.badHeader("missing channels= for \(name)")
        }
        return QATPlyField(
            name: name, dtype: dtype, channels: channels,
            scaleKind: scaleKind, packedPerByte: packedPerByte,
            scaleB64Offset: scaleB64Offset,
            scaleB64Length: scaleB64Length
        )
    }
}
