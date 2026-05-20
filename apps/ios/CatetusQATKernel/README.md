# CatetusQATKernel

Metal-direct on-device decoder for the [QAT-PLY v1](../../codec/qat-ply-c/)
wire format. Drop-in SwiftPM module for the Catetus iOS viewer.

- One thread per (anchor × channel) on a Metal compute pipeline.
- int8 + per-channel scale: 32-channel f_anchor_feat dequantizes in a
  single dispatch.
- int4-packed (2 nibbles per byte) + per-anchor scale: matches the C
  reference byte-for-byte (signed shift `nibble - 8` on-device).
- Pure-Swift header parser — no `@_silgen_name` or C bridge needed.

## Install (SwiftPM)

```swift
// In your Package.swift dependencies:
.package(url: "https://github.com/catetus/Catetus.git",
         from: "0.1.0", path: "apps/ios/CatetusQATKernel")
```

Or in Xcode: File > Add Packages > paste the Catetus repo URL, pick
`CatetusQATKernel`.

## Build & test

```bash
cd apps/ios/CatetusQATKernel
swift build       # macOS host build (no iOS simulator required)
swift test        # runs XCTest against MTLCreateSystemDefaultDevice()
```

The XCTest suite runs all 10 conformance fixtures and asserts byte-exact
fp32 equality with the JSON expectations published at
`apps/codec/conformance/conformance.json`.

## Usage

```swift
import CatetusQATKernel

let data = try Data(contentsOf: plyURL)
let hdr  = try QATPlyHeaderParser.parse(data)
guard let bodyOff = QATPlyHeaderParser.findEndHeader(data) else { ... }

let decoder = try QATPlyDecoder()  // uses MTLCreateSystemDefaultDevice
let fields  = try decoder.decodeAll(
    header: hdr,
    headerBytes: data.subdata(in: 0..<bodyOff),
    body:        data.subdata(in: bodyOff..<data.count)
)

// fields == [QATPlyDecodedField], one per `comment quantized_field ...` line.
// Each .values is row-major fp32 of length rows*channels.
```

## Output binary size

The compiled `.metallib` is ~3 KB (two kernels, no helpers). The Swift
module is ~25 KB after `swift build -c release`. Both ship as part of
the SwiftPM target — no extra bundling required.

## License

MIT.
