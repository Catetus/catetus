// swift-tools-version:5.9
//
// iOSDemo — minimal SwiftUI app that loads a bundled `bonsai-7k.glb` and
// renders it with `SplatforgeViewer`.
//
// To run on a real device:
//   open Package.swift  # in Xcode, choose iOSDemo scheme + your iPhone
// To smoke-build on macOS the host-side library compiles already; the iOS
// platform requires Xcode + an iOS SDK (PENDING — not available in this CI).

import PackageDescription

let package = Package(
    name: "iOSDemo",
    platforms: [
        .iOS(.v15),
        .macOS(.v13)
    ],
    products: [
        .library(name: "iOSDemoLib", targets: ["iOSDemo"])
    ],
    dependencies: [
        .package(path: "../../ios")
    ],
    targets: [
        .target(
            name: "iOSDemo",
            dependencies: [
                .product(name: "SplatforgeViewer", package: "ios")
            ],
            path: "Sources/iOSDemo",
            resources: [
                .copy("Assets/bonsai-7k.glb.placeholder"),
                .copy("Assets/synth.glb")
            ]
        )
    ]
)
