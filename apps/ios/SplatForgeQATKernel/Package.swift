// swift-tools-version:5.9
//
// SplatForgeQATKernel — Metal-direct on-device decoder for the QAT-PLY v1
// wire format. Drop-in module for the existing SplatForge iOS viewer.
//
// SPDX-License-Identifier: MIT

import PackageDescription

let package = Package(
    name: "SplatForgeQATKernel",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "SplatForgeQATKernel",
            targets: ["SplatForgeQATKernel"]
        )
    ],
    targets: [
        .target(
            name: "SplatForgeQATKernel",
            path: "Sources/SplatForgeQATKernel",
            resources: [
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "SplatForgeQATKernelTests",
            dependencies: ["SplatForgeQATKernel"],
            path: "Tests/SplatForgeQATKernelTests",
            resources: [
                .copy("Fixtures")
            ]
        )
    ]
)
