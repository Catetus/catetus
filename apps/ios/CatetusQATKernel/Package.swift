// swift-tools-version:5.9
//
// CatetusQATKernel — Metal-direct on-device decoder for the QAT-PLY v1
// wire format. Drop-in module for the existing Catetus iOS viewer.
//
// SPDX-License-Identifier: MIT

import PackageDescription

let package = Package(
    name: "CatetusQATKernel",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "CatetusQATKernel",
            targets: ["CatetusQATKernel"]
        )
    ],
    targets: [
        .target(
            name: "CatetusQATKernel",
            path: "Sources/CatetusQATKernel",
            resources: [
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "CatetusQATKernelTests",
            dependencies: ["CatetusQATKernel"],
            path: "Tests/CatetusQATKernelTests",
            resources: [
                .copy("Fixtures")
            ]
        )
    ]
)
