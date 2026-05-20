// CatetusShaders — resource-only Swift target that hosts the MSL compute
// kernels (`*.metal`) for the iOS viewer. Both the main `CatetusViewer`
// target and the `CatetusViewerTests` target depend on this so the
// shaders live in exactly one place on disk and SwiftPM auto-generates
// a `Bundle.module` for resource lookup.

import Foundation

/// Resource accessor for the bundled MSL kernel sources.
public enum CatetusShaders {
    /// The package's resource bundle. Resources are produced by the
    /// `resources: [.process("Shaders")]` rule in Package.swift.
    public static var bundle: Bundle { .module }

    /// Read a kernel source file by base name (e.g. "RadixSort"). Returns the
    /// .metal source text, or `nil` if the file is not in the bundle.
    public static func source(forKernel name: String) -> String? {
        guard let url = bundle.url(forResource: name, withExtension: "metal") else {
            return nil
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    /// URL of a kernel source file, or `nil` if missing.
    public static func url(forKernel name: String) -> URL? {
        bundle.url(forResource: name, withExtension: "metal")
    }
}
