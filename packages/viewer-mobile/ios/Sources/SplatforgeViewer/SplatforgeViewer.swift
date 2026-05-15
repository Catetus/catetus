// SplatforgeViewer — SwiftUI entry point.
//
// Drop this view anywhere in a SwiftUI hierarchy with a `.glb` URL and you get
// a Metal-backed splat viewer. Phase-1 renderer is a point-sprite quad pass
// (one textured quad per splat, fragment shader = soft round alpha). The
// compute-shader 2D-cov projection + radix-sort port is the follow-up PR.
//
// The package builds on iOS, iPadOS, and macOS — we abstract the platform
// view-representable shim with `PlatformViewRepresentable` so the same
// SwiftUI surface works everywhere.

import SwiftUI
import MetalKit
import SplatforgeViewerC

#if canImport(UIKit)
import UIKit
typealias PlatformMetalContainer = UIViewRepresentable
#elseif canImport(AppKit)
import AppKit
typealias PlatformMetalContainer = NSViewRepresentable
#endif

/// Top-level SwiftUI view. The `assetURL` should be a `.glb` with the
/// `KHR_gaussian_splatting` extension. The view loads asynchronously and
/// displays an empty Metal-clear color until decode completes.
public struct SplatforgeViewer: View {
    /// `.glb` location. `file://` and `https://` are both fine; the loader
    /// reads bytes into memory before handing them to the Rust core.
    public let assetURL: URL

    /// SwiftUI initializer.
    public init(assetURL: URL) {
        self.assetURL = assetURL
    }

    public var body: some View {
        SplatforgeMetalView(assetURL: assetURL)
    }
}

/// Platform-agnostic `*ViewRepresentable` wrapper. Owns the `MTKView` and the
/// Rust buffer handle. We make a dedicated `Coordinator` so the renderer can
/// survive SwiftUI redraws without re-decoding the asset.
struct SplatforgeMetalView: PlatformMetalContainer {
    let assetURL: URL

    func makeCoordinator() -> SplatforgeRenderer {
        SplatforgeRenderer()
    }

    private func buildMTKView(coordinator: SplatforgeRenderer) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.delegate = coordinator
        view.preferredFramesPerSecond = 60
        coordinator.attach(to: view)
        coordinator.load(url: assetURL)
        return view
    }

#if canImport(UIKit)
    func makeUIView(context: Context) -> MTKView {
        let v = buildMTKView(coordinator: context.coordinator)
        // Touch-driven orbit + pinch zoom. Multiple recognizers can drive the
        // same coordinator; the renderer accumulates azimuth / elevation /
        // distance state.
        let pan = UIPanGestureRecognizer(target: context.coordinator,
                                         action: #selector(SplatforgeRenderer.handlePan(_:)))
        let pinch = UIPinchGestureRecognizer(target: context.coordinator,
                                             action: #selector(SplatforgeRenderer.handlePinch(_:)))
        v.addGestureRecognizer(pan)
        v.addGestureRecognizer(pinch)
        return v
    }
    func updateUIView(_ uiView: MTKView, context: Context) {}
#elseif canImport(AppKit)
    func makeNSView(context: Context) -> MTKView {
        let v = buildMTKView(coordinator: context.coordinator)
        let pan = NSPanGestureRecognizer(target: context.coordinator,
                                         action: #selector(SplatforgeRenderer.handleMacPan(_:)))
        let mag = NSMagnificationGestureRecognizer(target: context.coordinator,
                                                   action: #selector(SplatforgeRenderer.handleMacMagnify(_:)))
        v.addGestureRecognizer(pan)
        v.addGestureRecognizer(mag)
        return v
    }
    func updateNSView(_ nsView: MTKView, context: Context) {}
#endif
}
