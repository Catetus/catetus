import Foundation

extension Foundation.Bundle {
    static let module: Bundle = {
        let mainPath = Bundle.main.bundleURL.appendingPathComponent("SplatforgeViewer_SplatforgeViewer.bundle").path
        let buildPath = "/Users/montabano1/Desktop/sf-mobile-wt/packages/viewer-mobile/ios/.build/arm64-apple-macosx/debug/SplatforgeViewer_SplatforgeViewer.bundle"

        let preferredBundle = Bundle(path: mainPath)

        guard let bundle = preferredBundle ?? Bundle(path: buildPath) else {
            // Users can write a function called fatalError themselves, we should be resilient against that.
            Swift.fatalError("could not load resource bundle: from \(mainPath) or \(buildPath)")
        }

        return bundle
    }()
}