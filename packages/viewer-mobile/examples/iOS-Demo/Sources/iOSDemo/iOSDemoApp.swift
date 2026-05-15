// iOSDemoApp — the minimum a real app needs to render a splat with the SDK.
//
// Drop `bonsai-7k.glb` into `Sources/iOSDemo/Assets/` and rename the
// placeholder; the loader is wired up to look for it in the resource bundle.

import SwiftUI
import SplatforgeViewer

@main
struct iOSDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        if let url = Bundle.module.url(forResource: "bonsai-7k", withExtension: "glb") {
            SplatforgeViewer(assetURL: url)
        } else {
            // Placeholder text shown until you drop the asset in.
            // The decoder + renderer are wired; this is the only missing piece.
            Text("bonsai-7k.glb not bundled — see README")
                .padding()
        }
    }
}
