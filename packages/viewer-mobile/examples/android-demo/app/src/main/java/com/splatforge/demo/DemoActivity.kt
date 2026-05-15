// DemoActivity — wires `SplatforgeViewer` to a bundled `bonsai-7k.glb`
// shipped in `app/src/main/assets/`. The asset is too large to vendor; drop
// the real `.glb` next to `bonsai-7k.glb.placeholder` before running.

package com.splatforge.demo

import android.app.Activity
import android.net.Uri
import android.os.Bundle
import com.splatforge.viewer.SplatforgeViewer

class DemoActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewer = SplatforgeViewer(this).apply {
            setAsset(Uri.parse("file:///android_asset/bonsai-7k.glb"))
        }
        setContentView(viewer)
    }
}
