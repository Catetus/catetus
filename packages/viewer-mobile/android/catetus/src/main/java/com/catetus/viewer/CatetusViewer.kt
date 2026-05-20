// CatetusViewer — Android entry point.
//
// Public API mirrors the iOS Swift SDK: instantiate the view, hand it a
// `Uri` pointing at a `.glb` (file:// or content://), and it renders.
//
// Internally we extend `GLSurfaceView` and drive an `CatetusRenderer`
// that calls into the Rust core through `CatetusNative`. The Vulkan
// compute follow-up will swap in a `SurfaceView` + Vulkan renderer behind
// the same Kotlin surface.

package com.catetus.viewer

import android.content.Context
import android.net.Uri
import android.opengl.GLSurfaceView
import android.util.AttributeSet

/**
 * Drop-in view that renders a Gaussian-splat `.glb` (`KHR_gaussian_splatting`).
 *
 * Typical usage from an Activity/Fragment:
 * ```kotlin
 * val viewer = CatetusViewer(context).apply {
 *     setAsset(Uri.parse("file:///android_asset/bonsai-7k.glb"))
 * }
 * setContentView(viewer)
 * ```
 */
class CatetusViewer @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    private val renderer = CatetusRenderer(context)

    init {
        setEGLContextClientVersion(3)
        setEGLConfigChooser(8, 8, 8, 8, 24, 0)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    /** Load a `.glb` from any URI the app's `ContentResolver` can open. */
    fun setAsset(uri: Uri) {
        renderer.queueAsset(uri)
    }
}
