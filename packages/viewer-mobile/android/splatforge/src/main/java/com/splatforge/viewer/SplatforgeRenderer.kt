// SplatforgeRenderer — GLES 3.1 point-sprite renderer.
//
// Mirrors the Phase-1 Metal renderer on iOS: load .glb, build a vertex buffer,
// draw N instanced quads. The shaders live in `res/raw/splat_*.glsl` and are
// inlined at build time (no asset roundtrip).
//
// Compute kernels (radix sort, 2D-cov projection) are PENDING — see the
// `STUB` `.glsl` files alongside.

package com.splatforge.viewer

import android.content.Context
import android.net.Uri
import android.opengl.GLES31
import android.opengl.GLSurfaceView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

internal class SplatforgeRenderer(private val context: Context) : GLSurfaceView.Renderer {
    @Volatile private var pendingAsset: Uri? = null
    private var bufferHandle: Long = 0
    private var splatCount: Int = 0
    private var vbo: Int = 0
    private var program: Int = 0
    private var aspect: Float = 1f

    fun queueAsset(uri: Uri) { pendingAsset = uri }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES31.glClearColor(0f, 0f, 0f, 1f)
        program = buildProgram()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES31.glViewport(0, 0, width, height)
        aspect = width.toFloat() / height.coerceAtLeast(1).toFloat()
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT or GLES31.GL_DEPTH_BUFFER_BIT)
        pendingAsset?.let { uri ->
            pendingAsset = null
            loadAsset(uri)
        }
        if (splatCount == 0 || program == 0) return
        GLES31.glUseProgram(program)
        GLES31.glEnable(GLES31.GL_BLEND)
        GLES31.glBlendFunc(GLES31.GL_SRC_ALPHA, GLES31.GL_ONE_MINUS_SRC_ALPHA)
        GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER, vbo)
        // Vertex attribs are laid out to match `SplatVertex` (56-byte stride).
        // Slot 0: position (vec3), slot 1: rotation (vec4),
        // slot 2: scale (vec3), slot 3: opacity (float),
        // slot 4: color (vec3).
        var off = 0
        GLES31.glEnableVertexAttribArray(0)
        GLES31.glVertexAttribPointer(0, 3, GLES31.GL_FLOAT, false, SplatforgeNative.vertexStride(), off); off += 12
        GLES31.glVertexAttribDivisor(0, 1)
        GLES31.glEnableVertexAttribArray(1)
        GLES31.glVertexAttribPointer(1, 4, GLES31.GL_FLOAT, false, SplatforgeNative.vertexStride(), off); off += 16
        GLES31.glVertexAttribDivisor(1, 1)
        GLES31.glEnableVertexAttribArray(2)
        GLES31.glVertexAttribPointer(2, 3, GLES31.GL_FLOAT, false, SplatforgeNative.vertexStride(), off); off += 12
        GLES31.glVertexAttribDivisor(2, 1)
        GLES31.glEnableVertexAttribArray(3)
        GLES31.glVertexAttribPointer(3, 1, GLES31.GL_FLOAT, false, SplatforgeNative.vertexStride(), off); off += 4
        GLES31.glVertexAttribDivisor(3, 1)
        GLES31.glEnableVertexAttribArray(4)
        GLES31.glVertexAttribPointer(4, 3, GLES31.GL_FLOAT, false, SplatforgeNative.vertexStride(), off)
        GLES31.glVertexAttribDivisor(4, 1)
        GLES31.glDrawArraysInstanced(GLES31.GL_TRIANGLE_STRIP, 0, 4, splatCount)
    }

    private fun loadAsset(uri: Uri) {
        val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return
        val newHandle = SplatforgeNative.decodeGlb(bytes)
        if (newHandle == 0L) return
        if (bufferHandle != 0L) SplatforgeNative.freeBuffer(bufferHandle)
        bufferHandle = newHandle
        splatCount = SplatforgeNative.bufferLen(newHandle)
        val byteCount = splatCount * SplatforgeNative.vertexStride()
        val staging = ByteBuffer.allocateDirect(byteCount).order(ByteOrder.nativeOrder())
        SplatforgeNative.copyVertices(newHandle, staging)
        staging.rewind()
        val buf = IntArray(1)
        if (vbo == 0) {
            GLES31.glGenBuffers(1, buf, 0)
            vbo = buf[0]
        }
        GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER, vbo)
        GLES31.glBufferData(GLES31.GL_ARRAY_BUFFER, byteCount, staging, GLES31.GL_STATIC_DRAW)
    }

    private fun buildProgram(): Int {
        val vs = readRaw("splat_point_vs")
        val fs = readRaw("splat_point_fs")
        val v = compile(GLES31.GL_VERTEX_SHADER, vs)
        val f = compile(GLES31.GL_FRAGMENT_SHADER, fs)
        val p = GLES31.glCreateProgram()
        GLES31.glAttachShader(p, v); GLES31.glAttachShader(p, f); GLES31.glLinkProgram(p)
        return p
    }

    private fun readRaw(name: String): String {
        val id = context.resources.getIdentifier(name, "raw", context.packageName)
        return context.resources.openRawResource(id).bufferedReader().use { it.readText() }
    }

    private fun compile(type: Int, src: String): Int {
        val s = GLES31.glCreateShader(type)
        GLES31.glShaderSource(s, src); GLES31.glCompileShader(s)
        return s
    }
}
