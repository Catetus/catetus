// QATPlyDecoder.kt — Kotlin facade over the native Vulkan decoder.
// Loads libsplatforge_qat.so on first use, then exposes idiomatic
// suspendable functions that take ByteArray/FloatArray inputs.
//
// SPDX-License-Identifier: MIT

package dev.splatforge.qat

/**
 * GPU-accelerated decoder for the QAT-PLY v1 wire format.
 *
 * Both functions block until the Vulkan dispatch completes. Production
 * renderers should call them off the main thread (e.g. via Dispatchers.IO
 * or a dedicated decoder coroutine).
 */
object QATPlyDecoder {

    init {
        System.loadLibrary("splatforge_qat")
    }

    /**
     * Dequantize an int8 + per-channel scale block.
     *
     * @param q         row-major [nRows * nChannels] signed bytes
     * @param scale     per-channel fp32 scales, length nChannels
     * @param nRows     anchor count
     * @param nChannels per-row channel count
     * @return          row-major fp32 of length nRows * nChannels
     */
    fun decodeInt8(q: ByteArray, scale: FloatArray, nRows: Int, nChannels: Int): FloatArray {
        require(q.size >= nRows * nChannels) { "q too small" }
        require(scale.size >= nChannels)     { "scale too small" }
        val out = FloatArray(nRows * nChannels)
        nativeDecodeInt8(q, scale, nRows, nChannels, out)
        return out
    }

    /**
     * Dequantize an int4-packed (two nibbles per byte) + per-anchor scale
     * block.
     *
     * @param packed    [nRows * ceil(nChannels/2)] bytes
     * @param scale     per-anchor fp32 scales, length nRows
     * @param nRows     anchor count
     * @param nChannels logical channel count
     * @return          row-major fp32 of length nRows * nChannels
     */
    fun decodeInt4Packed(packed: ByteArray, scale: FloatArray, nRows: Int, nChannels: Int): FloatArray {
        val B = (nChannels + 1) ushr 1
        require(packed.size >= nRows * B) { "packed too small" }
        require(scale.size >= nRows)      { "scale too small" }
        val out = FloatArray(nRows * nChannels)
        nativeDecodeInt4Packed(packed, scale, nRows, nChannels, out)
        return out
    }

    @JvmStatic
    private external fun nativeDecodeInt8(
        q: ByteArray, scale: FloatArray, nRows: Int, nChannels: Int, out: FloatArray
    )

    @JvmStatic
    private external fun nativeDecodeInt4Packed(
        packed: ByteArray, scale: FloatArray, nRows: Int, nChannels: Int, out: FloatArray
    )
}
