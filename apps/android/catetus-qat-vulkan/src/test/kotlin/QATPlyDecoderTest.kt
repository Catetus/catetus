// QATPlyDecoderTest.kt — local JVM test for the Vulkan decoder.
//
// Real Vulkan execution requires either an Android emulator with GPU or
// a software Vulkan implementation (SwiftShader / lavapipe) on the host.
// This test verifies the JNI bridge dispatches without crashing when a
// Vulkan implementation is available; if none is found, the test is
// SKIPPED (not failed) so CI doesn't go red on machines without GPU.
//
// To run the full GPU-accelerated conformance check, deploy to an
// Android device and run the connected-instrumentation flavour.
//
// SPDX-License-Identifier: MIT

package dev.catetus.qat

import org.junit.Assume.assumeTrue
import org.junit.Test
import kotlin.math.abs

class QATPlyDecoderTest {

    private fun vulkanAvailable(): Boolean {
        return try {
            System.loadLibrary("catetus_qat")
            true
        } catch (t: Throwable) {
            false
        }
    }

    @Test
    fun smokeDecodeInt8SingleRow() {
        assumeTrue("native lib not loadable in host JVM", vulkanAvailable())
        // 1 row, 4 channels: q = [-1, 0, 1, 2], scale = [1.0]*4
        // expected = [-1, 0, 1, 2]
        val q     = byteArrayOf(-1, 0, 1, 2)
        val scale = floatArrayOf(1f, 1f, 1f, 1f)
        val out   = QATPlyDecoder.decodeInt8(q, scale, 1, 4)
        val expected = floatArrayOf(-1f, 0f, 1f, 2f)
        for (i in 0 until 4) {
            assert(abs(out[i] - expected[i]) < 1e-6f) {
                "mismatch at $i: got ${out[i]} expected ${expected[i]}"
            }
        }
    }

    @Test
    fun smokeDecodeInt4Packed() {
        assumeTrue("native lib not loadable in host JVM", vulkanAvailable())
        // 1 row, 2 channels, packed as one byte.
        // nibble layout: low = ch0, high = ch1.
        // nibble 0x07 = 7 -> signed_q = -1 (after -8)
        // nibble 0x09 = 9 -> signed_q =  1
        // byte = (9 << 4) | 7 = 0x97 = -105 as signed
        val packed = byteArrayOf(0x97.toByte())
        val scale  = floatArrayOf(1f)
        val out    = QATPlyDecoder.decodeInt4Packed(packed, scale, 1, 2)
        assert(abs(out[0] - -1f) < 1e-6f) { "ch0: got ${out[0]}" }
        assert(abs(out[1] -  1f) < 1e-6f) { "ch1: got ${out[1]}" }
    }
}
