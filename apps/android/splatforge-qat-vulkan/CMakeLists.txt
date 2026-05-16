# CMakeLists.txt — native build for SplatForge QAT-PLY v1 Vulkan
# decoder. Compiles the GLSL compute shader to SPIR-V, embeds it as a
# C header, then builds libsplatforge_qat.so with the JNI bridge.
#
# Toolchain expectations (Android Studio sets these automatically):
#   - cmake >= 3.22
#   - glslc OR glslangValidator on PATH (NDK 26+ ships glslc; Homebrew
#     glslang ships glslangValidator)
#   - Vulkan loader libvulkan.so from the NDK
#
# Standalone configure (no Android SDK / NDK): `cmake -B build`.
#
# License: MIT.

cmake_minimum_required(VERSION 3.22)
project(splatforge_qat LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------
# 1. Compile qat_dequant.comp -> qat_dequant.spv -> qat_dequant_spv.h.
# -----------------------------------------------------------------------
find_program(GLSLC glslc HINTS
    "$ENV{VULKAN_SDK}/bin"
    "${ANDROID_NDK}/shader-tools/darwin-x86_64"
    "${ANDROID_NDK}/shader-tools/linux-x86_64"
)
if(NOT GLSLC)
    # Fallback to glslangValidator / glslang (Homebrew, Vulkan SDK).
    find_program(GLSLANG NAMES glslangValidator glslang HINTS
        "$ENV{VULKAN_SDK}/bin"
    )
endif()
if(NOT GLSLC AND NOT GLSLANG)
    message(WARNING "glslc/glslangValidator not found. Android build will fail; "
                    "configure still succeeds for syntactic validation.")
endif()

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/shaders/qat_dequant.comp)
set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/qat_dequant.spv)
set(SHADER_HDR ${CMAKE_CURRENT_BINARY_DIR}/qat_dequant_spv.h)

if(GLSLC)
    add_custom_command(
        OUTPUT  ${SHADER_SPV}
        COMMAND ${GLSLC} -O ${SHADER_SRC} -o ${SHADER_SPV}
        DEPENDS ${SHADER_SRC}
        COMMENT "glslc qat_dequant.comp"
    )
    add_custom_command(
        OUTPUT  ${SHADER_HDR}
        COMMAND ${CMAKE_COMMAND} -DSPV=${SHADER_SPV} -DHDR=${SHADER_HDR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSPV.cmake
        DEPENDS ${SHADER_SPV}
        COMMENT "embed qat_dequant.spv as C array"
    )
elseif(GLSLANG)
    add_custom_command(
        OUTPUT  ${SHADER_SPV}
        COMMAND ${GLSLANG} -V ${SHADER_SRC} -o ${SHADER_SPV}
        DEPENDS ${SHADER_SRC}
        COMMENT "glslang -V qat_dequant.comp"
    )
    add_custom_command(
        OUTPUT  ${SHADER_HDR}
        COMMAND ${CMAKE_COMMAND} -DSPV=${SHADER_SPV} -DHDR=${SHADER_HDR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSPV.cmake
        DEPENDS ${SHADER_SPV}
        COMMENT "embed qat_dequant.spv as C array"
    )
else()
    # Placeholder header so configure works even when no compiler is found.
    file(WRITE ${SHADER_HDR}
        "#pragma once\n"
        "#include <cstdint>\n"
        "#include <cstddef>\n"
        "static const uint32_t kQatDequantSPV[1] = {0};\n"
        "static const size_t kQatDequantSPVSize = 0;\n")
endif()

# -----------------------------------------------------------------------
# 2. Build libsplatforge_qat.so. Requires <vulkan/vulkan.h>, which the
#    Android NDK supplies automatically. On a host machine without the
#    NDK, the configure step (alone) is the syntactic check.
# -----------------------------------------------------------------------
add_library(splatforge_qat SHARED
    src/main/cpp/qat_decoder.cpp
    src/main/cpp/jni_bridge.cpp
    ${SHADER_HDR}
)

target_include_directories(splatforge_qat PRIVATE
    src/main/cpp
    ${CMAKE_CURRENT_BINARY_DIR}
)

find_library(VULKAN_LIB vulkan)
if(VULKAN_LIB)
    target_link_libraries(splatforge_qat ${VULKAN_LIB})
else()
    message(WARNING "libvulkan not found at link time. The Android NDK "
                    "supplies it automatically for android-* triples.")
endif()
