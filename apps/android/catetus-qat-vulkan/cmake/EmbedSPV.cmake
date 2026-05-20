# EmbedSPV.cmake — convert a SPIR-V binary into a C header with a
# uint32 array. Usage:
#   cmake -DSPV=<input.spv> -DHDR=<output.h> -P EmbedSPV.cmake

file(READ ${SPV} CONTENTS HEX)
string(LENGTH ${CONTENTS} CHARS)
math(EXPR BYTES "${CHARS} / 2")
math(EXPR WORDS "${BYTES} / 4")

set(OUT
"#pragma once\n#include <cstdint>\n#include <cstddef>\n\nstatic const uint32_t kQatDequantSPV[] = {\n")

set(i 0)
while(i LESS BYTES)
    math(EXPR i0 "${i} * 2")
    string(SUBSTRING ${CONTENTS} ${i0} 8 WORD_HEX)
    # SPIR-V is little-endian: bytes b0 b1 b2 b3 -> 0xb3b2b1b0
    string(SUBSTRING ${WORD_HEX} 0 2 B0)
    string(SUBSTRING ${WORD_HEX} 2 2 B1)
    string(SUBSTRING ${WORD_HEX} 4 2 B2)
    string(SUBSTRING ${WORD_HEX} 6 2 B3)
    string(APPEND OUT "0x${B3}${B2}${B1}${B0},")
    math(EXPR mod "(${i} / 4) % 8")
    if(mod EQUAL 7)
        string(APPEND OUT "\n")
    endif()
    math(EXPR i "${i} + 4")
endwhile()

string(APPEND OUT "\n};\nstatic const size_t kQatDequantSPVSize = sizeof(kQatDequantSPV);\n")
file(WRITE ${HDR} "${OUT}")
