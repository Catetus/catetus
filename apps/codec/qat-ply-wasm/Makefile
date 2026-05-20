# Makefile — build the QAT-PLY v1 reference C decoder as a WebAssembly
# module via Emscripten. The resulting `qat_ply_decode.{js,wasm}` pair is
# the browser fallback for non-WebGPU targets and matches the C decoder
# byte-for-byte.
#
# Usage:
#   make build      # produces dist/qat_ply_decode.js + dist/qat_ply_decode.wasm
#   make size       # uncompressed + gzipped size table
#   make clean
#
# Requires: emcc (Emscripten). Install: `brew install emscripten` or see
# https://emscripten.org/docs/getting_started/downloads.html
#
# License: MIT.

EMCC ?= emcc

C_SRC   := ../qat-ply-c/qat_ply_decode.c
CXX_SRC := src/wrapper.cc
OUT_DIR := dist
OUT_JS  := $(OUT_DIR)/qat_ply_decode.js
OUT_WASM:= $(OUT_DIR)/qat_ply_decode.wasm

EM_FLAGS := \
  -O3 \
  -std=c++17 \
  -I../qat-ply-c \
  --bind \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s EXPORT_NAME=createQatPlyModule \
  -s ENVIRONMENT=web,worker,node \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s STACK_SIZE=1048576 \
  -s FILESYSTEM=0 \
  -s SINGLE_FILE=0

.PHONY: build size clean

build: $(OUT_JS)

$(OUT_JS): $(C_SRC) $(CXX_SRC)
	@mkdir -p $(OUT_DIR)
	$(EMCC) $(EM_FLAGS) $(C_SRC) $(CXX_SRC) -o $(OUT_JS)

size: build
	@echo "Component        Uncompressed    Gzipped"
	@printf "%-15s  %-14s  %-8s\n" "qat_ply_decode.wasm" \
	  "$$(wc -c < $(OUT_WASM) | tr -d ' ') B" \
	  "$$(gzip -c -9 $(OUT_WASM) | wc -c | tr -d ' ') B"
	@printf "%-15s  %-14s  %-8s\n" "qat_ply_decode.js" \
	  "$$(wc -c < $(OUT_JS) | tr -d ' ') B" \
	  "$$(gzip -c -9 $(OUT_JS) | wc -c | tr -d ' ') B"

clean:
	rm -rf $(OUT_DIR)
