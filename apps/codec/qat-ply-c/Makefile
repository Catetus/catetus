# Reference C99 decoder for SplatForge QAT-PLY v1.
# License: MIT (see LICENSE).

CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -Werror -O2 -pedantic

SRC := qat_ply_decode.c
HDR := qat_ply_decode.h
TEST_SRC := tests/test_qat_ply_decode.c
TEST_BIN := tests/test_qat_ply_decode

.PHONY: all test clean

all: $(TEST_BIN)

$(TEST_BIN): $(SRC) $(HDR) $(TEST_SRC)
	$(CC) $(CFLAGS) -I. $(SRC) $(TEST_SRC) -o $(TEST_BIN)

test: $(TEST_BIN)
	@$(TEST_BIN)

clean:
	rm -f $(TEST_BIN)
