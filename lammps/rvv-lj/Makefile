# RVV LJ/cut pair microkernel — build on SpaceMiT X60 / RV2
#
# EESSI RISC-V (required for published numbers):
#   export EESSI_VERSION_OVERRIDE=2025.06-001
#   source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
#   module load GCC/14.3.0
#   make CC=gcc
# Docs: https://www.eessi.io/docs/repositories/dev.eessi.io-riscv/
# Compat init = software.eessi.io; apps/toolchain = dev.eessi.io/riscv (riscv64/generic).
CC ?= gcc
CFLAGS ?= -O3 -ffast-math -fno-math-errno
ARCH ?= -march=rv64gcv_zvl256b
LDFLAGS ?=
BIN = lj_pair_bench

.PHONY: all clean

all: $(BIN)

$(BIN): bench.c lj_pair_scalar.c lj_pair_rvv.c lj_pair.h
	$(CC) $(CFLAGS) $(ARCH) -o $@ bench.c lj_pair_scalar.c lj_pair_rvv.c $(LDFLAGS) -lm
	@echo "built with: $$($(CC) --version | head -1)"

clean:
	rm -f $(BIN)
