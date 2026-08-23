# ime-bench - int8 (s8s8s32) GEMM: SpaceMiT X60 IME (smt.vmadot) vs RVV vs scalar.
#
# The IME kernel emits smt.vmadot as a raw instruction word (.insn), so it needs
# no xsmtvdot toolchain - any RVV-capable binutils builds it. RVV intrinsics need
# 'v' in -march. qemu-user cannot execute vmadot: run the board targets on X60.
CC          ?= cc
CFLAGS      ?= -O2 -Wall -Wextra
BOARD_MARCH ?= rv64gcv_zvl256b_zfh_zvfh

# Portable self-test (any host, x86 included): scalar ref + packed ref only.
# Verifies the shared packing / tile-index math and the harness with no RISC-V.
host: bench.c gemm_ref.c
	$(CC) $(CFLAGS) bench.c gemm_ref.c -o ime-bench-host

# Full X60 build: scalar + RVV baseline + IME microkernel (IME via raw .insn).
board: bench.c gemm_ref.c gemm_rvv.c gemm_ime.c
	$(CC) $(CFLAGS) -march=$(BOARD_MARCH) bench.c gemm_ref.c gemm_rvv.c gemm_ime.c -o ime-bench

# RVV-only X60 build (skips the IME microkernel entirely).
board-rvv: bench.c gemm_ref.c gemm_rvv.c
	$(CC) $(CFLAGS) -march=$(BOARD_MARCH) -DGEMM_NO_IME bench.c gemm_ref.c gemm_rvv.c -o ime-bench

# Isolated vmadot peak (cpufp-style).
board-peak: bench_peak.c gemm_ref.c gemm_ime.c
	$(CC) $(CFLAGS) -march=$(BOARD_MARCH) bench_peak.c gemm_ref.c gemm_ime.c -o ime-bench-peak

# Multi-core synthetic IME throughput (OpenMP M-split).
board-mc: bench_mc.c gemm_ref.c gemm_ime.c
	$(CC) $(CFLAGS) -fopenmp -march=$(BOARD_MARCH) bench_mc.c gemm_ref.c gemm_ime.c -o ime-bench-mc

clean:
	rm -f ime-bench ime-bench-host ime-bench-mc ime-bench-peak

.PHONY: host board board-rvv board-peak board-mc clean
