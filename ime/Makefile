# ime-bench - int8 (s8s8s32) GEMM: SpaceMiT X60 IME (smt.vmadot) vs RVV vs scalar.
#
# The IME mnemonic is enabled by `.option arch, +xsmtvdot` inside the asm, so no
# -march=...xsmtvdot is needed - but the assembler must know xsmtvdot (binutils
# >=2.46, or a SpaceMiT/Bianbu cross toolchain that ships xsmtvdot). The
# board-insn target sidesteps this entirely via a raw .insn encoding.
# RVV intrinsics need 'v' in -march. qemu-user cannot execute vmadot: run on X60.
CC          ?= cc
CFLAGS      ?= -O2 -Wall -Wextra
BOARD_MARCH ?= rv64gcv_zvl256b_zfh_zvfh

# Portable self-test (any host, x86 included): scalar ref + packed ref only.
# Verifies the shared packing / tile-index math and the harness with no RISC-V.
host: bench.c gemm_ref.c
	$(CC) $(CFLAGS) bench.c gemm_ref.c -o ime-bench-host

# Full X60 build: scalar + RVV baseline + IME microkernel.
board: bench.c gemm_ref.c gemm_rvv.c gemm_ime.c
	$(CC) $(CFLAGS) -march=$(BOARD_MARCH) bench.c gemm_ref.c gemm_rvv.c gemm_ime.c -o ime-bench

# RVV-only X60 build for a board whose native binutils < 2.46 (no xsmtvdot).
board-rvv: bench.c gemm_ref.c gemm_rvv.c
	$(CC) $(CFLAGS) -march=$(BOARD_MARCH) -DGEMM_NO_IME bench.c gemm_ref.c gemm_rvv.c -o ime-bench

# Full X60 build with IME emitted as a raw instruction word (.insn 0xe294382b)
# instead of the smt.vmadot mnemonic, so a board whose native binutils < 2.46
# (no xsmtvdot) still gets the IME path. No cross toolchain needed.
board-insn: bench.c gemm_ref.c gemm_rvv.c gemm_ime.c
	$(CC) $(CFLAGS) -march=$(BOARD_MARCH) -DGEMM_IME_INSN bench.c gemm_ref.c gemm_rvv.c gemm_ime.c -o ime-bench

clean:
	rm -f ime-bench ime-bench-host

.PHONY: host board board-rvv board-insn clean
