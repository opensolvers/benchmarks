# ScaLAPACK pdsyev benchmark - build against a ScaLAPACK module.
#
# Requires a ScaLAPACK module to be loaded first, e.g.:
#   module load ScaLAPACK/2.2.2-gompi-2025b-fb
# which sets EBROOT{SCALAPACK,FLEXIBLAS} and provides mpicc. Netlib ScaLAPACK
# 2.2 bundles the BLACS (Cblacs_* symbols) inside libscalapack.

CC      ?= mpicc
CFLAGS  ?= -O2 -Wall
LDLIBS  := -lscalapack -lflexiblas -lm
LDFLAGS := -L$(EBROOTSCALAPACK)/lib -L$(EBROOTFLEXIBLAS)/lib \
           -Wl,-rpath,$(EBROOTSCALAPACK)/lib -Wl,-rpath,$(EBROOTFLEXIBLAS)/lib

scalapack_bench: scalapack_bench.c
	@test -n "$(EBROOTSCALAPACK)" || { echo "ERROR: load a ScaLAPACK module first (e.g. 'module load ScaLAPACK/2.2.2-gompi-2025b-fb')"; exit 1; }
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f scalapack_bench

.PHONY: clean
