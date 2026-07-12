# ELPA eigensolver benchmark - build against an EESSI/EasyBuild ELPA module.
#
# Requires an ELPA module to be loaded first, e.g.:
#   module load ELPA/2025.06.002-foss-2025b
# which sets EBROOT{ELPA,SCALAPACK,FLEXIBLAS}. The OpenMP ELPA build ships
# headers under include/elpa_openmp-<ver>/ and libelpa_openmp.so; a non-OpenMP
# build ships include/elpa-<ver>/ and libelpa.so. This Makefile detects both.

CC      ?= mpicc
CFLAGS  ?= -O2 -fopenmp -Wall

# Prefer the OpenMP ELPA build if present, else the plain build.
ELPA_OMP := $(wildcard $(EBROOTELPA)/include/elpa_openmp-*/)
INCDIR   := $(firstword $(ELPA_OMP) $(wildcard $(EBROOTELPA)/include/elpa-*/))
ELPALIB  := $(if $(ELPA_OMP),-lelpa_openmp,-lelpa)

LDLIBS  := $(ELPALIB) -lscalapack -lflexiblas -lm
LDFLAGS := -L$(EBROOTELPA)/lib -L$(EBROOTSCALAPACK)/lib -L$(EBROOTFLEXIBLAS)/lib \
           -Wl,-rpath,$(EBROOTELPA)/lib -Wl,-rpath,$(EBROOTSCALAPACK)/lib -Wl,-rpath,$(EBROOTFLEXIBLAS)/lib

elpa_bench: elpa_bench.c
	@test -n "$(EBROOTELPA)" || { echo "ERROR: load an ELPA module first (e.g. 'module load ELPA/2025.06.002-foss-2025b')"; exit 1; }
	@test -n "$(INCDIR)"     || { echo "ERROR: could not find ELPA headers under $(EBROOTELPA)/include"; exit 1; }
	$(CC) $(CFLAGS) -I$(INCDIR) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f elpa_bench

.PHONY: clean
