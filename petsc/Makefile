# PETSc benchmarks - build against EESSI/EasyBuild PETSc + FlexiBLAS modules.
#
#   module load foss/2025b PETSc/3.24.0-foss-2025b

CC      := mpicc
CFLAGS  ?= -O2 -Wall

PETSC_INC := $(EBROOTPETSC)/include
PETSC_LIB := $(EBROOTPETSC)/lib
FLEX_LIB  := $(EBROOTFLEXIBLAS)/lib

RPATH := -Wl,-rpath,$(PETSC_LIB) -Wl,-rpath,$(FLEX_LIB)
LIBS  := -L$(PETSC_LIB) -L$(FLEX_LIB) $(RPATH) -lpetsc -lflexiblas -lm

BINS := petsc_ksp_bench petsc_dense_bench petsc_direct_bench petsc_spmv_rvv_bench

all: $(BINS)

petsc_ksp_bench: petsc_ksp_bench.c
	@test -n "$(EBROOTPETSC)" || { echo "ERROR: load PETSc first"; exit 1; }
	@command -v mpicc >/dev/null || { echo "ERROR: mpicc not in PATH"; exit 1; }
	$(CC) $(CFLAGS) -I$(PETSC_INC) $< -o $@ $(LIBS)

petsc_dense_bench: petsc_dense_bench.c
	@test -n "$(EBROOTPETSC)" || { echo "ERROR: load PETSc first"; exit 1; }
	$(CC) $(CFLAGS) -I$(PETSC_INC) $< -o $@ $(LIBS)

petsc_direct_bench: petsc_direct_bench.c
	@test -n "$(EBROOTPETSC)" || { echo "ERROR: load PETSc first"; exit 1; }
	$(CC) $(CFLAGS) -I$(PETSC_INC) $< -o $@ $(LIBS)

# RVV SpMV needs vector ISA in the march string.
RVV_CFLAGS ?= -O3 -Wall -march=rv64gcv -mabi=lp64d

petsc_spmv_rvv_bench: petsc_spmv_rvv_bench.c
	@test -n "$(EBROOTPETSC)" || { echo "ERROR: load PETSc first"; exit 1; }
	$(CC) $(RVV_CFLAGS) -I$(PETSC_INC) $< -o $@ $(LIBS)

clean:
	rm -f $(BINS)

.PHONY: all clean
