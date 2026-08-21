# Build against loaded ScaFaCoS/foss-2025b (uses cmake SCAFACOS_LIBRARIES).
CC = mpicc
CFLAGS ?= -O2 -Wall

scafacos_bench: scafacos_bench.c armci_stubs.c
	@test -n "$(EBROOTSCAFACOS)" || { echo "load ScaFaCoS/1.0.4-foss-2025b first"; exit 1; }
	$(CC) $(CFLAGS) -I$(EBROOTSCAFACOS)/include $^ -o $@ \
	  -Wl,-rpath,$(EBROOTSCAFACOS)/lib \
	  -Wl,-rpath,$(EBROOTFFTW)/lib \
	  -Wl,-rpath,$(EBROOTGSL)/lib \
	  $$(sed -n 's/set(SCAFACOS_LIBRARIES "\(.*\)")/\1/p' \
	       $(EBROOTSCAFACOS)/share/cmake/scafacos/scafacosConfig.cmake \
	     | sed 's|\$${SCAFACOS_ROOT_DIR}|$(EBROOTSCAFACOS)|g') \
	  -lgsl -lgslcblas -lgmp

clean:
	rm -f scafacos_bench

.PHONY: clean
