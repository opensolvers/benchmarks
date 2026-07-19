# qnbit-mlas-bench - isolated microbenchmark for ORT's MLAS SQNBit int4 CompInt8
# GEMM kernel on the SpaceMiT X60. Links straight out of the ORT build tree's
# libonnxruntime_mlas.a, so it needs a built ONNX Runtime.
#
# Point ORT_BUILD at the easybuild_obj dir (holds the .a archives) and ORT_SRC
# at the checked-out onnxruntime source (holds core/mlas/inc headers). Defaults
# match this repo's X60 EasyBuild tree; override on the make command line.
#
# The kernel emits SpaceMiT smt.vmadot as raw instruction words, so any RVV
# toolchain builds it; qemu-user cannot execute vmadot -> run on the X60.
#
# TOOLCHAIN NOTE: the ORT archives were built with GCCcore/14.3.0. The EESSI
# compat-layer default c++ is GCC 13 and produces a binary needing CXXABI_1.3.15
# that is absent at runtime. Build AND run with GCC 14:
#   GCC14=/cvmfs/.../software/GCCcore/14.3.0
#   make board CXX=$GCC14/bin/g++
#   LD_LIBRARY_PATH=$GCC14/lib64:$LD_LIBRARY_PATH ./qnbit-mlas-bench 1 4096 11008 50
CXX         ?= c++
CXXFLAGS    ?= -O2 -std=c++17 -Wall -Wextra
BOARD_MARCH ?= rv64gcv_zvl256b_zfh_zvfh

ORT_BUILD ?= /home/orangepi/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot/easybuild_obj
ORT_SRC   ?= $(ORT_BUILD)/../onnxruntime-361184e61957410f19153754f325806972546d5b/onnxruntime

MLAS_INC  := $(ORT_SRC)/core/mlas/inc
INCLUDES  := -I$(MLAS_INC) -I$(ORT_SRC)

# The mlas archive references ThreadPool/CPUIDInfo (thread-count query + platform
# probe) even on the nullptr-pool path; those live in libonnxruntime_common.a,
# which in turn needs abseil (bundled in the ORT build tree). --start/--end-group
# resolves the cross-archive back-references; -pthread covers the platform init.
ABSEIL_LIBS ?= $(shell find $(ORT_BUILD)/_deps/abseil_cpp-build -name 'libabsl_*.a')
LIBS := -Wl,--start-group \
        $(ORT_BUILD)/libonnxruntime_mlas.a \
        $(ORT_BUILD)/libonnxruntime_common.a \
        $(ABSEIL_LIBS) \
        -Wl,--end-group -pthread

board: bench_qnbit_mlas.cpp
	$(CXX) $(CXXFLAGS) -march=$(BOARD_MARCH) $(INCLUDES) bench_qnbit_mlas.cpp $(LIBS) -o qnbit-mlas-bench

clean:
	rm -f qnbit-mlas-bench

.PHONY: board clean
