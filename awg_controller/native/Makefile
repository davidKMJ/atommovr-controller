# Builds libawg_engine.so -- the precompute-a-whole-round-then-stream
# SCAPP engine used by awg_controller.awg_engine. Mirrors the
# vendored example Makefiles under scapp/cuda_rdma/*/Makefile (same
# NVCC/GENCODE/LIBRARIES setup), trimmed to a Linux shared library instead
# of a standalone executable.
#
# The vendored Spectrum SCAPP SDK lives at the repo root (`scapp/`),
# alongside PPSU2023 -- see SCAPP_DIR below.
#
# Override on the command line if needed, e.g.:
#   make CUDA_PATH=/usr/local/cuda-12.4 SMS="75 86"
#   make SCAPP_DIR=/path/to/scapp

CUDA_PATH  ?= /usr/local/cuda
NVCC       := $(CUDA_PATH)/bin/nvcc

# GPU architectures to build for (space-separated compute capability
# numbers, e.g. "75 86"). Default 75 = Turing, matching the lab machine's
# T1000; override with the target machine's actual compute capability
# (`nvidia-smi --query-gpu=compute_cap --format=csv` reports it, e.g. 8.6
# -> SMS=86).
#
# Getting this wrong is not always loud: each listed arch gets real SASS,
# and the *highest* one additionally gets PTX, which the driver can JIT
# **forward** onto a newer GPU but never backward onto an older one. So a
# too-low SMS still runs (via JIT, with a load-time stall) while a too-high
# SMS links fine and then fails at kernel launch. Note awg_engine.py's
# _lib() auto-runs `make` with no arguments when libawg_engine.so is
# missing, so this default is what an unattended build gets.
SMS ?= 75

GENCODE_FLAGS := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))
HIGHEST_SM := $(lastword $(sort $(SMS)))
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

SCAPP_DIR ?= ../../scapp

INCLUDES  := -I$(SCAPP_DIR)/c_header -I$(SCAPP_DIR)/cuda_rdma/common
LIBRARIES := -lspcm_linux -lcuda -lcudart

CXXFLAGS := -std=c++14 -O2 -Xcompiler -fPIC -Xcompiler -pthread

TARGET := libawg_engine.so

all: $(TARGET)

# All of these are #included into awg_engine.cu, so a change to any one of
# them must force a rebuild. Listing only awg_engine.{cu,h} would leave make
# thinking the .o is up to date after e.g. a phase.h edit, and you would
# silently keep testing the previous build.
HEADERS := awg_engine.h phase.h schedule.h render.cuh stream.cuh sequence.cuh

awg_engine.o: awg_engine.cu $(HEADERS)
	$(NVCC) $(INCLUDES) $(CXXFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<

spcm_cuda_common.o: $(SCAPP_DIR)/cuda_rdma/common/spcm_cuda_common.cu
	$(NVCC) $(INCLUDES) $(CXXFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<

$(TARGET): awg_engine.o spcm_cuda_common.o
	$(NVCC) -shared $(CXXFLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

clean:
	rm -f *.o $(TARGET) tests/test_phase tests/test_schedule tests/bench_render tests/card_info

# Host-side numerics tests: plain C++, no CUDA and no card, so the phase and
# schedule maths can be checked on a laptop before touching hardware.
#   make test          -- fixed-point phase vs a double reference
#   make test-scapp    -- full round diffed against awg_controller.scapp
CXX ?= c++
HOSTFLAGS := -O2 -std=c++14 -Wall

tests/test_phase: tests/test_phase.cpp phase.h
	$(CXX) $(HOSTFLAGS) -o $@ $<

tests/test_schedule: tests/test_schedule.cpp schedule.h phase.h awg_engine.h
	$(CXX) $(HOSTFLAGS) -o $@ $<

# Render-throughput benchmark: needs a CUDA GPU but no card and no driver, so
# it can answer "will 15x15 stream in real time?" before any hardware run.
tests/bench_render: tests/bench_render.cu $(HEADERS)
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -o $@ $<

bench: tests/bench_render
	./tests/bench_render 500

# Link diagnostic: opens the card and reports the negotiated PCIe generation
# and lane count, hence the maximum FIFO sample rate the bus can sustain.
# Needs the card and the spcm driver, but touches no output and no GPU.
tests/card_info: tests/card_info.c
	$(CC) -O2 -Wall -o $@ $< -lspcm_linux

card-info: tests/card_info
	./tests/card_info

test: tests/test_phase
	./tests/test_phase

test-scapp: tests/test_schedule
	cd ../.. && python awg_controller/native/tests/compare_with_scapp.py \
		awg_controller/native/tests/test_schedule

.PHONY: all clean test test-scapp bench card-info
