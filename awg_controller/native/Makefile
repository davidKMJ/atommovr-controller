CUDA_PATH  ?= /usr/local/cuda
NVCC       := /usr/bin/nvcc

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

HEADERS := awg_engine.h phase.h schedule.h render.cuh stream.cuh sequence.cuh

awg_engine.o: awg_engine.cu $(HEADERS)
	$(NVCC) $(INCLUDES) $(CXXFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<

spcm_cuda_common.o: $(SCAPP_DIR)/cuda_rdma/common/spcm_cuda_common.cu
	$(NVCC) $(INCLUDES) $(CXXFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<

$(TARGET): awg_engine.o spcm_cuda_common.o
	$(NVCC) -shared $(CXXFLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

clean:
	rm -f *.o $(TARGET) tests/test_phase tests/test_schedule

CXX ?= c++
HOSTFLAGS := -O2 -std=c++14 -Wall

tests/test_phase: tests/test_phase.cpp phase.h
	$(CXX) $(HOSTFLAGS) -o $@ $<

tests/test_schedule: tests/test_schedule.cpp schedule.h phase.h awg_engine.h
	$(CXX) $(HOSTFLAGS) -o $@ $<

test: tests/test_phase tests/test_schedule
	./tests/test_phase
	./tests/test_schedule

.PHONY: all clean test
