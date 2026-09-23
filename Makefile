################################################################################
# Makefile - Batch CUDA Image Edge Detection Pipeline
# GPU Specialization Capstone Project
################################################################################

CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

SRC_DIR := src
INCLUDES := -I$(SRC_DIR)

# Gencode: broad coverage of common Coursera-lab GPU architectures
SMS ?= 50 52 60 61 70 75 80 86
GENCODE_FLAGS :=
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM := $(lastword $(sort $(SMS)))
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

NVCCFLAGS := --std=c++14 -O3
LDFLAGS :=

TARGET := edge_detect.exe

all: build

build: $(TARGET)

$(TARGET): $(SRC_DIR)/edge_detect.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $< $(LDFLAGS)

run: build
	./$(TARGET) images_in images_out --threshold 100

clean:
	rm -f $(TARGET) *.o

clobber: clean
	rm -rf images_out

.PHONY: all build run clean clobber
