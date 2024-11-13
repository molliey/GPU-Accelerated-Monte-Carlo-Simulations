# compile
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O2 -Iinclude

# file paths
SRC = src
INCLUDE = include
BIN = bin

# output executable name
EXEC = $(BIN)/monte_carlo_sim

# source files
CPU_SRCS = $(SRC)/simulate_cpu.cpp $(SRC)/calculate.cpp $(SRC)/output.cpp
GPU_SRCS = $(SRC)/simulate_gpu.cu
MAIN_SRC = $(SRC)/main.cpp

all: build

build:
	mkdir -p $(BIN)
	$(NVCC) $(CXXFLAGS) $(CPU_SRCS) $(GPU_SRCS) $(MAIN_SRC) -o $(EXEC)

clean:
	rm -rf $(BIN)/*
