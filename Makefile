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
MAIN_SRC = $(SRC)/main.cpp
CPU_SRCS = $(SRC)/simulate_cpu.cpp $(SRC)/calculate.cpp $(SRC)/output.cpp
GPU_SRCS = $(SRC)/simulate_gpu.cu $(SRC)/calculate_gpu.cu

# object files
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)
CPU_OBJS = $(CPU_SRCS:.cpp=.o)
GPU_OBJS = $(GPU_SRCS:.cu=.o)
OBJS = $(MAIN_OBJ) $(CPU_OBJS) $(GPU_OBJS) 

# default target
.PHONY: all clean

all: build

build: $(EXEC)

$(EXEC): $(OBJS)
	mkdir -p $(BIN)
	$(NVCC) $(CXXFLAGS) $^ -o $@

# complie rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# clean target
clean:
	rm -rf $(BIN)/*.o $(EXEC)
