#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib> // For std::atoi
#include <cuda_runtime.h> // Include CUDA runtime
#include <curand_kernel.h> // Include cuRAND kernel
#include "simulate_cpu.h"
#include "simulate_gpu.h"
#include "calculate.h"
#include "output.h"

int main(int argc, char* argv[]) {
    // Check for command-line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <numPaths>" << std::endl;
        return 1;
    }

    // Parse the number of paths from the command line
    int numPaths = std::atoi(argv[1]);
    if (numPaths <= 0) {
        std::cerr << "Error: numPaths must be a positive integer." << std::endl;
        return 1;
    }

    float meanReturn = 0.0005;
    float stdDev = 0.02;
    float confidenceLevel = 0.95;

    std::cout << "Running Monte Carlo Simulation VaR with numPaths = " << numPaths << "...\n";

    // Allocate GPU memory once and reuse
    float* d_returns;
    curandState* d_states;
    cudaMalloc(&d_returns, numPaths * sizeof(float));
    cudaMalloc(&d_states, numPaths * sizeof(curandState));

    // CPU simulation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> cpuReturns = simulateReturnsCPU(numPaths, meanReturn, stdDev);
    float cpuVaR = calculateVaR(cpuReturns, confidenceLevel);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;

    // GPU simulation
    auto gpu_start = std::chrono::high_resolution_clock::now();
    simulateReturnsGPU(numPaths, meanReturn, stdDev, d_returns, d_states);
    float gpuVaR = calculateVaRGPU(d_returns, numPaths, confidenceLevel);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_elapsed = gpu_end - gpu_start;

    // Output results
    outputResults(cpuVaR, gpuVaR, cpu_elapsed.count(), gpu_elapsed.count());

    // Free GPU memory
    cudaFree(d_states);
    cudaFree(d_returns);

    return 0;
}
