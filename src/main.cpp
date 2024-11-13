#include <iostream>
#include <chrono>
#include <vector>
#include "simulate_cpu.h"
#include "simulate_gpu.h"
#include "calculate.h"
#include "output.h"

int main() {
    int numPaths = 100000;
    float meanReturn = 0.0005;
    float stdDev = 0.02;
    float confidenceLevel = 0.95;

    std::cout << "Running Monte Carlo Simulation VaR...\n";

    // CPU simulation 
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> cpuReturns = simulateReturnsCPU(numPaths, meanReturn, stdDev); 
    float cpuVaR = calculateVaR(cpuReturns, confidenceLevel);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;

    // GPU simulation
    auto gpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> gpuReturns = simulateReturnsGPU(numPaths, meanReturn, stdDev);
    float gpuVaR = calculateVaR(gpuReturns, confidenceLevel);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_elapsed = gpu_end - gpu_start;

    outputResults(cpuVaR, gpuVaR, cpu_elapsed.count(), gpu_elapsed.count());

    return 0;
}