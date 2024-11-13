#include <iostream>
#include "simulate_cpu.h"
#include "simulate_gpu.h"
#include "calculate.h"
#include "output.h"

int main() {
    const int NUM_PATHS = 100000;
    const float MEAN_RETURN = 0.0005;
    const float STD_DEV = 0.02;
    const float CONFIDENCE_LEVEL = 0.95;

    std::cout << "Running Monte Carlo VaR Simulation...\n";

    // CPU simulation 
    std::vector<float> cpuReturns = simulateReturnsCPU(MEAN_RETURN, STD_DEV, NUM_PATHS);
    float cpuVaR = calculateVaR(cpuReturns, CONFIDENCE_LEVEL);

    // GPU simulation
    std::vector<float> gpuReturns = simulateReturnsGPU(MEAN_RETURN, STD_DEV, NUM_PATHS);
    float gpuVaR = calculateVaR(gpuReturns, CONFIDENCE_LEVEL);

    outputResults(cpuVaR, gpuVaR, "output.csv");

    return 0;
}