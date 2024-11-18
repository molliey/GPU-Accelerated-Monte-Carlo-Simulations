#ifndef SIMULATE_GPU_H
#define SIMULATE_GPU_H

#include <curand_kernel.h>

void simulateReturnsGPU(int numPaths, float meanReturn, float stdDev, float* d_returns, curandState* d_states);
float calculateVaRGPU(float* d_returns, int numPaths, float confidenceLevel);

#endif
