#include "simulate_gpu.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>

__global__ void monteCarloKernel(float* d_returns, int numPaths, float mean, float stdDev, curandState* states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPaths) return;

    curandState localState = states[tid];
    d_returns[tid] = mean + stdDev * curand_normal(&localState);
    states[tid] = localState; 
}

__global__ void initializeCurandStates(curandState* states, unsigned long seed, int numPaths) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPaths) return;

    curand_init(seed, tid, 0, &states[tid]);
}

std::vector<float> simulateReturnsGPU(int numPaths, float mean, float stdDev) {
    float* d_returns;
    cudaMalloc(&d_returns, numPaths * sizeof(float));

    curandState* d_states;
    cudaMalloc(&d_states, numPaths * sizeof(curandState));

    int blockSize = 256;
    int gridSize = (numPaths + blockSize - 1) / blockSize;

    initializeCurandStates<<<gridSize, blockSize>>>(d_states, clock(), numPaths);

    monteCarloKernel<<<gridSize, blockSize>>>(d_returns, numPaths, mean, stdDev, d_states);

    std::vector<float> h_returns(numPaths);
    cudaMemcpy(h_returns.data(), d_returns, numPaths * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_states);
    cudaFree(d_returns);

    return h_returns;
}
