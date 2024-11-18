#include "simulate_gpu.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <iostream>

__global__ void monteCarloCombinedKernel(float* d_returns, int numPaths, float mean, float stdDev, curandState* states, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPaths) return;

    // Initialize cuRAND state and generate a random sample
    curand_init(seed, tid, 0, &states[tid]);
    d_returns[tid] = mean + stdDev * curand_normal(&states[tid]);
}

// Function to calculate VaR directly on the GPU using Thrust
float calculateVaRGPU(float* d_returns, int numPaths, float confidenceLevel) {
    // Wrap raw pointer in Thrust device vector
    thrust::device_vector<float> d_vec(d_returns, d_returns + numPaths);

    // Sort the returns on the GPU
    thrust::sort(d_vec.begin(), d_vec.end());

    // Calculate the VaR (5th percentile for 95% confidence level)
    int index = static_cast<int>((1.0 - confidenceLevel) * numPaths);
    return -d_vec[index];
}

// Updated simulateReturnsGPU function with float return type
float simulateReturnsGPU(int numPaths, float mean, float stdDev) {
    float* d_returns;
    cudaMalloc(&d_returns, numPaths * sizeof(float));

    curandState* d_states;
    cudaMalloc(&d_states, numPaths * sizeof(curandState));

    int blockSize = 512;
    int gridSize = (numPaths + blockSize - 1) / blockSize;

    // Launch the combined kernel
    unsigned long seed = clock();
    monteCarloCombinedKernel<<<gridSize, blockSize>>>(d_returns, numPaths, mean, stdDev, d_states, seed);

    // Calculate VaR directly on the GPU
    float gpuVaR = calculateVaRGPU(d_returns, numPaths, 0.95);

    // Free GPU memory
    cudaFree(d_states);
    cudaFree(d_returns);

    return gpuVaR;
}
