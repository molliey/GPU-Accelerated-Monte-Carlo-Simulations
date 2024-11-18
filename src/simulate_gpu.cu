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

float calculateVaRGPU(float* d_returns, int numPaths, float confidenceLevel) {
    // Wrap the raw pointer in a Thrust device vector
    thrust::device_vector<float> d_vec(d_returns, d_returns + numPaths);

    // Sort the vector on the GPU using Thrust
    thrust::sort(d_vec.begin(), d_vec.end());

    // Calculate the VaR (5th percentile for 95% confidence level)
    int index = static_cast<int>((1.0 - confidenceLevel) * numPaths);
    return -d_vec[index];
}

void simulateReturnsGPU(int numPaths, float mean, float stdDev, float* d_returns, curandState* d_states) {
    int blockSize = 512;
    int gridSize = (numPaths + blockSize - 1) / blockSize;

    unsigned long seed = clock();
    monteCarloCombinedKernel<<<gridSize, blockSize>>>(d_returns, numPaths, mean, stdDev, d_states, seed);

    // Synchronize to ensure the kernel has finished
    cudaDeviceSynchronize();
}
