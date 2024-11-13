#include "simulate_gpu.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

__global__ void monteCarloKernel(float* d_returns, float mean, float stdDev, int numPaths) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPaths) return;

    curandState state;
    curand_init(1234, tid, 0, &state);

    d_returns[tid] = mean + stdDev * curand_normal(&state);
}

std::vector<float> simulateReturnsGPU(float mean, float stdDev, int numPaths) {
    float* d_returns;
    cudaMalloc(&d_returns, numPaths * sizeof(float));

    int blockSize = 256;
    int gridSize = (numPaths + blockSize - 1) / blockSize;
    monteCarloKernel<<<gridSize, blockSize>>>(d_returns, mean, stdDev, numPaths);

    std::vector<float> h_returns(numPaths);
    cudaMemcpy(h_returns.data(), d_returns, numPaths * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_returns);

    return h_returns;
}
