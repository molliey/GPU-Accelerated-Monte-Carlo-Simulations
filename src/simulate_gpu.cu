#include "simulate_gpu.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>

__global__ void monteCarloKernel(float* d_returns, float mean, float stdDev, int numPaths) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPaths) return;

    curandState state;
    curand_init(clock64(), tid, 0, &state);
    d_returns[tid] = mean + stdDev * curand_normal(&state);
}

std::vector<float> simulateReturnsGPU(int numPaths, float mean, float stdDev) {
    float* d_returns;
    cudaError_t err;
    
    err = cudaMalloc(&d_returns, numPaths * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float>();
    }

    int blockSize = 256;
    int gridSize = (numPaths + blockSize - 1) / blockSize;
    
    monteCarloKernel<<<gridSize, blockSize>>>(d_returns, mean, stdDev, numPaths);
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_returns);
        return std::vector<float>();
    }

    std::vector<float> h_returns(numPaths);
    err = cudaMemcpy(h_returns.data(), d_returns, numPaths * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_returns);
        return std::vector<float>();
    }

    cudaFree(d_returns);

    return h_returns;
}
