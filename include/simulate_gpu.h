#ifndef SIMULATE_GPU_H
#define SIMULATE_GPU_H

#include <vector>

std::vector<float> simulateReturnsGPU(float meanReturn, float stdDev, int numPaths);

#endif