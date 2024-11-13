#ifndef SIMULATE_GPU_H
#define SIMULATE_GPU_H

#include <vector>

std::vector<float> simulateReturnsGPU(int numPaths, float meanReturn, float stdDev);

#endif