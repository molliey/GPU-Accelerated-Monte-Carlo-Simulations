#include "simulate_cpu.h"
#include <vector>
#include <random>

std::vector<float> simulateReturnsCPU(float meanReturn, float stdDev, int numPaths) {
    std::vector<float> returns(numPaths);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(meanReturn, stdDev);

    for (int i = 0; i < numPaths; ++i) {
        returns[i] = dist(gen);
    }
    return returns;
}
