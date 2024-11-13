#include "calculate.h"
#include <vector>
#include <algorithm>

float calculateVaR(std::vector<float>& returns, float confidenceLevel) {
    std::sort(returns.begin(), returns.end());
    int index = static_cast<int>((1.0 - confidenceLevel) * returns.size());
    return -returns[index];
}
