#include "calculate.h"
#include <vector>
#include <algorithm>

float calculateVaR(const std::vector<float>& returns, float confidenceLevel) {
    std::vector<float> sortedReturns = returns;
    std::sort(sortedReturns.begin(), sortedReturns.end()); 
    
    int index = static_cast<int>((1.0 - confidenceLevel) * sortedReturns.size());
    return -sortedReturns[index];
}