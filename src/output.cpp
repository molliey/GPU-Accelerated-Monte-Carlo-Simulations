#include "output.h"
#include <iostream>
#include <fstream>

void outputResults(float cpuVaR, float gpuVaR, const std::string& outputPath) {
    std::cout << "CPU VaR (95% Confidence Level): " << cpuVaR << std::endl;
    std::cout << "GPU VaR (95% Confidence Level): " << gpuVaR << std::endl;

    std::ofstream outFile(outputPath);
    if (outFile.is_open()) {
        outFile << "Method,VaR\n";
        outFile << "CPU VaR: " << cpuVaR << "\n";
        outFile << "GPU VaR: " << gpuVaR << "\n";
        outFile.close();
    } else {
        std::cerr << "Failed" << outputPath << std::endl;
    }
}
