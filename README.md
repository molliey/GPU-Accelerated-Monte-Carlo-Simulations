# GPU-Accelerated Monte Carlo Simulations for VaR (Value at Risk)

## OVERVIEW
This project implements a **GPU-accelerated Monte Carlo simulation** to estimate **Value at Risk (VaR)**, a critical metric in financial risk management. By leveraging **CUDA** and **Thrust**, the implementation demonstrates significant performance improvements over traditional CPU-based methods, particularly for large-scale simulations.

The project includes:
- A Monte Carlo simulation framework for financial risk estimation.
- Optimized GPU implementation for massive parallelism.
- Direct computation of VaR on the GPU using efficient sorting and percentile calculation.
- Performance benchmarking between CPU and GPU implementations.

## FEATURES
1. **Monte Carlo Simulation**:
   - Simulates portfolio returns based on a normal distribution.
   - Generates random paths using **cuRAND** for GPU parallelism.
   
2. **GPU Optimization**:
   - Full GPU computation of VaR using **CUDA** and **Thrust** libraries.
   - Sorting and percentile computation are performed entirely on the GPU to minimize data transfer.

3. **CPU vs. GPU Benchmarking**:
   - Compares performance for various portfolio sizes (`numPaths`) ranging from 10,000 to 10,000,000.
   - Captures execution time for both CPU and GPU implementations.

4. **Scalable Design**:
   - Handles large datasets by dynamically allocating GPU memory.
   - Efficiently uses GPU resources for maximum parallelization.


## INSTALL

### Prerequisites
- **CUDA Toolkit** (Version 11.0 or later)
- A CUDA-capable GPU
- **g++** (C++ compiler supporting C++11 standard)
- **Make** (for building the project)

### Clone the Repository
```bash
git clone <repository-url>
cd GPU-Parallel-Application
```

## Build the Project
Run the following commands to compile the code:
```bash
make clean
make
```

## USAGE

### Running the Program
The program simulates portfolio returns for different numPaths and benchmarks CPU vs. GPU performance. Provide numPaths as a command-line argument.
```bash
./bin/monte_carlo_sim <numPaths>
```
### Running for Multiple `numPaths`
The project also includes an automated script to run multiple simulations and save the results to `results.txt`. Update the list of `numPaths` values in `main.cpp` to customize.

## Experimental Results and Analysis

The project conducted extensive experiments to evaluate performance, focusing on block size optimization and comparison across different CUDA-enabled machines.

### Block Size Optimization

numPaths (100,000; 1,000,000; 10,000,000) and block sizes (128, 256, 512, 1024) showed that block size 512 offered the best performance balance and consistent speedup across a wide range of numPaths, particularly for medium to large-scale workloads.

### CUDA Machine Comparison

Performance was evaluated across four NYU CIMS GPU nodes: CUDA2 (GeForce RTX 2080 Ti), CUDA3 (TITAN V), CUDA4 (TITAN X), and CUDA5 (TITAN Z).

- CUDA 5 consistently achieved the highest speedup across all workload sizes, demonstrating exceptional performance for larger workloads.
- CUDA 3 showed strong performance and scalability, excelling in medium to large workloads with competitive speedups for high workloads.
- CUDA 4 provided steady and reliable growth, making it a viable option for moderate workloads.
- CUDA 2 exhibited the least speedup and is better suited for minimal workloads.



### CONCLUSION

This project successfully demonstrates the effectiveness of GPU-based parallelization for Monte Carlo simulations in VaR estimation. By optimizing random number generation with cuRAND and sorting with Thrust, substantial performance improvements and scalability were achieved. This work highlights the transformative impact of GPU parallelization on computationally intensive financial tasks, paving the way for faster and more precise risk assessment in real-time scenarios. 







