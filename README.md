# GPU-Accelerated Monte Carlo Simulations for VaR (Value at Risk)

## OVERVIEW
This project implements a **GPU-accelerated Monte Carlo Simulation** for estimating **Value at Risk (VaR)**, a critical metric in financial risk management. Using **GPU** and **CUDA** techniques, it executes end-to-end simulations on the GPU and achieves substantial speedups over a CPU baseline, especially at large path counts.

**Monte Carlo Simulation**：A computational method that estimates quantities by repeatedly sampling random variables and aggregating outcomes; highly parallel and well-suited to GPUs.

**VaR (Value at Risk)**：A risk metric estimating the maximum expected loss over a specified time horizon at a chosen confidence level (e.g., 95% or 99%).

## ACCELERATION TECHNIQUES

**Random Number Generation - cuRAND**：
- NVIDIA’s GPU library for fast, parallel generation of random numbers (e.g., uniform, normal) directly on the device.

**Sorting - Thrust**：
- A C++ parallel algorithms library (CUDA backend) providing STL-like primitives (e.g., sort, reduce, transform) that operate on GPU memory.

  
**Blocksize Configuration**：
- Threads per CUDA block; controls occupancy, latency hiding, and memory access patterns, impacting kernel throughput.

  
**CUDA machines Comparision**：


## EXPERIMENT and ANALYSIS

### Block Size Optimization

The experiment was conducted with three scales numPaths (100,000; 1,000,000; 10,000,000) and four block sizes (128, 256, 512, 1024), mainly to compare the GPU speedup and identify the optimal block size configuration.

- For small numPaths, blocksize **128** or **256** has better results due to ability to minimize idle threads and synchronization costs.
- Blocksize **1024** underperforms in most cases, especially for smaller numPaths, as it introduces inefficiencies in thread utilization and increases synchronization overhead. 
- Blocksize **512** achieves the best performance balance, offering consistent speedup across a wide range of numPaths, particularly for
medium to large scale workloads. 
- Overall, blocksize **512** serves as the optimal configuration.


### CUDA Machine Comparison

The experiment was conducted across four different CUDA machines and compare the speedup (GPU Time/CPU Time) among them.

#### CUDA Machines:
- CUDA2 (NVIDIA GeForce RTX 2080 Ti, RAM 256GB)
- CUDA3 (NVIDIA TITAN V)
- CUDA4 (NVIDIA TITAN X)
- CUDA5 (TITAN Z)

#### Results:
- CUDA 2 exhibited the least speedup and is better suited for minimal workloads.
- CUDA 3 showed strong performance and scalability, excelling in medium to large workloads with competitive speedups for high workloads.
- CUDA 4 provided steady and reliable growth, making it a viable option for moderate workloads.
- CUDA 5 consistently achieved the highest speedup across all workload sizes, demonstrating exceptional performance for larger workloads.

#### Graph:



## CONCLUSION

This project successfully demonstrates the effectiveness of GPU-based parallelization for Monte Carlo simulations in VaR estimation. By optimizing random number generation with cuRAND and sorting with Thrust, substantial performance improvements and scalability were achieved. This work highlights the transformative impact of GPU parallelization on computationally intensive financial tasks, paving the way for faster and more precise risk assessment in real-time scenarios. 


## INSTALLATION

### Prerequisites
- CUDA Toolkit (Version 11.0 or later)
- A CUDA-capable GPU
- g++ (C++ compiler supporting C++11 standard)
- Make (for building the project)

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

### Running the Program
The program simulates portfolio returns for different numPaths and benchmarks CPU vs. GPU performance. Provide numPaths as a command-line argument.
```bash
./bin/monte_carlo_sim <numPaths>
```
### Running for Multiple `numPaths`
The project also includes an automated script to run multiple simulations and save the results to `results.txt`. Update the list of `numPaths` values in `main.cpp` to customize.



