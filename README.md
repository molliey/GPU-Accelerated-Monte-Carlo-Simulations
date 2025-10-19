# GPU-Accelerated Monte Carlo Simulations for VaR (Value at Risk)

## OVERVIEW
This project implements a **GPU-accelerated Monte Carlo Simulation** for estimating **Value at Risk (VaR)**, a critical metric in financial risk management. Using **GPU** and **CUDA** techniques, it executes end-to-end simulations on the GPU and achieves substantial speedups over a CPU baseline, especially at large path counts.

**Monte Carlo Simulation**：A computational method that estimates quantities by repeatedly sampling random variables and aggregating outcomes; highly parallel and well-suited to GPUs.

**VaR (Value at Risk)**：A risk metric estimating the maximum expected loss over a specified time horizon at a chosen confidence level.

## ACCELERATION TECHNIQUES

**Random Number Generation - cuRAND**：
- NVIDIA’s GPU library for fast, parallel generation of random numbers (e.g., uniform, normal) directly on the device.
- cuRAND produces Gaussian draws directly on device.

**Sorting - Thrust**：
- C++ parallel algorithms library (CUDA) providing STL-like primitives (e.g., sort, reduce, transform) operates on GPU memory.
- Replace naïve GPU/CPU sorts with Thrust device-side sort (radix/merge under the hood).

**Data Transfer**
- Keep the entire Monte Carlo VaR pipeline on the GPU (RNG → simulation → sort/percentile) and send back only the final VaR scalar to the CPU to eliminate large host–device transfers that hurt performance.

**Blocksize Configuration**：
- Threads per CUDA block; controls occupancy, latency hiding, and memory access patterns, impacting kernel throughput.
- Tested {128, 256, 512, 1024} across 100,000; 1,000,000; 10,000,00 numPaths.
- 512 delivered the most speedups (balance of occupancy, memory access, and synchronization).
  
**CUDA Machines**：
- CUDA2 (NVIDIA GeForce RTX 2080 Ti; RAM 256GB)
- CUDA3 (NVIDIA TITAN V; RAM: 128GB)
- CUDA4 (NVIDIA TITAN X; RAM: 128GB)
- CUDA5 (NVIDIA TITAN Z; RAM: 64GB)

## EXPERIMENT and ANALYSIS

The experiment was conducted across four different CUDA machines and compare the speedup (GPU Time/CPU Time) among them.

**Results**:
- CUDA 2 exhibited the least speedup and is better suited for minimal workloads.
- CUDA 3 showed strong performance and scalability, excelling in medium to large workloads with competitive speedups.
- CUDA 4 provided steady and reliable growth, making it a viable option for moderate workloads.
- CUDA 5 consistently achieved the highest speedup across all workload sizes and exceptional performance for larger workloads.

**Graph**:
- Graph1 uses a logarithmic scale for the x-axis, focusing on precise comparisons at smaller numPaths.
- Graph2 uses a linear x-axis with scientific notation to label the numPaths, highlighting trends in speedup as the workload increases.

<img width="471" height="333" alt="Image" src="https://github.com/user-attachments/assets/29dcd38c-5e47-45df-96ed-8b90210b9952" />

<img width="464" height="366" alt="Image" src="https://github.com/user-attachments/assets/ccffb3ea-fe8c-4d0f-a702-4d27cc783032" />



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



