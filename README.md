# Matrix Multiplication on Crack

> A progressive series of CUDA GEMM optimizations, from naïve global-memory kernels to register-tiled, async-pipelined implementations — with a live benchmark harness.

## Overview

This project systematically implements and benchmarks six GPU matrix multiplication (GEMM) kernels, each introducing a new optimization technique on top of the previous one. The goal is to understand *why* each optimization works by measuring its concrete impact on throughput (GFLOPS) and execution time.

Starting from a naïve uncoalesced kernel and progressing through coalescing, shared memory tiling, 1D thread-level tiling, software pipelining with `cp.async`, and finally 2D thread-level tiling with outer-product accumulation, the project demonstrates how modern GPU GEMM kernels approach cuBLAS-level performance through a sequence of well-understood micro-architectural techniques.

All kernels are verified for correctness against a CPU reference implementation before being benchmarked, and the benchmark harness uses CUDA event timers with warm-up runs to produce accurate, reproducible results.

## Features

- **6 progressively optimized CUDA GEMM kernels** in a unified interface
- **Automated correctness testing** against CPU reference via `matmul_test`
- **Live performance benchmark** with GFLOPS, timing, and speedup tables via `matmul_benchmark`
- **Colorized terminal output** with per-kernel pass/fail and fastest-kernel highlighting
- Supports **arbitrary matrix dimensions** (non-power-of-two, rectangular) with boundary handling
- Template-parameterized kernels supporting `float`, `double`, and `int`
- CUDA event-based timing with warm-up runs for accurate measurements

## Kernel Progression

| Kernel | Key Technique | Expected Benefit |
|---|---|---|
| `Uncoalesced` | Naïve global memory (non-coalesced) | Baseline |
| `Coalesced` | Thread→column mapping for coalesced access | 2–4× over uncoalesced |
| `SharedMemCaching` | BLOCK_SIZE×BLOCK_SIZE shared memory tiles | Reduce global memory traffic |
| `Tiling1D` | Thread computes TM=8 output rows per column | Higher arithmetic intensity |
| `Tiling1D-Async` | `cp.async` double-buffered pipeline (Ampere+) | Overlap load and compute |
| `Tiling2D-Vectorized` | TM×TN=8×8 outer product per thread | Maximum register reuse |

## Requirements

- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer, for `cp.async` support in `Tiling1DKernelAsync`)
  - All other kernels work on older architectures
- CUDA Toolkit ≥ 11.4
- CMake ≥ 3.28
- C++17-capable compiler (GCC ≥ 9 or Clang ≥ 9)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd matrix-multiplication-on-crack

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The build produces two executables:
- `build/matmul_benchmark` — performance comparison across all kernels
- `build/matmul_test` — correctness validation against CPU reference

## Usage

**Run benchmarks:**
```bash
./build/matmul_benchmark
```

Example output:
```
Device0: NVIDIA RTX 5090
Compute Capability: 12.0
...

========================================
Matrix Multiplication: 2048x2048 * 2048x2048
========================================

Kernel                    Time (ms)         GFLOPS        Speedup
Uncoalesced               85.312          103.73          1.00x
Coalesced                 18.241          485.21          4.68x
SharedMemCaching           8.103         1092.35         10.53x
Tiling1D                   3.412         2595.18         25.00x
Tiling1D-Async             2.891         3062.44         29.51x
Tiling2D-Vectorized        1.832         4836.11         46.57x  ⚡

⚡ Fastest: Tiling2D-Vectorized (4836.11 GFLOPS)
```

**Run correctness tests:**
```bash
./build/matmul_test
```

## Project Structure

```
.
├── include/
│   ├── kernels/          # Kernel class declarations (.h)
│   │   ├── UncoalescedKernel.h
│   │   ├── CoalescedKernel.h
│   │   ├── SharedMemCachingKernel.h
│   │   ├── Tiling1DKernel.h
│   │   ├── Tiling1DKernelAsync.h
│   │   └── Tiling2DKernel.h
│   ├── matrix/
│   │   ├── Matrix.h              # Dual-buffer (pinned host + device) matrix class
│   │   └── MatrixInitializer.h   # Host-side initialization patterns
│   └── utils/
│       ├── CudaUtils.h           # CUDA_CHECK macro + device info printer
│       ├── CudaTimer.h           # RAII CUDA event timer
│       ├── CpuMatMul.h           # CPU reference implementation
│       ├── TestUtils.h           # Matrix comparison with tolerance
│       └── Colors.h              # ANSI terminal color constants
├── src/
│   ├── main.cpp                  # Benchmark driver
│   ├── test.cpp                  # Correctness test driver
│   └── kernels/                  # Kernel implementations (.cu)
├── CMakeLists.txt
└── README.md
```

## How It Works

### Memory Hierarchy Optimization Path

The kernel sequence targets the GPU memory hierarchy one level at a time:

1. **Uncoalesced → Coalesced**: Fix the thread-to-memory mapping so adjacent warp threads access adjacent memory addresses, enabling 128-byte cache-line coalescing.

2. **Coalesced → SharedMemCaching**: Load a BLOCK_SIZE×BLOCK_SIZE tile of A and B into shared memory once, then have all threads in the block read from it — reducing global memory traffic by BLOCK_SIZE×.

3. **SharedMemCaching → Tiling1D**: Each thread now computes TM=8 output elements instead of 1, increasing the FMA-to-load ratio and better utilizing registers. A cached B register (`tmpB`) is reused across TM rows.

4. **Tiling1D → Tiling1D-Async**: Use the Ampere `cp.async` PTX instruction to issue global→shared memory copies asynchronously, allowing the compute unit and memory unit to work in parallel. Double buffering (NUM_STAGES=2) keeps one tile loading while the other is being consumed.

5. **Tiling1D-Async → Tiling2D**: Extend thread tiling to 2D (TM×TN = 8×8). The inner loop uses an outer product formulation — loading a column from A and a row from B into registers once, then computing TM×TN FMAs — maximizing arithmetic intensity and register reuse.
