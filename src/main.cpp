/**
 * @file main.cpp
 * @brief Benchmark driver: compares all GEMM kernel implementations across matrix sizes.
 *
 * Runs a progressive set of square and rectangular matrix multiplication benchmarks,
 * printing a formatted performance table (kernel name, average time, GFLOPS, speedup
 * relative to the slowest kernel) for each problem size. The fastest kernel is
 * highlighted with a lightning bolt symbol.
 *
 * Benchmark methodology:
 *   - 3 warm-up runs (to populate GPU caches and JIT-compile any deferred code).
 *   - 10 timed runs using CudaTimer (CUDA events for GPU-side timing accuracy).
 *   - Average time and GFLOPS are computed from the 10 timed runs.
 *   - GFLOPS formula: (2 * M * N * K) / (avg_time_ms * 1e6)
 *     The factor of 2 accounts for one multiply and one add per element of the
 *     inner product (2 FLOPs per K iteration).
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "matrix/Matrix.h"
#include "matrix/MatrixInitializer.h"
#include "kernels/CoalescedKernel.h"
#include "kernels/UncoalescedKernel.h"
#include "kernels/SharedMemCachingKernel.h"
#include "kernels/Tiling1DKernel.h"
#include "kernels/Tiling1DKernelAsync.h"
#include "kernels/Tiling2DKernel.h"
#include "utils/CudaTimer.h"
#include "utils/CudaUtils.h"
#include "utils/Colors.h"

/**
 * @brief Stores the result of benchmarking one kernel variant.
 */
struct BenchmarkResult {
    std::string kernel_name; ///< Human-readable name from KernelType::name().
    float time_ms;           ///< Average execution time in milliseconds.
    double gflops;           ///< Effective throughput in GFLOPS.
};

/**
 * @brief Benchmarks a single kernel type with warm-up and averaged timing.
 *
 * Runs the kernel 3 times for warm-up (to avoid cold-start effects), then
 * times it 10 times using CUDA events and returns the average.
 *
 * GFLOPS = (2 * M * N * K) / (avg_time_ms * 1e6).
 *
 * @tparam T          Element type (e.g., float).
 * @tparam KernelType Template class implementing compute() and name().
 *
 * @param A  Input matrix A (M×K), device data already populated.
 * @param B  Input matrix B (K×N), device data already populated.
 * @param C  Output matrix C (M×N), overwritten on each call.
 * @param M  Rows of A.
 * @param N  Cols of B.
 * @param K  Inner dimension.
 * @return   BenchmarkResult with kernel name, average time, and GFLOPS.
 */
template <typename T, template<typename> class KernelType>
BenchmarkResult benchmarkKernel(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, size_t M, size_t N, size_t K)
{
    KernelType<T> kernel;

    // Warm-up runs: ensure GPU is at steady-state clock and caches are primed.
    for (int i = 0; i < 3; ++i) {
        kernel.compute(A, B, C);
    }

    // Timed runs - average over multiple iterations
    const int num_runs = 10;
    float total_time = 0.0f;
    CudaTimer timer;

    for (int i = 0; i < num_runs; ++i) {
        timer.start();
        kernel.compute(A, B, C);
        total_time += timer.stop();
    }

    float avg_time = total_time / num_runs;
    // 2 FLOPs per inner-product element (multiply + add), divided by time in ns.
    double gflops = (2.0 * M * N * K) / (avg_time * 1e6);

    return {kernel.name(), avg_time, gflops};
}

/**
 * @brief Benchmarks all kernels for a given M×K × K×N matrix multiplication.
 *
 * Allocates matrices, initializes with random uniform data, transfers to the
 * device, then runs and times all kernels. Prints a formatted table with timing,
 * GFLOPS, and speedup relative to the slowest kernel. Highlights the fastest.
 *
 * @tparam T  Element type (e.g., float).
 * @param M   Rows of output matrix C.
 * @param N   Columns of output matrix C.
 * @param K   Inner dimension (cols of A, rows of B).
 */
template <typename T>
void runComparison(size_t M, size_t N, size_t K)
{
    std::cout << "\n" << Colors::BOLD_CYAN << "========================================"
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_CYAN << "Matrix Multiplication: " << Colors::RESET
              << Colors::BOLD_WHITE << M << "x" << K << " * " << K << "x" << N
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_CYAN << "Output: " << Colors::RESET
              << Colors::BOLD_WHITE << M << "x" << N << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_CYAN << "========================================"
              << Colors::RESET << std::endl;

    // Create matrices
    Matrix<T> A(M, K);
    Matrix<T> B(K, N);
    Matrix<T> C(M, N);

    // Initialize with random data
    MatrixInitializer<T>::initialize(A, MatrixInitializer<T>::Pattern::RANDOM_UNIFORM);
    MatrixInitializer<T>::initialize(B, MatrixInitializer<T>::Pattern::RANDOM_UNIFORM);

    // Transfer to device
    CUDA_CHECK(cudaMemcpy(A.m_d_data, A.m_h_data, A.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B.m_d_data, B.m_h_data, B.bytes(), cudaMemcpyHostToDevice));

    // Benchmark all kernels
    std::vector<BenchmarkResult> results;

    std::cout << "\n" << Colors::BOLD_YELLOW << "Benchmarking kernels..."
              << Colors::RESET << std::endl;

    results.push_back(benchmarkKernel<T, UncoalescedKernel>(A, B, C, M, N, K));
    std::cout << "  " << Colors::GREEN << "✓ " << Colors::RESET
              << results.back().kernel_name << " completed" << std::endl;

    results.push_back(benchmarkKernel<T, CoalescedKernel>(A, B, C, M, N, K));
    std::cout << "  " << Colors::GREEN << "✓ " << Colors::RESET
              << results.back().kernel_name << " completed" << std::endl;

    results.push_back(benchmarkKernel<T, SharedMemCachingKernel>(A, B, C, M, N, K));
    std::cout << "  " << Colors::GREEN << "✓ " << Colors::RESET
              << results.back().kernel_name << " completed" << std::endl;
    results.push_back(benchmarkKernel<T, Tiling1DKernel>(A, B, C, M, N, K));
    std::cout << "  " << Colors::GREEN << "✓ " << Colors::RESET
              << results.back().kernel_name << " completed" << std::endl;
    results.push_back(benchmarkKernel<T, Tiling1DKernelAsync>(A, B, C, M, N, K));
    std::cout << "  " << Colors::GREEN << "✓ " << Colors::RESET
              << results.back().kernel_name << " completed" << std::endl;
    results.push_back(benchmarkKernel<T, Tiling2DKernel>(A, B, C, M, N, K));
    std::cout << "  " << Colors::GREEN << "✓ " << Colors::RESET
              << results.back().kernel_name << " completed" << std::endl;

    // Print results table
    std::cout << "\n" << Colors::BOLD_MAGENTA << "----------------------------------------"
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_MAGENTA << "Performance Comparison" << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_MAGENTA << "----------------------------------------"
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD << std::left << std::setw(25) << "Kernel"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(15) << "Speedup" << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_MAGENTA << "----------------------------------------"
              << Colors::RESET << std::endl;

    // Find baseline (slowest kernel) for speedup calculation.
    float baseline_time = 0.0f;
    for (const auto& result : results) {
        if (result.time_ms > baseline_time) {
            baseline_time = result.time_ms;
        }
    }

    // Find fastest kernel for highlighting
    auto fastest = results[0];
    for (const auto& result : results) {
        if (result.gflops > fastest.gflops) {
            fastest = result;
        }
    }

    // Print each kernel's performance
    for (const auto& result : results) {
        float speedup = baseline_time / result.time_ms;
        bool is_fastest = (result.kernel_name == fastest.kernel_name);

        if (is_fastest) {
            std::cout << Colors::BOLD_GREEN;
        }

        std::cout << std::left << std::setw(25) << result.kernel_name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << result.time_ms
                  << std::setw(15) << std::setprecision(2) << result.gflops
                  << std::setw(14) << std::setprecision(2) << speedup << "x";

        if (is_fastest) {
            std::cout << "  " << Colors::BOLD_YELLOW << "⚡" << Colors::RESET;
        }

        std::cout << Colors::RESET << std::endl;
    }

    std::cout << Colors::BOLD_MAGENTA << "----------------------------------------"
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_YELLOW << "⚡ Fastest: " << Colors::BOLD_GREEN
              << fastest.kernel_name << Colors::RESET
              << " (" << Colors::BOLD_CYAN << std::fixed << std::setprecision(2)
              << fastest.gflops << " GFLOPS" << Colors::RESET << ")" << std::endl;
}

/**
 * @brief Entry point: prints device info and runs all benchmark configurations.
 *
 * Benchmarks square sizes (512, 1024, 2048) and two rectangular configurations
 * to stress-test kernels with non-square tiles and varying aspect ratios.
 */
int main()
{
    printDeviceInfo();

    // Benchmark different matrix sizes
    runComparison<float>(512, 512, 512);
    runComparison<float>(1024, 1024, 1024);
    runComparison<float>(2048, 2048, 2048);

    // Test rectangular matrices
    runComparison<float>(1024, 512, 2048);
    runComparison<float>(2048, 256, 1024);

    return 0;
}
