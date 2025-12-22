#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "matrix/Matrix.h"
#include "matrix/MatrixInitializer.h"
#include "kernels/CoalescedKernel.h"
#include "kernels/UncoalescedKernel.h"
#include "kernels/SharedMemCachingKernel.h"
#include "utils/CudaTimer.h"
#include "utils/CudaUtils.h"
#include "utils/Colors.h"

struct BenchmarkResult {
    std::string kernel_name;
    float time_ms;
    double gflops;
};

template <typename T, template<typename> class KernelType>
BenchmarkResult benchmarkKernel(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, size_t M, size_t N, size_t K)
{
    KernelType<T> kernel;

    // Warm-up runs
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
    double gflops = (2.0 * M * N * K) / (avg_time * 1e6);

    return {kernel.name(), avg_time, gflops};
}

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

    // Find baseline (slowest kernel)
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
