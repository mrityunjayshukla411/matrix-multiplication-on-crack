#include <iostream>
#include <iomanip>
#include "matrix/Matrix.h"
#include "matrix/MatrixInitializer.h"
#include "kernels/CoalescedKernel.h"
#include "kernels/UncoalescedKernel.h"
#include "kernels/SharedMemCachingKernel.h"
#include "utils/CudaUtils.h"
#include "utils/CpuMatMul.h"
#include "utils/TestUtils.h"
#include "utils/Colors.h"

template<typename T, template<typename> class KernelType>
bool testKernel(const char* kernel_name, size_t M, size_t N, size_t K) {
    std::cout << "\nTesting " << kernel_name << " with dimensions: "
              << M << "x" << K << " * " << K << "x" << N << std::endl;

    // Create matrices
    Matrix<T> A(M, K);
    Matrix<T> B(K, N);
    Matrix<T> C_gpu(M, N);
    Matrix<T> C_cpu(M, N);

    // Initialize with random data
    MatrixInitializer<T>::initialize(A, MatrixInitializer<T>::Pattern::RANDOM_UNIFORM);
    MatrixInitializer<T>::initialize(B, MatrixInitializer<T>::Pattern::RANDOM_UNIFORM);

    // Compute CPU reference
    CpuMatMul<T>::compute(A, B, C_cpu);

    // Transfer inputs to device
    CUDA_CHECK(cudaMemcpy(A.m_d_data, A.m_h_data, A.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B.m_d_data, B.m_h_data, B.bytes(), cudaMemcpyHostToDevice));

    // Compute GPU result
    KernelType<T> kernel;
    kernel.compute(A, B, C_gpu);

    // Transfer result back
    CUDA_CHECK(cudaMemcpy(C_gpu.m_h_data, C_gpu.m_d_data, C_gpu.bytes(), cudaMemcpyDeviceToHost));

    // Compare results
    bool passed = TestUtils<T>::compareMatrices(C_cpu, C_gpu);

    return passed;
}

template<typename T>
void runAllTests() {
    int passed = 0;
    int total = 0;

    std::cout << "\n" << Colors::BOLD_CYAN << "========================================"
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_CYAN << "Running Matrix Multiplication Tests"
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_CYAN << "========================================"
              << Colors::RESET << std::endl;

    // Test CoalescedKernel
    struct TestCase {
        const char* name;
        size_t M, N, K;
    };

    TestCase test_cases[] = {
        {"Small square (16x16)", 16, 16, 16},
        {"Small rectangular (32x16x24)", 32, 16, 24},
        {"Medium square (128x128)", 128, 128, 128},
        {"Medium rectangular (256x128x64)", 256, 128, 64},
        {"Large square (512x512)", 512, 512, 512},
        {"Large rectangular (1024x512x256)", 1024, 512, 256},
        {"Very tall (2048x64x128)", 2048, 64, 128},
        {"Very wide (64x2048x128)", 64, 2048, 128},
    };

    std::cout << "\n" << Colors::BOLD_YELLOW << "--- Testing UncoalescedKernel ---"
              << Colors::RESET << std::endl;
    for (const auto& tc : test_cases) {
        total++;
        bool result = testKernel<T, UncoalescedKernel>(tc.name, tc.M, tc.N, tc.K);
        TestUtils<T>::printTestResult(tc.name, result);
        if (result) passed++;
    }

    std::cout << "\n" << Colors::BOLD_YELLOW << "--- Testing CoalescedKernel ---"
              << Colors::RESET << std::endl;
    for (const auto& tc : test_cases) {
        total++;
        bool result = testKernel<T, CoalescedKernel>(tc.name, tc.M, tc.N, tc.K);
        TestUtils<T>::printTestResult(tc.name, result);
        if (result) passed++;
    }

    std::cout << "\n" << Colors::BOLD_YELLOW << "--- Testing SharedMemCachingKernel ---"
              << Colors::RESET << std::endl;
    for (const auto& tc : test_cases) {
        total++;
        bool result = testKernel<T, SharedMemCachingKernel>(tc.name, tc.M, tc.N, tc.K);
        TestUtils<T>::printTestResult(tc.name, result);
        if (result) passed++;
    }


    // Print summary
    std::cout << "\n" << Colors::BOLD_CYAN << "========================================"
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_CYAN << "Test Summary" << Colors::RESET << std::endl;
    std::cout << Colors::BOLD_CYAN << "========================================"
              << Colors::RESET << std::endl;
    std::cout << Colors::BOLD << "Total tests: " << Colors::RESET << total << std::endl;
    std::cout << Colors::BOLD_GREEN << "Passed: " << Colors::RESET << passed << std::endl;

    if (total - passed > 0) {
        std::cout << Colors::BOLD_RED << "Failed: " << Colors::RESET << (total - passed) << std::endl;
    } else {
        std::cout << Colors::DIM << "Failed: 0" << Colors::RESET << std::endl;
    }

    std::cout << Colors::BOLD << "Success rate: " << Colors::RESET
              << std::fixed << std::setprecision(1)
              << (100.0 * passed / total) << "%" << std::endl;

    if (passed == total) {
        std::cout << "\n" << Colors::BOLD_GREEN << "✓ All tests PASSED!"
                  << Colors::RESET << std::endl;
    } else {
        std::cout << "\n" << Colors::BOLD_RED << "✗ Some tests FAILED!"
                  << Colors::RESET << std::endl;
    }
}

int main() {
    printDeviceInfo();

    // Run tests with float type
    runAllTests<float>();

    return 0;
}
