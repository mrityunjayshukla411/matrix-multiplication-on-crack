#include <iostream>
#include <iomanip>
#include "matrix/Matrix.h"
#include "matrix/MatrixInitializer.h"
#include "kernels/NaiveKernel.h"
#include "utils/CudaTimer.h"
#include "utils/CudaUtils.h"

template<typename T>
void runBenchmark(int M, int N, int K) {
    std::cout << "\n=== Matrix Multiplication: " 
              << M << "x" << K << " * " << K << "x" << N 
              << " ===" << std::endl;
    
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
    
    // Create kernel
    NaiveKernel<T> kernel;
    
    // Warm-up run
    kernel.compute(A, B, C);
    
    // Timed run
    CudaTimer timer;
    timer.start();
    kernel.compute(A, B, C);
    float elapsed_ms = timer.stop();
    
    // Transfer result back
    CUDA_CHECK(cudaMemcpy(C.m_h_data, C.m_d_data, C.bytes(), cudaMemcpyDeviceToHost));
    
    // Calculate performance
    double gflops = (2.0 * M * N * K) / (elapsed_ms * 1e6);
    
    std::cout << "Kernel: " << kernel.name() << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(3) 
              << elapsed_ms << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) 
              << gflops << " GFLOPS" << std::endl;
    
    // Verify a few elements (sanity check)
    std::cout << "Sample results (first 3 elements of C): ";
    for (size_t i = 0; i < 3 && i < C.size(); ++i) {
        std::cout << C.m_h_data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    printDeviceInfo();
    
    // Test with different sizes
    runBenchmark<float>(512, 512, 512);
    runBenchmark<float>(1024, 1024, 1024);
    
    return 0;
}