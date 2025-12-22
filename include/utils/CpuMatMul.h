#pragma once
#include "matrix/Matrix.h"
#include <cstring>

template<typename T>
class CpuMatMul {
public:
    // CPU reference implementation for matrix multiplication
    // C = A * B where A is MxK, B is KxN, C is MxN
    static void compute(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        size_t M = A.m_rows;
        size_t K = A.m_cols;
        size_t N = B.m_cols;

        // Zero out result matrix
        std::memset(C.m_h_data, 0, C.bytes());

        // Standard row-major matrix multiplication
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    sum += A.m_h_data[i * K + k] * B.m_h_data[k * N + j];
                }
                C.m_h_data[i * N + j] = sum;
            }
        }
    }
};
