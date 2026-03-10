/**
 * @file CpuMatMul.h
 * @brief CPU reference implementation of matrix multiplication for correctness testing.
 *
 * Provides a naïve O(M·N·K) triple-loop implementation that operates entirely on
 * host (pinned) memory. Used in matmul_test to produce a ground-truth result against
 * which all GPU kernels are compared via TestUtils::compareMatrices().
 */
#pragma once
#include "matrix/Matrix.h"
#include <cstring>

/**
 * @brief Stateless CPU matrix multiplier.
 *
 * @tparam T Element type (float, double, int).
 */
template<typename T>
class CpuMatMul {
public:
    /**
     * @brief Computes C = A × B using a naïve triple-nested loop on the host.
     *
     * Operates on the host buffers (m_h_data) of each matrix. No GPU calls are made.
     * The result matrix C is zeroed before accumulation.
     *
     * Time complexity: O(M·N·K) — suitable only for small/medium matrices in tests.
     *
     * @param A  Input matrix of shape M×K (host buffer must be initialized).
     * @param B  Input matrix of shape K×N (host buffer must be initialized).
     * @param C  Output matrix of shape M×N; host buffer is overwritten.
     */
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
