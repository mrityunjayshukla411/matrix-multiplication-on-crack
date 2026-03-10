/**
 * @file TestUtils.h
 * @brief Matrix comparison and test result printing utilities.
 *
 * TestUtils<T> provides element-wise comparison of two matrices using a
 * combined absolute + relative tolerance check — the standard approach for
 * validating floating-point GPU results against a CPU reference.
 */
#pragma once
#include "matrix/Matrix.h"
#include "utils/Colors.h"
#include <cmath>
#include <iostream>
#include <iomanip>

/**
 * @brief Stateless test utilities for matrix correctness checking.
 *
 * @tparam T Element type (float, double, int).
 */
template<typename T>
class TestUtils {
public:
    /**
     * @brief Element-wise comparison of two matrices with mixed tolerance.
     *
     * An element pair (a, b) is considered erroneous if:
     *   |a - b| > abs_tolerance  AND  |a - b| > rel_tolerance * max(|a|, |b|)
     *
     * Using both tolerances avoids false positives near zero (where a purely
     * relative check becomes overly strict) while still catching large relative
     * errors away from zero.
     *
     * Compares host buffers (m_h_data). Call cudaMemcpy DeviceToHost before
     * comparing GPU results.
     *
     * @param A             Reference (CPU) matrix.
     * @param B             Computed (GPU) matrix to validate.
     * @param rel_tolerance Relative error threshold. Default 1e-5.
     * @param abs_tolerance Absolute error threshold. Default 1e-8.
     * @return true if all elements pass the tolerance check, false otherwise.
     */
    static bool compareMatrices(const Matrix<T>& A, const Matrix<T>& B,
                                T rel_tolerance = 1e-5, T abs_tolerance = 1e-8) {
        if (A.m_rows != B.m_rows || A.m_cols != B.m_cols) {
            std::cerr << "Matrix dimensions mismatch: "
                      << A.m_rows << "x" << A.m_cols << " vs "
                      << B.m_rows << "x" << B.m_cols << std::endl;
            return false;
        }

        size_t errors = 0;
        T max_error = 0;
        size_t first_error_idx = 0;

        for (size_t i = 0; i < A.size(); ++i) {
            T diff = std::abs(A.m_h_data[i] - B.m_h_data[i]);
            T magnitude = std::max(std::abs(A.m_h_data[i]), std::abs(B.m_h_data[i]));

            // Check both relative and absolute tolerance
            bool error = diff > abs_tolerance && diff > rel_tolerance * magnitude;

            if (error) {
                if (errors == 0) {
                    first_error_idx = i;
                }
                errors++;
                max_error = std::max(max_error, diff);
            }
        }

        if (errors > 0) {
            size_t row = first_error_idx / A.m_cols;
            size_t col = first_error_idx % A.m_cols;
            std::cerr << Colors::BOLD_RED << "Matrix comparison failed!" << Colors::RESET << std::endl;
            std::cerr << Colors::RED << "Total errors: " << Colors::RESET << errors << " / " << A.size()
                      << " (" << std::fixed << std::setprecision(2)
                      << (100.0 * errors / A.size()) << "%)" << std::endl;
            std::cerr << Colors::RED << "Max error: " << Colors::RESET << std::scientific << max_error << std::endl;
            std::cerr << Colors::RED << "First error at [" << row << "," << col << "]: " << Colors::RESET
                      << "expected " << A.m_h_data[first_error_idx]
                      << ", got " << B.m_h_data[first_error_idx]
                      << " (diff: " << std::abs(A.m_h_data[first_error_idx] - B.m_h_data[first_error_idx])
                      << ")" << std::endl;
            return false;
        }

        return true;
    }

    /**
     * @brief Prints a colored [PASS] or [FAIL] line for a named test case.
     *
     * @param test_name Human-readable test description.
     * @param passed    Result of the test.
     */
    static void printTestResult(const char* test_name, bool passed) {
        if (passed) {
            std::cout << "[" << Colors::BOLD_GREEN << "PASS" << Colors::RESET << "] "
                      << test_name << std::endl;
        } else {
            std::cout << "[" << Colors::BOLD_RED << "FAIL" << Colors::RESET << "] "
                      << Colors::RED << test_name << Colors::RESET << std::endl;
        }
    }
};
