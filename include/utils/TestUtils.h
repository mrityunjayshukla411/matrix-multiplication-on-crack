#pragma once
#include "matrix/Matrix.h"
#include "utils/Colors.h"
#include <cmath>
#include <iostream>
#include <iomanip>

template<typename T>
class TestUtils {
public:
    // Compare two matrices with a relative tolerance
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

    // Print test result with color
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
