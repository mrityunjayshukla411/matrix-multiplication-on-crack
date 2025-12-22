#pragma once
#include "matrix/Matrix.h"

constexpr size_t BLOCK_SIZE = 16;
template <typename T>
class SharedMemCachingKernel
{
public:
    void compute(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C);
    const char *name() const { return "SharedMemCaching"; }
};