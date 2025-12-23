#pragma once
#include "matrix/Matrix.h"

template <typename T>
class Tiling1DKernel
{
public:
    void compute(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C);
    const char *name() const { return "Tiling1D"; }
};
