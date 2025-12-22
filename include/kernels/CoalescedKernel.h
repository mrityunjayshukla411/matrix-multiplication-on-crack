#pragma once
#include "matrix/Matrix.h"

template<typename T>
class CoalescedKernel {
public:
    void compute(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
    
    const char* name() const { return "Coalesced"; }
};
