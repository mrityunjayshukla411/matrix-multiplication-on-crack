#include "kernels/CoalescedKernel.h"
#include "utils/CudaUtils.h"

template<typename T>
__global__  void coalescedKernel(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N)
    {
        T sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template<typename T>
void CoalescedKernel<T>::compute(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    size_t M = A.m_rows;
    size_t K = A.m_cols;
    size_t N = B.m_cols;

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    coalescedKernel<T><<<gridDim, blockDim>>>(A.m_d_data, B.m_d_data, C.m_d_data, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit template instantiations
template class CoalescedKernel<float>;
template class CoalescedKernel<double>;
template class CoalescedKernel<int>; 
