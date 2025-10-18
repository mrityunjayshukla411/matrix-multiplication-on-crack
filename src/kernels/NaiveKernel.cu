#include "kernels/NaiveKernel.h"
#include "utils/CudaUtils.h"

template<typename T>
__global__  void naiveKernel(const T* A, const T* B, T* C, size_t M, size_t N, size_t K) {

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < K)
    {
        T sum = 0;
        for (size_t n = 0; n < N; ++n) {
            sum += A[row * N + n] * B[n * K + col];
        }
        C[row * K + col] = sum;
    }
}

template<typename T>
void NaiveKernel<T>::compute(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    size_t M = A.m_rows;
    size_t N = A.m_cols;
    size_t K = B.m_cols;

    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    naiveKernel<T><<<gridDim, blockDim>>>(A.m_d_data, B.m_d_data, C.m_d_data, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit template instantiations
template class NaiveKernel<float>;
template class NaiveKernel<double>;
template class NaiveKernel<int>; 