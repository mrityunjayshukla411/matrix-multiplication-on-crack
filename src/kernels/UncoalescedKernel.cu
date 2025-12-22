#include "kernels/UncoalescedKernel.h"
#include "matrix/Matrix.h"

template <typename T>
__global__ void uncoalescedKernel(const T *A, const T *B, T *C, size_t M, size_t K, size_t N)
{
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N)
  {
    T tmp = 0.0;
    for (size_t i = 0; i < K; i++)
    {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = tmp;
  }
}

template <typename T>
void UncoalescedKernel<T>::compute(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C)
{
  size_t M = A.m_rows;
  size_t K = A.m_cols;
  size_t N = B.m_cols;

  dim3 blockDim(16, 16);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

  uncoalescedKernel<T><<<gridDim, blockDim>>>(A.m_d_data, B.m_d_data, C.m_d_data, M, K, N);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit template instantiations
template class UncoalescedKernel<float>;
template class UncoalescedKernel<double>;
template class UncoalescedKernel<int>;
