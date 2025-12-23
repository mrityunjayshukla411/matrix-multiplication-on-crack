#include "kernels/Tiling1DKernel.h"
#include "matrix/Matrix.h"

template <typename T, const int BM, const int BN, const int BK, const int TM>
__global__ void tiling1DKernel(const T *A, const T *B, T *C, size_t M, size_t K, size_t N)
{
    const size_t cRow = blockIdx.x;
    const size_t cCol = blockIdx.y;

    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    const size_t threadCol = threadIdx.x % BN;
    const size_t threadRow = threadIdx.x / BN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    size_t innerColA = threadIdx.x % BK;
    size_t innerRowA = threadIdx.x / BK;
    size_t innerColB = threadIdx.x % BN;
    size_t innerRowB = threadIdx.x / BN;


    T threadResults[TM]= {0.0};

    for (size_t bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // Load A tile into shared memory with boundary check
        size_t aRow = cRow * BM + innerRowA;
        size_t aCol = bkIdx + innerColA;
        As[innerRowA * BK + innerColA] = (aRow < M && aCol < K) ? A[innerRowA * K + innerColA] : 0.0;

        // Load B tile into shared memory with boundary check
        size_t bRow = bkIdx + innerRowB;
        size_t bCol = cCol * BN + innerColB;
        Bs[innerRowB * BN + innerColB] = (bRow < K && bCol < N) ? B[innerRowB * N + innerColB] : 0.0;

        __syncthreads();

        A += BK;
        B += BK * N;

        for (size_t dotIdx = 0; dotIdx < BK; dotIdx++)
        {
          float tmpB = Bs[dotIdx * BN + threadCol];
          for(size_t resIdx = 0; resIdx < TM ; resIdx++)
          {
            threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
          }
        }

        __syncthreads();
    }

    for(size_t resIdx = 0 ; resIdx < TM ; resIdx++)
    {
      size_t row = cRow * BM + threadRow * TM + resIdx;
      size_t col = cCol * BN + threadCol;
      if (row < M && col < N) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
      }
    }
}

template <typename T>
void Tiling1DKernel<T>::compute(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C)
{
    size_t M = A.m_rows;
    size_t K = A.m_cols;
    size_t N = B.m_cols;

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    dim3 blockDim((BM*BN)/TM);
    dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);

    tiling1DKernel<T, BM, BN, BK, TM><<<gridDim, blockDim>>>(A.m_d_data, B.m_d_data, C.m_d_data, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit template instantiations
template class Tiling1DKernel<float>;
template class Tiling1DKernel<double>;
template class Tiling1DKernel<int>;
