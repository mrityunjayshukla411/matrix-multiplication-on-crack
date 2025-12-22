#include "kernels/SharedMemCachingKernel.h"
#include "matrix/Matrix.h"

template <typename T>
__global__ void sharedMemCachingKernel(const T *A, const T *B, T *C, size_t M, size_t K, size_t N)
{
    const size_t cRow = blockIdx.x;
    const size_t cCol = blockIdx.y;

    __shared__ T As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE * BLOCK_SIZE];

    const size_t threadCol = threadIdx.x % BLOCK_SIZE;
    const size_t threadRow = threadIdx.x / BLOCK_SIZE;

    const T *A_start = A + cRow * BLOCK_SIZE * K;
    const T *B_start = B + cCol * BLOCK_SIZE;
    T *C_start = C + cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

    T tmp = 0.0;

    for (size_t bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE)
    {
        As[threadRow * BLOCK_SIZE + threadCol] = A_start[threadRow * K + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B_start[threadRow * N + threadCol];

        __syncthreads();

        for (size_t dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++)
        {
            tmp += As[threadRow * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + threadCol];
        }

        __syncthreads();

        A_start += BLOCK_SIZE;
        B_start += BLOCK_SIZE * N;
    }

    C_start[threadRow * N + threadCol] = tmp;
}

template <typename T>
void SharedMemCachingKernel<T>::compute(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C)
{
    size_t M = A.m_rows;
    size_t K = A.m_cols;
    size_t N = B.m_cols;

    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sharedMemCachingKernel<T><<<gridDim, blockDim>>>(A.m_d_data, B.m_d_data, C.m_d_data, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit template instantiations
template class SharedMemCachingKernel<float>;
template class SharedMemCachingKernel<double>;
template class SharedMemCachingKernel<int>;