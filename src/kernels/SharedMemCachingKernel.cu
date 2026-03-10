/**
 * @file SharedMemCachingKernel.cu
 * @brief Tiled matrix multiplication using shared memory to reduce global memory traffic.
 *
 * This kernel implements the classic "shared memory blocking" optimization:
 * each block loads a BLOCK_SIZE×BLOCK_SIZE tile of A and B into shared memory,
 * then all threads within the block compute partial dot products from those tiles.
 * This converts repeated global memory accesses (once per block per K step in
 * CoalescedKernel) into fast shared memory reads, reducing global bandwidth by
 * roughly BLOCK_SIZE×.
 *
 * Key design choices:
 *   - Uses a 1D thread layout (BLOCK_SIZE*BLOCK_SIZE threads per block) to
 *     simplify shared memory indexing. threadCol = threadIdx.x % BLOCK_SIZE,
 *     threadRow = threadIdx.x / BLOCK_SIZE.
 *   - Two __syncthreads() barriers per tile: one after loading to ensure all
 *     threads see the complete tile before computing, one after computing to
 *     ensure shared memory is safe to overwrite with the next tile.
 *   - A_start and B_start are advanced by BLOCK_SIZE each iteration to step
 *     through the K dimension.
 *
 * Shared memory usage: 2 × BLOCK_SIZE² × sizeof(T) bytes per block (= 8 KB
 * for BLOCK_SIZE=16, float).
 *
 * @note BLOCK_SIZE is defined as a constexpr in SharedMemCachingKernel.h (= 16).
 */
#include "kernels/SharedMemCachingKernel.h"
#include "matrix/Matrix.h"

/**
 * @brief GPU kernel: tiled matrix multiplication with shared memory caching.
 *
 * Launch config: <<<gridDim, blockDim>>> where blockDim = (BLOCK_SIZE²) = (256,)
 * and gridDim = (ceil(M/BLOCK_SIZE), ceil(N/BLOCK_SIZE)).
 *
 * Shared memory usage: As[BLOCK_SIZE×BLOCK_SIZE] + Bs[BLOCK_SIZE×BLOCK_SIZE].
 *
 * Each block is responsible for computing the BLOCK_SIZE×BLOCK_SIZE output tile
 * C[cRow*BS : (cRow+1)*BS][cCol*BS : (cCol+1)*BS].
 *
 * @param A  Device pointer to input matrix A (M×K, row-major).
 * @param B  Device pointer to input matrix B (K×N, row-major).
 * @param C  Device pointer to output matrix C (M×N, row-major).
 * @param M  Number of rows in A and C.
 * @param K  Shared inner dimension.
 * @param N  Number of columns in B and C.
 */
template <typename T>
__global__ void sharedMemCachingKernel(const T *A, const T *B, T *C, size_t M, size_t K, size_t N)
{
    // Block indices identify which output tile this block computes.
    const size_t cRow = blockIdx.x;
    const size_t cCol = blockIdx.y;

    // Shared memory tiles: each BLOCK_SIZE×BLOCK_SIZE elements.
    __shared__ T As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE * BLOCK_SIZE];

    // Decode 1D thread index into 2D (row, col) within the tile.
    const size_t threadCol = threadIdx.x % BLOCK_SIZE;
    const size_t threadRow = threadIdx.x / BLOCK_SIZE;

    // Each block's starting position in the full A, B, C matrices.
    const T *A_start = A + cRow * BLOCK_SIZE * K;
    const T *B_start = B + cCol * BLOCK_SIZE;
    T *C_start = C + cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

    T tmp = 0.0;

    // Tile over the K dimension in steps of BLOCK_SIZE.
    for (size_t bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE)
    {
        // Each thread loads one element of the A tile and one of the B tile
        // into shared memory. No bounds check here — assumes M, K, N are
        // exact multiples of BLOCK_SIZE (padding should be added for general use).
        As[threadRow * BLOCK_SIZE + threadCol] = A_start[threadRow * K + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B_start[threadRow * N + threadCol];

        // Barrier: all threads must finish loading before any thread computes.
        __syncthreads();

        // Compute partial dot product using the loaded tile from shared memory.
        for (size_t dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++)
        {
            tmp += As[threadRow * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + threadCol];
        }

        // Barrier: ensure computation is complete before loading the next tile,
        // preventing a race condition where fast threads overwrite shared memory
        // before slower threads finish reading it.
        __syncthreads();

        // Advance to the next BLOCK_SIZE columns of A and rows of B.
        A_start += BLOCK_SIZE;
        B_start += BLOCK_SIZE * N;
    }

    C_start[threadRow * N + threadCol] = tmp;
}

/**
 * @brief Launches the shared memory caching GEMM kernel and synchronizes.
 *
 * Grid: ceil(M/BLOCK_SIZE) × ceil(N/BLOCK_SIZE) blocks.
 * Block: BLOCK_SIZE² = 256 threads (1D).
 *
 * @param A  Input matrix A (device buffer must be populated).
 * @param B  Input matrix B (device buffer must be populated).
 * @param C  Output matrix C (device buffer overwritten).
 */
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
