/**
 * @file Tiling1DKernel.cu
 * @brief 1D thread-level tiling: each thread computes TM output elements per column.
 *
 * Extends shared memory blocking by having each thread compute a vertical strip
 * of TM output elements rather than a single element. This increases arithmetic
 * intensity (more FMAs per global memory byte) and better utilizes register files,
 * while keeping shared memory usage the same as the basic tiled kernel.
 *
 * Tiling parameters (compile-time constants):
 *   BM = 64  Block tile rows
 *   BN = 64  Block tile columns
 *   BK = 8   Block tile K-depth (inner dimension per step)
 *   TM = 8   Thread tile rows (each thread owns TM consecutive output rows)
 *
 * Block configuration:
 *   Threads per block: (BM * BN) / TM = 512
 *   Each thread is identified by (threadRow, threadCol) where:
 *     threadCol = threadIdx.x % BN  → which output column
 *     threadRow = threadIdx.x / BN  → which group of TM output rows
 *
 * Shared memory usage: As[BM×BK] + Bs[BK×BN] = 64*8 + 8*64 = 1024 elements
 *   = 4 KB for float.
 *
 * The inner compute loop caches a single B element (tmpB) in a register and reuses
 * it across all TM rows — exploiting data reuse along the row dimension.
 *
 * Boundary handling: out-of-bounds global indices are zero-padded in shared memory,
 * allowing the kernel to handle non-multiple-of-BM/BN/BK matrix sizes correctly.
 */
#include "kernels/Tiling1DKernel.h"
#include "matrix/Matrix.h"

/**
 * @brief GPU kernel: 1D thread-level tiling for matrix multiplication.
 *
 * Launch config: <<<gridDim, blockDim>>> where blockDim = (BM*BN/TM,) = (512,)
 * and gridDim = (ceil(M/BM), ceil(N/BN)).
 *
 * Shared memory: As[BM*BK] + Bs[BK*BN].
 * Each thread accumulates TM partial results in the threadResults[] register array.
 *
 * @tparam T   Element type.
 * @tparam BM  Block tile M (rows per block).
 * @tparam BN  Block tile N (cols per block).
 * @tparam BK  Block tile K (inner dimension per step).
 * @tparam TM  Thread tile M (rows per thread).
 *
 * @param A  Device pointer to input matrix A (M×K, row-major).
 * @param B  Device pointer to input matrix B (K×N, row-major).
 * @param C  Device pointer to output matrix C (M×N, row-major).
 * @param M  Number of rows in A and C.
 * @param K  Shared inner dimension.
 * @param N  Number of columns in B and C.
 */
template <typename T, const int BM, const int BN, const int BK, const int TM>
__global__ void tiling1DKernel(const T *A, const T *B, T *C, size_t M, size_t K, size_t N)
{
    const size_t cRow = blockIdx.x;
    const size_t cCol = blockIdx.y;

    // Shared memory tiles for the current BK-slice of A and B.
    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    // Each thread's output position within the BM×BN block tile.
    // threadRow × TM ... threadRow × TM + (TM-1) are the rows this thread owns.
    const size_t threadCol = threadIdx.x % BN;
    const size_t threadRow = threadIdx.x / BN;

    // Move global pointers to this block's top-left corner.
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Indices for cooperatively loading the A tile: all threads collectively fill
    // the BM×BK tile, each loading one element based on its linear thread index.
    size_t innerColA = threadIdx.x % BK;
    size_t innerRowA = threadIdx.x / BK;
    // Indices for loading the B tile (BK×BN).
    size_t innerColB = threadIdx.x % BN;
    size_t innerRowB = threadIdx.x / BN;

    // Per-thread accumulator: TM values corresponding to TM consecutive output rows.
    T threadResults[TM]= {0.0};

    for (size_t bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // Load A tile into shared memory with boundary check.
        // Threads not mapping to valid global indices contribute zero.
        size_t aRow = cRow * BM + innerRowA;
        size_t aCol = bkIdx + innerColA;
        As[innerRowA * BK + innerColA] = (aRow < M && aCol < K) ? A[innerRowA * K + innerColA] : 0.0;

        // Load B tile into shared memory with boundary check.
        size_t bRow = bkIdx + innerRowB;
        size_t bCol = cCol * BN + innerColB;
        Bs[innerRowB * BN + innerColB] = (bRow < K && bCol < N) ? B[innerRowB * N + innerColB] : 0.0;

        __syncthreads();

        // Advance A and B pointers by BK for the next iteration.
        A += BK;
        B += BK * N;

        // Compute TM partial products for this BK slice.
        // Cache Bs[dotIdx][threadCol] in a register to avoid repeated shared mem reads
        // across the TM rows — each B value is reused TM times.
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

    // Write TM results back to global memory, respecting boundaries.
    for(size_t resIdx = 0 ; resIdx < TM ; resIdx++)
    {
      size_t row = cRow * BM + threadRow * TM + resIdx;
      size_t col = cCol * BN + threadCol;
      if (row < M && col < N) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
      }
    }
}

/**
 * @brief Launches the 1D tiling GEMM kernel and synchronizes.
 *
 * Tiling params: BM=64, BN=64, BK=8, TM=8.
 * Grid: ceil(M/64) × ceil(N/64). Block: 512 threads (1D).
 *
 * @param A  Input matrix A (device buffer must be populated).
 * @param B  Input matrix B (device buffer must be populated).
 * @param C  Output matrix C (device buffer overwritten).
 */
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

    // 512 threads per block: each computes TM=8 output elements.
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
