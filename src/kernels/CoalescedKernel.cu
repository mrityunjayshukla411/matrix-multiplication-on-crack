/**
 * @file CoalescedKernel.cu
 * @brief Matrix multiplication with coalesced global memory access.
 *
 * The key improvement over UncoalescedKernel: threadIdx.x maps to the column (N)
 * dimension and threadIdx.y to the row (M) dimension. This means adjacent threads
 * in a warp (consecutive threadIdx.x values) access consecutive columns of C and B,
 * producing 128-byte coalesced cache-line loads and stores.
 *
 * Memory access pattern:
 *   - B: thread (row, col) reads B[k * N + col]. Across a warp, col increments
 *        → coalesced loads from global memory.
 *   - C: written at C[row * N + col] → coalesced writes across the warp.
 *   - A: each thread reads row `row` sequentially. Threads in a warp differ in
 *        `row`, so A reads are strided — but this is unavoidable without shared mem.
 *
 * Still reads A and B directly from global memory on every compute step; shared
 * memory caching (SharedMemCachingKernel) reduces redundant global reads further.
 */
#include "kernels/CoalescedKernel.h"
#include "utils/CudaUtils.h"

/**
 * @brief GPU kernel: matrix multiplication with coalesced memory access.
 *
 * Launch config: <<<gridDim, blockDim>>> where blockDim = (16, 16).
 * gridDim = (ceil(N/16), ceil(M/16)).
 *
 * threadIdx.x → col, threadIdx.y → row. Adjacent warp threads have adjacent
 * col values → coalesced reads of B and coalesced writes of C.
 *
 * @param A  Device pointer to input matrix A (M×K, row-major).
 * @param B  Device pointer to input matrix B (K×N, row-major).
 * @param C  Device pointer to output matrix C (M×N, row-major).
 * @param M  Number of rows in A and C.
 * @param K  Shared inner dimension.
 * @param N  Number of columns in B and C.
 */
template<typename T>
__global__  void coalescedKernel(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {

    // threadIdx.x → col: adjacent threads → adjacent columns → coalesced.
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

/**
 * @brief Launches the coalesced GEMM kernel and synchronizes the device.
 *
 * Grid: ceil(N/16) blocks in x (cols), ceil(M/16) blocks in y (rows).
 * Block: 16×16 = 256 threads.
 *
 * @param A  Input matrix A (device buffer must be populated).
 * @param B  Input matrix B (device buffer must be populated).
 * @param C  Output matrix C (device buffer overwritten).
 */
template<typename T>
void CoalescedKernel<T>::compute(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    size_t M = A.m_rows;
    size_t K = A.m_cols;
    size_t N = B.m_cols;

    dim3 blockDim(16, 16);
    // x covers columns (N), y covers rows (M) — matching threadIdx.x→col mapping.
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    coalescedKernel<T><<<gridDim, blockDim>>>(A.m_d_data, B.m_d_data, C.m_d_data, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit template instantiations
template class CoalescedKernel<float>;
template class CoalescedKernel<double>;
template class CoalescedKernel<int>;
