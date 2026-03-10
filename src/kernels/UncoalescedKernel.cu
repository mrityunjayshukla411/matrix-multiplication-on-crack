/**
 * @file UncoalescedKernel.cu
 * @brief Baseline naïve matrix multiplication with uncoalesced memory access.
 *
 * This kernel maps threadIdx.x to the row (M) dimension and threadIdx.y to the
 * column (N) dimension. Consequently, adjacent threads in a warp access consecutive
 * rows of B — a stride-N access pattern — which prevents memory coalescing and
 * results in many separate cache-line fetches. It serves as a performance baseline
 * to quantify the benefit of proper memory access patterns.
 *
 * Memory access pattern:
 *   - A: thread (x, y) reads row x sequentially. Across the warp, threads differ
 *        in x → different rows → strided global reads of A.
 *   - B: thread (x, y) reads down column y → B[i*N + y]. Adjacent threads in a warp
 *        differ in x but share y → the same column element is read redundantly.
 *        Not coalesced.
 *   - C: written at C[x*N + y] → adjacent threads write to the same row,
 *        different columns only if y varies, but y is tied to blockIdx.y here.
 */
#include "kernels/UncoalescedKernel.h"
#include "matrix/Matrix.h"

/**
 * @brief GPU kernel: naïve matrix multiplication with uncoalesced memory access.
 *
 * Launch config: <<<gridDim, blockDim>>> where blockDim = (16, 16).
 * gridDim = (ceil(M/16), ceil(N/16)).
 *
 * Each thread computes one output element C[x][y] = sum_i A[x][i] * B[i][y].
 * threadIdx.x → row (x), threadIdx.y → col (y). Adjacent threads in a warp
 * differ in x (not y), making B accesses non-coalesced.
 *
 * @param A  Device pointer to input matrix A (M×K, row-major).
 * @param B  Device pointer to input matrix B (K×N, row-major).
 * @param C  Device pointer to output matrix C (M×N, row-major).
 * @param M  Number of rows in A and C.
 * @param K  Shared inner dimension (cols of A, rows of B).
 * @param N  Number of columns in B and C.
 */
template <typename T>
__global__ void uncoalescedKernel(const T *A, const T *B, T *C, size_t M, size_t K, size_t N)
{
  // x = row index, y = col index.
  // threadIdx.x covers rows: adjacent warp threads → adjacent rows → non-coalesced.
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

/**
 * @brief Launches the uncoalesced GEMM kernel and synchronizes the device.
 *
 * Grid: ceil(M/16) blocks in x (rows), ceil(N/16) blocks in y (cols).
 * Block: 16×16 = 256 threads.
 *
 * @param A  Input matrix A (device buffer must be populated).
 * @param B  Input matrix B (device buffer must be populated).
 * @param C  Output matrix C (device buffer overwritten).
 */
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
