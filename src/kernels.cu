#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

template <typename T>
__global__ void trace_kernel(const T* input, size_t N, T* output) {
  // 编译报错: extern __shared__ 不能直接用模板类型 T
  // 借助 AI 修复
    extern __shared__ unsigned char smem_raw[]; 
    T* smem = reinterpret_cast<T*>(smem_raw);

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < N) ? input[idx] : T(0);
    __syncthreads();

    // tree-based reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // 编译期检查，只允许 float 或 int
    static_assert(std::is_same<T, float>::value || std::is_same<T, int>::value,
                  "trace() only supports float or int types.");
                  
  if (rows == 0 || cols == 0)
    return T(0);
  const size_t N = std::min(rows, cols);
  std::vector<T> h_diags(N);
  for (size_t i = 0; i < N; i++) {
    h_diags[i] = h_input[i * cols + i];
  }

  T* d_diags = nullptr;
  RUNTIME_CHECK(cudaMalloc((void**) &d_diags, N * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_diags, h_diags.data(), N * sizeof(T), cudaMemcpyHostToDevice));

  T* d_out = nullptr;
  RUNTIME_CHECK(cudaMalloc((void**) &d_out, sizeof(T)));
  RUNTIME_CHECK(cudaMemset(d_out, 0, sizeof(T)));

  int block_size = 256;
  int grid_size = (int)((N + block_size - 1) / block_size);
  if (grid_size < 1) grid_size = 1;

  size_t smem_size = block_size * sizeof(T);
  trace_kernel<<<grid_size, block_size, smem_size>>>(d_diags, N, d_out);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  T ans;
  RUNTIME_CHECK(cudaMemcpy(&ans, d_out, sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_diags));
  RUNTIME_CHECK(cudaFree(d_out));

  return ans;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
