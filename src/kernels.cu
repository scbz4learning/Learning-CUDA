#include <vector>
#include <cuda_fp16.h>
#include <iostream>

#include "../tester/utils.h"

template <typename T>
__global__ void trace_kernel(T* A, size_t N, size_t SIZE, double *p) {
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < SIZE && i % (N+1) == 0) {
		*p += static_cast<double>(A[i]);
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
  // TODO: Implement the trace function
  if (rows == 0 || cols == 0)
    return T(0);
  const size_t N = rows<cols ? rows : cols;
  const size_t SIZE = rows*cols;
  dim3 grid_size(256);
  dim3 block_size(
		  (N+grid_size.x-1)/grid_size.x
		 );

  T* d_input;
  RUNTIME_CHECK(cudaMalloc((void**) &d_input, SIZE * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(T), cudaMemcpyHostToDevice));

  double *p;
  RUNTIME_CHECK(cudaMalloc((void**) &p, sizeof(double)));
  RUNTIME_CHECK(cudaMemset(p, 0, sizeof(double)));

  trace_kernel<<<grid_size,block_size>>>(d_input, N, SIZE, p);

  double ans;
  RUNTIME_CHECK(cudaMemcpy(&ans, p, sizeof(double), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(p));

  return static_cast<T>(ans);
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
