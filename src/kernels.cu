#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

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

// 复用课上的 smem-tree 方法.
// ncu 不能用 (错误提示需要独占设备才能使用)
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
            smem[tid] += smem[tid + s]; // no bank conflict
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

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

    // 显然这个是个访存密集型的
    // 先在 cpu 端筛选数据肯定效果比直接传在 gpu 端筛选好很多
    T* d_diags = nullptr;
    RUNTIME_CHECK(cudaMalloc((void**)&d_diags, N * sizeof(T)));
    RUNTIME_CHECK(cudaMemcpy(d_diags, h_diags.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    T* d_out = nullptr;
    RUNTIME_CHECK(cudaMalloc((void**)&d_out, sizeof(T)));
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
 
// 研究了很久的 Online Softmax, 参考了 https://www.bilibili.com/video/BV1UT421k7rA
// 只加载一遍矩阵, softmax 局部求值 再最后修正.
// 原理听懂了, 但是写不出来.

// 这个版本有重复计算
// 事实上即使这个版本我也不能独立完成, 也是 Gemini 辅助写的

#define BLOCK_SIZE 32

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}
template <>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    return static_cast<T>(v);
}
template <>
__device__ __forceinline__ half from_float<half>(float v) {
    return __float2half(v);
}

template <typename T>
__global__ void flash_attention_kernel(const T* Q, const T* K, const T* V, T* O,
                                       int batch_size, int tgt_len, int src_seq_len,
                                       int q_heads, int kv_heads, int head_dim,
                                       bool is_causal) {
    // 当前 block 中的id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 如前面说的, 张量的映射不直观, 所以定义一下比较好
    const int q_tile = blockIdx.x;
    const int q_head = blockIdx.y;
    const int b = blockIdx.z;

    // 全局 thread id
    const int row = q_tile * BLOCK_SIZE + ty;

    // GQA: 找 gmem 中的映射
    const int group = (kv_heads > 0) ? (q_heads / kv_heads) : 1;
    const int kv_head = (group > 0) ? (q_head / group) : 0;

    const float scale = 1.0f / sqrtf((float)head_dim);

    __shared__ float Qs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Ks[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Vs[BLOCK_SIZE][BLOCK_SIZE];
    
    // S = Q * K^T
    __shared__ float S[BLOCK_SIZE][BLOCK_SIZE];

    // 为了计算局部 softmax.
    __shared__ float row_max[BLOCK_SIZE];
    __shared__ float row_sum[BLOCK_SIZE];

    // // 用每个block的 lane0 来初始化 block 的
    // if (tx == 0) {
    //     if (row < tgt_len) {
    //         row_max[ty] = -INFINITY;
    //         row_sum[ty] = 0.0f;
    //     }
    // }
    // __syncthreads();
    
    // 只用一个 wrap
    if (ty == 0) {
        // Calculate the row index this thread (tx) is responsible for initializing
        int abs_row = q_tile * BLOCK_SIZE + tx;
        if (abs_row < tgt_len) {
            row_max[tx] = -INFINITY;
            row_sum[tx] = 0.0f;
        }
    }
    __syncthreads();

    // row max & row sum
    for (int k0 = 0; k0 < src_seq_len; k0 += BLOCK_SIZE) {
        float score = 0.0f;

        for (int d0 = 0; d0 < head_dim; d0 += BLOCK_SIZE) {
            // Load Q: [row, d] -> Qs[ty][tx]
            if (row < tgt_len && d0 + tx < head_dim) {
                int q_idx = ((b * tgt_len + row) * q_heads + q_head) * head_dim + (d0 + tx);
                Qs[ty][tx] = to_float<T>(Q[q_idx]);
            } else {
                Qs[ty][tx] = 0.0f;
            }

            // 映射处理不好, Gemini 改的.
            int k_row = k0 + tx;
            if (k_row < src_seq_len && d0 + ty < head_dim) {
                int k_idx = ((b * src_seq_len + k_row) * kv_heads + kv_head) * head_dim + (d0 + ty);
                Ks[tx][ty] = to_float<T>(K[k_idx]);
            } else {
                Ks[tx][ty] = 0.0f;
            }
            __syncthreads();

            if (row < tgt_len) {
                int k_col = k0 + tx;
                
                // 只计算 1. 范围内, 2. 下三角
                if (k_col < src_seq_len && (!is_causal || k_col <= row)) {
                    for (int i = 0; i < BLOCK_SIZE; ++i) {
                        score += Qs[ty][i] * Ks[tx][i];
                    }
                }
            }
            __syncthreads();
        }

        // Score
        float s_val = -INFINITY;
        if (row < tgt_len) {
            int k_col = k0 + tx;
            if (k_col < src_seq_len && (!is_causal || k_col <= row)) {
                s_val = score * scale;
            }
        }
        S[ty][tx] = s_val;
        __syncthreads();


        // Softmax
        // Reduce Max (destroys S[ty][tx])
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tx < stride) {
                S[ty][tx] = fmaxf(S[ty][tx], S[ty][tx + stride]);
            }
            __syncthreads();
        }
        float tile_max = S[ty][0];

        // Compute Sum(exp(s - tile_max))
        // Use s_val (register) because S is corrupted
        float expv = 0.0f;
        if (row < tgt_len) {
            int k_col = k0 + tx;
            if (k_col < src_seq_len && s_val > -INFINITY / 2) {
                expv = expf(s_val - tile_max);
            }
        }
        S[ty][tx] = expv; // Reuse S for sum reduction
        __syncthreads();

        // Reduce Sum
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tx < stride) {
                S[ty][tx] += S[ty][tx + stride];
            }
            __syncthreads();
        }
        float tile_sum = S[ty][0];

        // Online Softmax Update
        if (tx == 0 && row < tgt_len) {
            float old_max = row_max[ty];
            float old_sum = row_sum[ty];
            float new_max = fmaxf(old_max, tile_max);
            // new_sum = sum(exp(x - new_max))
            //         = old_sum * exp(old_max - new_max) + tile_sum * exp(tile_max - new_max)
            float new_sum = old_sum * expf(old_max - new_max) + tile_sum * expf(tile_max - new_max);
            
            row_max[ty] = new_max;
            row_sum[ty] = new_sum;
        }
        __syncthreads();
    }

    // 到现在, 我们有了全局最大, 全局分母
    // 但是 Q, K, S 都被污染了, 因为新的 block 会覆盖 smem
    // 之前的 S 虽然计算过, 但是没有保存, 我们不得不重新计算一下了.
    // 显然这不是最优, 但是最优的我写不出来. 
    // Gemini 推荐(并辅助我)写的这个, 因为计算相比较于写全局内存, 确实要更划算.
    for (int d0 = 0; d0 < head_dim; d0 += BLOCK_SIZE) {
        float acc = 0.0f;

        // Iterate over K/V tiles
        for (int k0 = 0; k0 < src_seq_len; k0 += BLOCK_SIZE) {
            float score = 0.0f;

            // Recompute Scores
            for (int dd = 0; dd < head_dim; dd += BLOCK_SIZE) {
                // Load Q
                if (row < tgt_len && dd + tx < head_dim) {
                    int q_idx = ((b * tgt_len + row) * q_heads + q_head) * head_dim + (dd + tx);
                    Qs[ty][tx] = to_float<T>(Q[q_idx]);
                } else {
                    Qs[ty][tx] = 0.0f;
                }
                
                // Load K
                int k_row = k0 + tx;
                if (k_row < src_seq_len && dd + ty < head_dim) {
                    int k_idx = ((b * src_seq_len + k_row) * kv_heads + kv_head) * head_dim + (dd + ty);
                    Ks[tx][ty] = to_float<T>(K[k_idx]);
                } else {
                    Ks[tx][ty] = 0.0f;
                }
                __syncthreads();

                // Accumulate dot for score(row, k)
                if (row < tgt_len) {
                    int k_col = k0 + tx;

                    if (k_col < src_seq_len && (!is_causal || k_col <= row)) {
                        for (int i = 0; i < BLOCK_SIZE; ++i) {
                            score += Qs[ty][i] * Ks[tx][i];
                        }
                    }
                }
                __syncthreads();
            }

            // Write Score to S
            if (row < tgt_len) {
                int k_col = k0 + tx;
                float s = (k_col < src_seq_len) ? score * scale : -INFINITY;
                if (is_causal && k_col > row) s = -INFINITY;
                S[ty][tx] = s;
            } else {
                S[ty][tx] = -INFINITY;
            }
            __syncthreads();

            // Vs[k_in_tile][d_in_tile] -> Vs[ty][tx] loads V[k0+ty, d0+tx]
            int v_row = k0 + ty;
            if (v_row < src_seq_len && d0 + tx < head_dim) {
                int v_idx = ((b * src_seq_len + v_row) * kv_heads + kv_head) * head_dim + (d0 + tx);
                Vs[ty][tx] = to_float<T>(V[v_idx]);
            } else {
                Vs[ty][tx] = 0.0f;
            }
            __syncthreads();

            // Output[row, d0+tx] += sum_k( P[row, k] * V[k, d0+tx] )
            if (row < tgt_len && d0 + tx < head_dim) {
                float global_max = row_max[ty];
                float global_sum = row_sum[ty];
                // Gemini 说要填一个这个小数, 防止分母为 0
                // 似乎是没有使用 safesoftmax 的原因, 不清楚对实际的影响.
                float inv_sum = 1.0f / (global_sum + 1e-6f);

                for (int kk = 0; kk < BLOCK_SIZE; ++kk) {
                    int k_col = k0 + kk;
                    if (k_col < src_seq_len) { 
                        float prob = expf(S[ty][kk] - global_max) * inv_sum;
                        acc += prob * Vs[kk][tx];
                    }
                }
            }
            __syncthreads();
        }

        if (row < tgt_len && d0 + tx < head_dim) {
            int o_idx = ((b * tgt_len + row) * q_heads + q_head) * head_dim + (d0 + tx);
            O[o_idx] = from_float<T>(acc);
        }
    }
}


template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {    
    // 编译期检查，只允许 float 或 int
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, half>::value,
        "flash attention only supports float or half types."
    );
        
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return;
    }

    const size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    const size_t k_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    const size_t v_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    const size_t o_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;

    T *d_q, *d_k, *d_v, *d_o;

    RUNTIME_CHECK(cudaMalloc((void**)&d_q, q_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc((void**)&d_k, k_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc((void**)&d_v, v_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc((void**)&d_o, o_size * sizeof(T)));

    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemset(d_o, 0, o_size * sizeof(T)));

    // block size 是定死的, 就是一个 tile 的大小
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // 这里的张量映射非常不直观, 张量是 [bs, seq, head, d], 
    // 理论上这里应该是 [head, seq, bs],
    // 或者 [bs, seq, head]
    // 但是 Gemini 给我检查的时候, 死活不让.
    // > 通常情况下，不会 这样映射。
    // 在 CUDA 中，gridDim.x 是三维网格中限制最宽松的维度（最大 2^31 -1)
    // 而 gridDim.y 和 gridDim.z 的限制较严（最大 65535）
    // 虽然在逻辑上看着顺眼（Batch, Seq, Head），但在实际使用中存在隐患.
    // 所以没办法, 后面的映射确实也挺困难.
    dim3 grid(
        (target_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,
        query_heads,
        batch_size
    );

    flash_attention_kernel<<<grid, block>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal
    );
    
    RUNTIME_CHECK(cudaDeviceSynchronize());
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));

    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));
}

#undef BLOCK_SIZE

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
