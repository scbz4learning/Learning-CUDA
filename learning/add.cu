#include "utils.h"
#include <vector>
#include <iostream>
using namespace std;

template <typename T>
__global__ void add_kernel(T *d_a, T *d_b, T *d_c, size_t SIZE) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE) {
        d_c[i] = d_a[i] + d_b[i];
    }
} 

template <typename T>
int run_add() {
    const size_t SIZE = 1 << 20;
    vector<T> a(SIZE, 0);
    vector<T> b(SIZE, 1);

    vector<T> c_cpu(SIZE, 0);

    for (int i = 0; i < SIZE; i++) {
        c_cpu[i] = a[i] + b[i];
    }

    vector<T> c(SIZE, 0);

    T *d_a, *d_b, *d_c;
    const size_t BYTES = SIZE * sizeof(T); 
    RUNTIME_CHECK(cudaMalloc((void**)&d_a, BYTES));
    RUNTIME_CHECK(cudaMalloc((void**)&d_b, BYTES));
    RUNTIME_CHECK(cudaMalloc((void**)&d_c, BYTES));
    RUNTIME_CHECK(cudaMemcpy(d_a, a.data(), BYTES, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_b, b.data(), BYTES, cudaMemcpyHostToDevice));
    // 初始化 d_c 为 0
    RUNTIME_CHECK(cudaMemset(d_c, 0, BYTES));

    int block_size = 256;
    int grid_size = (SIZE + block_size - 1) / block_size;

    add_kernel<T><<<grid_size, block_size>>>(d_a, d_b, d_c, SIZE);

    RUNTIME_CHECK(cudaDeviceSynchronize());  // 显式同步，确保内核完成

    RUNTIME_CHECK(cudaMemcpy(c.data(), d_c, BYTES, cudaMemcpyDeviceToHost));
    
    int ans = 0;

    for (int i = 0; i < SIZE; i++) {
        if (c_cpu[i] != c[i]){
            ans++;
        }
    }

    cout << ans << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ans;
}

int main () {
    run_add<int>();
    run_add<float>();
}
