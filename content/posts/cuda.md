---
title: CUDA 常用算子案例
subtitle:
date: 2025-08-07T00:18:46+08:00
slug: ebaa040
draft: false
author:
    name: "yitao"
emoji: true
categories: [找工作]
tags: [八股]
---

记录一些常用的 CUDA 算子写法

<!--more-->


## Reduce

### Reduce Sum

```CUDA
template <const int kWarpSize = 256>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
```


### Block reduce sum
```CUDA
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;

    static __shared__ reduce_sum[NUM_WARPS];

    float sum = warp_reduce_sum_f32<WARP_SIZE>(val);

    if (lane == 0)
        reduce_sum[warp] = sum;
    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_sum[lane] : 0.0f;
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    sum = __shfl_sync(0xffffffff, sum, 0, 32);
    return sum;
}
```

### Reduce Max

```CUDA
template <const int kWarpSize = 256>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
```

### block reduce max
```CUDA
// grid 1D block 1D, grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    static __shared__ float reduce_max[NUM_WARPS];

    float value = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) {
        shared[warp] = value;
    }   
    __syncthreads();

    value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
    value = warp_reduce_max_f32<NUM_WARPS>(value);

    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}
```

## Softmax

### naive版
```CUDA
template <const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    float exp_val = (idx < N) ? expf[x[idx]] : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

    if (idx < N) {
        y[idx] = exp_val / exp_sum;
    }
}
```

### 向量化存取

```CUDA
#define FLOAT4(value) (reinterpret_cast<int4 *>(&(value))[0])
template <const int NUM_THREADS = 256>
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x + tid) * 4;

    // 向量化 取
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_exp;
    reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
    reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
    reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
    reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;

    float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

    // 向量化 存
    if (idx + 3 < N) {
       float4 reg_y;
       reg_y.x = reg_exp.x / exp_sum;
       reg_y.y = reg_exp.y / exp_sum;
       reg_y.z = reg_exp.z / exp_sum;
       reg_y.w = reg_exp.w / exp_sum;
       FLOAT4[y[idx]] = reg_y;
    }
}
```


### safe-softmax
```CUDA

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    float val = (idx < N) ? x[idx] : -FLT_MAX;
    float max_val = block_reduce_max_f32<NUM_THREADS>(val);
    float exp_val = (idx < N) ? expf[x[idx] - max_val] : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

    if (idx < N)
        y[idx] = exp_val / exp_sum;
}
```


### online-softmax

```CUDA
struct __align__(8) MD {
    float M;
    float D;
};

template <const int NUM_THREADS = 256>
__global__ void online_softmax_f32_per_token_kernel(float *x, float *y, int N) {
    //
}

__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float psum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++ k) {
            psum += a[m * K + k] * [k * N + n];
        }
        c[m * M + n] = psum;
    }
}

template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {
    // block tile ： 32x32 的 block 处理 c 上一块 32x32 的元素计算
    // K tile：      使用共享内存，将 K 分块成 BK 大小的块
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.y * blockDim.x + tx;

    int load_smem_a_m = tid / 32;
    int load_smem_a_n = tid % 32;
    int load_smem_b_n = tid / 32;
    int load_smem_b_k = tid % 32;

    int load_gmem_a_m = by * BM + load_smem_a_m;  // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n;  // global col of b and c

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    float sum = 0.0f;
    for (int bk = 0; bk < (K + BK - 1) / BK; ++ bk) {
        // 加载 a 的全局内容到共享内存
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_a[load_gmem_a_m][load_gmem_a_k] = a[load_gmem_a_addr];

        // 加载 b 的全局内容到共享内存
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        s_b[load_gmem_b_k][load_gmem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < BK; ++ k) {
        // 共享内存内进行 block gemm
        sum += s_a[load_smem_a_m][k] * s_b[k][load_smem_b_n];
    }
    __syncthreads();

    // 存
    int store_gmem_c_addr = load_gmem_a_m * N + load_gmem_b_n;
    c[store_gmem_c_addr] = sum;
}
```

## Matmul

### naive 版

```CUDA
__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float psum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++ k) {
            psum += a[m * K + k] * [k * N + n];
        }
        c[m * M + n] = psum;
    }
}
```

### shared_mem 优化

```CUDA
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {
    // block tile ： 32x32 的 block 处理 c 上一块 32x32 的元素计算
    // K tile：      使用共享内存，将 K 分块成 BK 大小的块
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.y * blockDim.x + tx;

    int load_smem_a_m = tid / 32;
    int load_smem_a_n = tid % 32;
    int load_smem_b_n = tid / 32;
    int load_smem_b_k = tid % 32;

    int load_gmem_a_m = by * BM + load_smem_a_m;  // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n;  // global col of b and c

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    float sum = 0.0f;
    for (int bk = 0; bk < (K + BK - 1) / BK; ++ bk) {
        // 加载 a 的全局内容到共享内存
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_a[load_gmem_a_m][load_gmem_a_k] = a[load_gmem_a_addr];

        // 加载 b 的全局内容到共享内存
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        s_b[load_gmem_b_k][load_gmem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < BK; ++ k) {
        // 共享内存内进行 block gemm
        sum += s_a[load_smem_a_m][k] * s_b[k][load_smem_b_n];
    }
    __syncthreads();

    // 存
    int store_gmem_c_addr = load_gmem_a_m * N + load_gmem_b_n;
    c[store_gmem_c_addr] = sum;
}
```

## Transpose

### naive

```CUDA
__global__ void transpose_naive(float *input, float *output, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        output[n * M + m] = input[m * N + n];
    }
}
```
