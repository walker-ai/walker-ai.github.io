# CUDA 常用算子案例


记录一些常用的 CUDA 算子写法

&lt;!--more--&gt;


## Reduce

### Reduce Sum

```CUDA
template &lt;const int kWarpSize = 256&gt;
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize; mask &gt;= 1; mask &gt;&gt;= 1) {
        val &#43;= __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
```


### Block reduce sum
```CUDA
template &lt;const int NUM_THREADS = 256&gt;
__device__ float block_reduce_sum_f32(float val) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x &#43; tid;

    constexpr int NUM_WARPS = (NUM_THREADS &#43; WARP_SIZE - 1) / WARP_SIZE;

    static __shared__ reduce_sum[NUM_WARPS];

    float sum = warp_reduce_sum_f32&lt;WARP_SIZE&gt;(val);

    if (lane == 0)
        reduce_sum[warp] = sum;
    __syncthreads();

    sum = (lane &lt; NUM_WARPS) ? reduce_sum[lane] : 0.0f;
    sum = warp_reduce_sum_f32&lt;NUM_WARPS&gt;(sum);
    sum = __shfl_sync(0xffffffff, sum, 0, 32);
    return sum;
}
```

### Reduce Max

```CUDA
template &lt;const int kWarpSize = 256&gt;
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize; mask &gt;= 1; mask &gt;&gt;= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
```

### block reduce max
```CUDA
// grid 1D block 1D, grid(N/256), block(256)
template &lt;const int NUM_THREADS = 256&gt;
__device__ float block_reduce_max_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS &#43; WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    static __shared__ float reduce_max[NUM_WARPS];

    float value = warp_reduce_sum_f32&lt;WARP_SIZE&gt;(val);
    if (lane == 0) {
        shared[warp] = value;
    }   
    __syncthreads();

    value = (lane &lt; NUM_WARPS) ? shared[lane] : -FLT_MAX;
    value = warp_reduce_max_f32&lt;NUM_WARPS&gt;(value);

    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}
```

## Softmax

### naive版
```CUDA
template &lt;const int NUM_THREADS = 256&gt;
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x &#43; tid;

    float exp_val = (idx &lt; N) ? expf[x[idx]] : 0.0f;
    float exp_sum = block_reduce_sum_f32&lt;NUM_THREADS&gt;(exp_val);

    if (idx &lt; N) {
        y[idx] = exp_val / exp_sum;
    }
}
```

### 向量化存取

```CUDA
#define FLOAT4(value) (reinterpret_cast&lt;int4 *&gt;(&amp;(value))[0])
template &lt;const int NUM_THREADS = 256&gt;
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x &#43; tid) * 4;

    // 向量化 取
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_exp;
    reg_exp.x = (idx &#43; 0 &lt; N) ? expf(reg_x.x) : 0.0f;
    reg_exp.y = (idx &#43; 1 &lt; N) ? expf(reg_x.y) : 0.0f;
    reg_exp.z = (idx &#43; 2 &lt; N) ? expf(reg_x.z) : 0.0f;
    reg_exp.w = (idx &#43; 3 &lt; N) ? expf(reg_x.w) : 0.0f;

    float exp_val = (reg_exp.x &#43; reg_exp.y &#43; reg_exp.z &#43; reg_exp.w);
    float exp_sum = block_reduce_sum_f32&lt;NUM_THREADS&gt;(exp_val);

    // 向量化 存
    if (idx &#43; 3 &lt; N) {
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

template &lt;const int NUM_THREADS = 256&gt;
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x &#43; tid;

    float val = (idx &lt; N) ? x[idx] : -FLT_MAX;
    float max_val = block_reduce_max_f32&lt;NUM_THREADS&gt;(val);
    float exp_val = (idx &lt; N) ? expf[x[idx] - max_val] : 0.0f;
    float exp_sum = block_reduce_sum_f32&lt;NUM_THREADS&gt;(exp_val);

    if (idx &lt; N)
        y[idx] = exp_val / exp_sum;
}
```


### online-softmax

```CUDA
struct __align__(8) MD {
    float M;
    float D;
};

template &lt;const int NUM_THREADS = 256&gt;
__global__ void online_softmax_f32_per_token_kernel(float *x, float *y, int N) {
    //
}

__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x &#43; threadIdx.x;
    int m = blockIdx.y * blockDim.y &#43; threadIdx.y;

    if (m &lt; M &amp;&amp; n &lt; N) {
        float psum = 0.0f;
#pragma unroll
        for (int k = 0; k &lt; K; &#43;&#43; k) {
            psum &#43;= a[m * K &#43; k] * [k * N &#43; n];
        }
        c[m * M &#43; n] = psum;
    }
}

template &lt;const int BM = 32, const int BN = 32, const int BK = 32&gt;
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {
    // block tile ： 32x32 的 block 处理 c 上一块 32x32 的元素计算
    // K tile：      使用共享内存，将 K 分块成 BK 大小的块
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.y * blockDim.x &#43; tx;

    int load_smem_a_m = tid / 32;
    int load_smem_a_n = tid % 32;
    int load_smem_b_n = tid / 32;
    int load_smem_b_k = tid % 32;

    int load_gmem_a_m = by * BM &#43; load_smem_a_m;  // global row of a and c
    int load_gmem_b_n = bx * BN &#43; load_smem_b_n;  // global col of b and c

    if (load_gmem_a_m &gt;= M || load_gmem_b_n &gt;= N) return;

    float sum = 0.0f;
    for (int bk = 0; bk &lt; (K &#43; BK - 1) / BK; &#43;&#43; bk) {
        // 加载 a 的全局内容到共享内存
        int load_gmem_a_k = bk * BK &#43; load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K &#43; load_gmem_a_k;
        s_a[load_gmem_a_m][load_gmem_a_k] = a[load_gmem_a_addr];

        // 加载 b 的全局内容到共享内存
        int load_gmem_b_k = bk * BK &#43; load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N &#43; load_gmem_b_n;
        s_b[load_gmem_b_k][load_gmem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k &lt; BK; &#43;&#43; k) {
        // 共享内存内进行 block gemm
        sum &#43;= s_a[load_smem_a_m][k] * s_b[k][load_smem_b_n];
    }
    __syncthreads();

    // 存
    int store_gmem_c_addr = load_gmem_a_m * N &#43; load_gmem_b_n;
    c[store_gmem_c_addr] = sum;
}
```

## Matmul

### naive 版

```CUDA
__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x &#43; threadIdx.x;
    int m = blockIdx.y * blockDim.y &#43; threadIdx.y;

    if (m &lt; M &amp;&amp; n &lt; N) {
        float psum = 0.0f;
#pragma unroll
        for (int k = 0; k &lt; K; &#43;&#43; k) {
            psum &#43;= a[m * K &#43; k] * [k * N &#43; n];
        }
        c[m * M &#43; n] = psum;
    }
}
```

### shared_mem 优化

```CUDA
template &lt;const int BM = 32, const int BN = 32, const int BK = 32&gt;
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {
    // block tile ： 32x32 的 block 处理 c 上一块 32x32 的元素计算
    // K tile：      使用共享内存，将 K 分块成 BK 大小的块
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.y * blockDim.x &#43; tx;

    int load_smem_a_m = tid / 32;
    int load_smem_a_n = tid % 32;
    int load_smem_b_n = tid / 32;
    int load_smem_b_k = tid % 32;

    int load_gmem_a_m = by * BM &#43; load_smem_a_m;  // global row of a and c
    int load_gmem_b_n = bx * BN &#43; load_smem_b_n;  // global col of b and c

    if (load_gmem_a_m &gt;= M || load_gmem_b_n &gt;= N) return;

    float sum = 0.0f;
    for (int bk = 0; bk &lt; (K &#43; BK - 1) / BK; &#43;&#43; bk) {
        // 加载 a 的全局内容到共享内存
        int load_gmem_a_k = bk * BK &#43; load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K &#43; load_gmem_a_k;
        s_a[load_gmem_a_m][load_gmem_a_k] = a[load_gmem_a_addr];

        // 加载 b 的全局内容到共享内存
        int load_gmem_b_k = bk * BK &#43; load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N &#43; load_gmem_b_n;
        s_b[load_gmem_b_k][load_gmem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k &lt; BK; &#43;&#43; k) {
        // 共享内存内进行 block gemm
        sum &#43;= s_a[load_smem_a_m][k] * s_b[k][load_smem_b_n];
    }
    __syncthreads();

    // 存
    int store_gmem_c_addr = load_gmem_a_m * N &#43; load_gmem_b_n;
    c[store_gmem_c_addr] = sum;
}
```

## Transpose

### naive

```CUDA
__global__ void transpose_naive(float *input, float *output, int M, int N) {
    int n = blockIdx.x * blockDim.x &#43; threadIdx.x;
    int m = blockIdx.y * blockDim.y &#43; threadIdx.y;

    if (m &lt; M &amp;&amp; n &lt; N) {
        output[n * M &#43; m] = input[m * N &#43; n];
    }
}
```

### 合并写入

```CUDA
__global__ void transpose(float* input, float* output, int M, int N) {
    // output的row和col
    int row = blockDim.y * blockIdx.y &#43; threadIdx.y;
    int col = blockDim.x * blockIdx.x &#43; threadIdx.x;
​
    if (row &lt; N &amp;&amp; col &lt; M) {
        output[row * M &#43; col] = __ldg(&amp;input[col * N &#43; row]);  // 合并写入，读取使用__ldg进行缓存
    }
}
```

### 使用共享内存中转，同时合并读取和写入

```cuda
// 输入矩阵是M行N列，输出矩阵是N行M列
 dim3 block(32, 32);
 dim3 grid(CEIL(N,32), CEIL(M,32));  // 根据input的形状(M行N列)进行切块
 transpose&lt;32&gt;&lt;&lt;&lt;grid, block&gt;&gt;&gt;(input, output, M, N);
 ​
 template &lt;const int BLOCK_SIZE&gt;
 __global__ void transpose(float* input, float* output, int M, int N) {
     __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE &#43; 1];  // 避免bank conflict
     int bx = blockIdx.x * BLOCK_SIZE;
     int by = blockIdx.y * BLOCK_SIZE;
     int x1 = bx &#43; threadIdx.x;
     int y1 = by &#43; threadIdx.y;

     if (x1 &lt; N &amp;&amp; y1 &lt; M) {
         s_mem[threadIdx.y][threadIdx.x] = input[y1 * N &#43; x1];
     }
     __syncthreads();
 ​
     int x2 = by &#43; threadIdx.x;
     int y2 = bx &#43; threadIdx.y;
     if (x2 &lt; M &amp;&amp; y2 &lt; N) {
         output[y2 * M &#43; x2] = s_mem[threadIdx.x][threadIdx.y];  // padding后，不存在bank conflict
     }
 }
```

### cuda example 补充

&gt; https://zhuanlan.zhihu.com/p/12661298743

主要是 transpose 优化、gemv、softmax_matrix，cuda前缀和，topk

### Memory Coalescing 和 Bank Conflict


内存合并访问：

&gt; 参考文献：[https://zhuanlan.zhihu.com/p/300785893](https://zhuanlan.zhihu.com/p/300785893)

Bank Conflict：

&gt; 参考文献：

1. [https://zhuanlan.zhihu.com/p/4746910252](https://zhuanlan.zhihu.com/p/4746910252)
2. [https://zhuanlan.zhihu.com/p/659142274](https://zhuanlan.zhihu.com/p/659142274)


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/ebaa040/  

