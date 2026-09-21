// Smallest probe for the sm_121a vs sm_120f question behind
// VLLM_PRESERVE_SM12X_TARGET: identical source, one binary per -arch, same GPU.
//
// The kernels are plain CUDA with no arch-accelerated intrinsics on purpose, so
// the only variable is the target the compiler picks. They are shaped after the
// ops vLLM compiles into its own extension rather than after vendor libraries:
// a streaming copy, a row reduction (RMSNorm), a per-group fp8 quantize, and a
// shared-memory tiled GEMM for the compute-bound end.
//
// Usage: ./probe_<arch>      (prints one line per kernel)
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define CK(x)                                                             \
    do {                                                                  \
        cudaError_t e_ = (x);                                             \
        if (e_ != cudaSuccess) {                                          \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e_),  \
                   __LINE__);                                             \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------- kernels

__global__ void copy_f4(const float4* __restrict__ in, float4* __restrict__ out,
                        long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = in[i];
}

// One block per row. 512-wide rows, blockDim 256.
__global__ void rmsnorm(const float* __restrict__ in, const float* __restrict__ w,
                        float* __restrict__ out, int rows, float eps) {
    constexpr int W = 512;
    __shared__ float s[256];
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* x = in + (long)r * W;
    float acc = 0.f;
    for (int i = threadIdx.x; i < W; i += blockDim.x) acc += x[i] * x[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    float scale = rsqrtf(s[0] / W + eps);
    for (int i = threadIdx.x; i < W; i += blockDim.x)
        out[(long)r * W + i] = x[i] * scale * w[i];
}

// One block per 128-element group: per-group absmax fp8 e4m3 quantize.
__global__ void quant_fp8(const float* __restrict__ in,
                          __nv_fp8_e4m3* __restrict__ out,
                          float* __restrict__ scale, int gsize) {
    __shared__ float s[256];
    int g = blockIdx.x;
    const float* x = in + (long)g * gsize;
    float m = 0.f;
    for (int i = threadIdx.x; i < gsize; i += blockDim.x) m = fmaxf(m, fabsf(x[i]));
    s[threadIdx.x] = m;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + off]);
        __syncthreads();
    }
    float sc = s[0] / 448.f;
    if (threadIdx.x == 0) scale[g] = sc;
    sc = (sc == 0.f) ? 1.f : sc;
    for (int i = threadIdx.x; i < gsize; i += blockDim.x)
        out[(long)g * gsize + i] = __nv_fp8_e4m3(x[i] / sc);
}

// 16x16 tiled fp32 GEMM, C[MxN] = A[MxK] * B[KxN], row-major.
__global__ void gemm_f32(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
    constexpr int T = 16;
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < K; t += T) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------- harness

namespace {

struct Timing {
    float min;
    float med;
};

// Each sample is `inner` launches, so launch overhead does not dominate the
// 0.1 ms kernels.
Timing timeit(void (*launch)(void*), void* ctx, int reps, int inner) {
    for (int i = 0; i < 3; ++i) launch(ctx);  // warmup
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> ts;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a));
        for (int j = 0; j < inner; ++j) launch(ctx);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        ts.push_back(ms / inner);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    std::sort(ts.begin(), ts.end());
    return {ts.front(), ts[ts.size() / 2]};
}

struct CopyCtx { float4 *in, *out; long n; };
void copy_launch(void* p) {
    auto* c = static_cast<CopyCtx*>(p);
    copy_f4<<<2048, 256>>>(c->in, c->out, c->n);
}
struct NormCtx { float *in, *w, *out; int rows; };
void norm_launch(void* p) {
    auto* c = static_cast<NormCtx*>(p);
    rmsnorm<<<c->rows, 256>>>(c->in, c->w, c->out, c->rows, 1e-6f);
}
struct QuantCtx { float* in; __nv_fp8_e4m3* out; float* scale; int groups, gsize; };
void quant_launch(void* p) {
    auto* c = static_cast<QuantCtx*>(p);
    quant_fp8<<<c->groups, 256>>>(c->in, c->out, c->scale, c->gsize);
}
struct GemmCtx { float *A, *B, *C; int M, N, K; };
void gemm_launch(void* p) {
    auto* c = static_cast<GemmCtx*>(p);
    dim3 blk(16, 16), grd(c->N / 16, c->M / 16);
    gemm_f32<<<grd, blk>>>(c->A, c->B, c->C, c->M, c->N, c->K);
}

}  // namespace

int main() {
    const int reps = 20;
    printf("%-10s %9s %9s %12s\n", "kernel", "min_ms", "med_ms", "rate");

    // copy: 256 MiB each way
    {
        long n = (256L << 20) / 16;
        float4 *in, *out;
        CK(cudaMalloc(&in, n * 16));
        CK(cudaMalloc(&out, n * 16));
        CK(cudaMemset(in, 1, n * 16));
        CopyCtx c{in, out, n};
        auto t = timeit(copy_launch, &c, reps, 5);
        double gb = 2.0 * n * 16 / 1e9;
        printf("%-10s %8.3f %8.3f %9.1f GB/s\n", "copy", t.min, t.med, gb / (t.med / 1e3));
        CK(cudaFree(in));
        CK(cudaFree(out));
    }

    // rmsnorm: 8192 rows x 512
    {
        int rows = 8192;
        float *in, *w, *out;
        CK(cudaMalloc(&in, (size_t)rows * 512 * 4));
        CK(cudaMalloc(&w, 512 * 4));
        CK(cudaMalloc(&out, (size_t)rows * 512 * 4));
        CK(cudaMemset(in, 1, (size_t)rows * 512 * 4));
        CK(cudaMemset(w, 1, 512 * 4));
        NormCtx c{in, w, out, rows};
        auto t = timeit(norm_launch, &c, reps, 50);
        double gb = 3.0 * rows * 512 * 4 / 1e9;  // read + write + weight
        printf("%-10s %8.3f %8.3f %9.1f GB/s\n", "rmsnorm", t.min, t.med, gb / (t.med / 1e3));
        CK(cudaFree(in));
        CK(cudaFree(w));
        CK(cudaFree(out));
    }

    // quant_fp8: 65536 groups x 128
    {
        int gsize = 128, groups = 65536;
        size_t n = (size_t)groups * gsize;
        float* in;
        __nv_fp8_e4m3* out;
        float* scale;
        CK(cudaMalloc(&in, n * 4));
        CK(cudaMalloc(&out, n));
        CK(cudaMalloc(&scale, groups * 4));
        CK(cudaMemset(in, 1, n * 4));
        QuantCtx c{in, out, scale, groups, gsize};
        auto t = timeit(quant_launch, &c, reps, 20);
        double gb = (n * 4 + n + groups * 4) / 1e9;
        printf("%-10s %8.3f %8.3f %9.1f GB/s\n", "quant_fp8", t.min, t.med, gb / (t.med / 1e3));
        CK(cudaFree(in));
        CK(cudaFree(out));
        CK(cudaFree(scale));
    }

    // gemm: 1024^3
    {
        int M = 1024, N = 1024, K = 1024;
        size_t an = (size_t)M * K, bn = (size_t)K * N, cn = (size_t)M * N;
        float *A, *B, *C;
        CK(cudaMalloc(&A, an * 4));
        CK(cudaMalloc(&B, bn * 4));
        CK(cudaMalloc(&C, cn * 4));
        CK(cudaMemset(A, 1, an * 4));
        CK(cudaMemset(B, 1, bn * 4));
        GemmCtx c{A, B, C, M, N, K};
        auto t = timeit(gemm_launch, &c, reps, 5);
        double tf = 2.0 * M * N * K / 1e12;
        printf("%-10s %8.3f %8.3f %9.3f TFLOP/s\n", "gemm_f32", t.min, t.med, tf / (t.med / 1e3));
        CK(cudaFree(A));
        CK(cudaFree(B));
        CK(cudaFree(C));
    }
    return 0;
}
