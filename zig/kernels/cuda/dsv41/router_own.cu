// TF_DS_PROMPT_ROUTER=own (stage ZR1, router.zig): the MoE gate's logits of this node's own rows with the per-element
// arithmetic of today's cuBLAS run (cublasSgemm_v2 T,N on 2,048 rows: x fp32 [rows, K] times gate_w fp32 [E, K]
// transposed). That run splits K into partitions [b_i, b_{i+1}) (split-K, serial: each partition is one FFMA chain in
// ascending k from +0, and the partitions are added to the output in order: D = p0, then D = p_i + D). Each output
// element here runs the same chains and the same adds, in the same order, with explicit round-to-nearest intrinsics,
// so it gets today's bits whatever rows share the launch. The partitions are found at start (router.zig: calibrate,
// against the library's own call); this file knows only the rule.
//
// Y [rows, E] fp32 (row stride E), X [rows, K] (row stride ldx), W [E, K] (row stride K). Two input forms, the same
// arithmetic: fp32 X and W (tf_ds_router_own_*: what cuBLAS reads, x.float() and gate_w.float()), or X bf16 and W fp16
// as the engine holds them (tf_ds_router_own_h_*: converted to fp32 in the program, exactly as .float() does, so every
// operand of every FFMA is the same value and today's two cast launches are not needed). Needs E a multiple of the
// block's BN, K and ldx multiples of 4, X and W aligned to 4 elements.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// split-K partitions: n in [1, 16], b[0] = 0 < b[1] < ... < b[n] = K (router.zig Parts)
struct Parts {
  int n;
  int b[17];
};

// four consecutive k of a row as fp32: fp32 as is; bf16 and fp16 widened exactly (.float())
__device__ __forceinline__ float4 load4(const float *p) { return *reinterpret_cast<const float4 *>(p); }
__device__ __forceinline__ float4 load4(const __nv_bfloat16 *p) {
  const uint2 u = *reinterpret_cast<const uint2 *>(p);
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16),
                     __uint_as_float(u.y & 0xffff0000u));
}
__device__ __forceinline__ float4 load4(const __half *p) {
  const uint2 u = *reinterpret_cast<const uint2 *>(p);
  const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u.x));
  const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float load1(const float *p) { return *p; }
__device__ __forceinline__ float load1(const __nv_bfloat16 *p) { return __bfloat162float(*p); }
__device__ __forceinline__ float load1(const __half *p) { return __half2float(*p); }

template <int BM, int BN, int TM, int TN, int BK, typename TX, typename TW>
__device__ __forceinline__ void own_body(const TX *__restrict__ X, long long ldx, const TW *__restrict__ W,
                                         float *__restrict__ Y, int rows, int K, int E, const Parts &p) {
  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ __align__(16) float As[2][BK][BM + 4];
  __shared__ __align__(16) float Bs[2][BK][BN + 4];
  const int tid = threadIdx.x;
  const int tx = tid % (BN / TN), ty = tid / (BN / TN);
  const int r0 = blockIdx.y * BM, e0 = blockIdx.x * BN;
  float acc[TM][TN], tot[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0.0f, tot[i][j] = 0.0f;
  constexpr int AV = BM * BK / 4, BV = BN * BK / 4;
  auto load = [&](int buf, int k0) {
#pragma unroll
    for (int v = tid; v < AV; v += NT) {
      const int m = v / (BK / 4), kq = (v % (BK / 4)) * 4;
      const int r = r0 + m;
      float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < rows) f = load4(X + r * ldx + k0 + kq);
      As[buf][kq][m] = f.x;
      As[buf][kq + 1][m] = f.y;
      As[buf][kq + 2][m] = f.z;
      As[buf][kq + 3][m] = f.w;
    }
#pragma unroll
    for (int v = tid; v < BV; v += NT) {
      const int nn = v / (BK / 4), kq = (v % (BK / 4)) * 4;
      const float4 f = load4(W + (long long)(e0 + nn) * K + k0 + kq);
      Bs[buf][kq][nn] = f.x;
      Bs[buf][kq + 1][nn] = f.y;
      Bs[buf][kq + 2][nn] = f.z;
      Bs[buf][kq + 3][nn] = f.w;
    }
  };
  // the partition boundary ahead (pi: the partition being summed; next: its end)
  int pi = 0;
  int next = p.b[1];
  // the end of partition pi reached: D = p0 for the first, D = p_i + D after it
  auto close = [&]() {
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) {
        tot[i][j] = pi == 0 ? acc[i][j] : __fadd_rn(acc[i][j], tot[i][j]);
        acc[i][j] = 0.0f;
      }
    pi++;
    next = pi < p.n ? p.b[pi + 1] : 0x7fffffff;
  };
  load(0, 0);
  __syncthreads();
  int buf = 0;
  const int kt = K / BK * BK;
  for (int k0 = 0; k0 < kt; k0 += BK) {
    if (k0 + BK < kt) load(buf ^ 1, k0 + BK);
    if (k0 + BK <= next) {
      // a whole tile inside one partition: BK steps of the chain
#pragma unroll
      for (int kk = 0; kk < BK; kk++) {
        float a[TM], b[TN];
#pragma unroll
        for (int i = 0; i < TM; i++) a[i] = As[buf][kk][ty * TM + i];
#pragma unroll
        for (int j = 0; j < TN; j++) b[j] = Bs[buf][kk][tx * TN + j];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
      }
    } else {
      // a tile a boundary falls in (or ends at): step by step
      for (int kk = 0; kk < BK; kk++) {
        float a[TM], b[TN];
#pragma unroll
        for (int i = 0; i < TM; i++) a[i] = As[buf][kk][ty * TM + i];
#pragma unroll
        for (int j = 0; j < TN; j++) b[j] = Bs[buf][kk][tx * TN + j];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
        if (k0 + kk + 1 == next) close();
      }
    }
    if (k0 + BK == next) close();
    __syncthreads();
    buf ^= 1;
  }
  // K past whole tiles (K % BK): straight from global memory
  for (int k = kt; k < K; k++) {
    float a[TM], b[TN];
#pragma unroll
    for (int i = 0; i < TM; i++) {
      const int r = r0 + ty * TM + i;
      a[i] = r < rows ? load1(X + r * ldx + k) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < TN; j++) b[j] = load1(W + (long long)(e0 + tx * TN + j) * K + k);
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
    if (k + 1 == next) close();
  }
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int r = r0 + ty * TM + i;
    if (r < rows) {
      float *y = Y + (long long)r * E + e0 + tx * TN;
#pragma unroll
      for (int j = 0; j < TN; j++) y[j] = tot[i][j];
    }
  }
}

}  // namespace

// the variants differ only in how the rows and experts are tiled (every element's chains are the same)
#define TF_DS_ROUTER_OWN(NAME, BM, BN, TM, TN, BK, TX, TW)                                                         \
  extern "C" __global__ void __launch_bounds__((BM / TM) * (BN / TN))                                             \
      NAME(const TX *__restrict__ X, long long ldx, const TW *__restrict__ W, float *__restrict__ Y, int rows, int K, \
           int E, Parts p) {                                                                                      \
    own_body<BM, BN, TM, TN, BK, TX, TW>(X, ldx, W, Y, rows, K, E, p);                                            \
  }

// fp32 operands (the probe; cuBLAS's inputs)
TF_DS_ROUTER_OWN(tf_ds_router_own_64x64, 64, 64, 4, 4, 16, float, float)
TF_DS_ROUTER_OWN(tf_ds_router_own_32x64, 32, 64, 2, 4, 16, float, float)
TF_DS_ROUTER_OWN(tf_ds_router_own_128x64, 128, 64, 8, 4, 8, float, float)
TF_DS_ROUTER_OWN(tf_ds_router_own_64x128, 64, 128, 4, 8, 8, float, float)
// x bf16, gate_w fp16 (the engine's)
TF_DS_ROUTER_OWN(tf_ds_router_own_h_64x64, 64, 64, 4, 4, 16, __nv_bfloat16, __half)
TF_DS_ROUTER_OWN(tf_ds_router_own_h_32x64, 32, 64, 2, 4, 16, __nv_bfloat16, __half)
TF_DS_ROUTER_OWN(tf_ds_router_own_h_128x64, 128, 64, 8, 4, 8, __nv_bfloat16, __half)
TF_DS_ROUTER_OWN(tf_ds_router_own_h_64x128, 64, 128, 4, 8, 8, __nv_bfloat16, __half)
