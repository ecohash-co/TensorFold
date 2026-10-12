// TF_DS_PREFILL_MXFP4 (stage MX3, milestone 1: kernels + bench only; nothing in the layer loop calls them yet): a 2D
// node's prompt-chunk routed experts on MXFP4 weights in TensorFold's 136-word "mx" blocks (mx_prompt.zig; slices from
// convert_mxfp4_slices.py), the shared expert on exact bf16 cuts through cuBLAS, and a plain slot combine.
//
// The GEMM is TensorFold's O2 prefill kernel (src/tensorfold/cuda/nvfp4/experts_prefill.cu, md5 a102af3f..., the
// Python fork's tensorfold_nvfp4_experts_prefill_v2) with torch stripped: the device code below between the "upstream"
// markers is that file's lines 20-265 as they are, except (1) FMT 0 (144-word nvfp4: e4m3f, Fmt<0>, block_scales<0>)
// is dropped, only FMT 1 ("mx") is built, and (2) the namespace is tf_mx, not anonymous (stable kernel symbols).
// Per (pair, column), in K order: p = mma.m16n8k16 bf16 (A = the pair's bf16 x row, B = e2m1 decoded
// in registers to bf16) on a zero accumulator over one 16-input half, acc = fmaf(p, 2^(b - 9), acc); then acc *= the
// (expert, matrix) fp32 scale; then experts.cuh's epilogue (EPI 2: bf16 SwiGLU of M = 2; EPI 0: fp32). The same flags
// as the Python build (-O3 and torch's), so the same SASS arithmetic.
//
// Entries (mx_prompt.zig loads them by name and reads tf_mx_entry for each one's launch shape):
//   tf_mx::nvfp4_prefill_kernel<1, 2, 2, geo>  FMT 1, M 2, EPI 2 (gate / up -> bf16 SwiGLU)  upstream geometries 2, 3, 4
//   tf_mx::nvfp4_prefill_kernel<1, 1, 0, geo>  FMT 1, M 1, EPI 0 (down -> fp32 a pair)        the same three
//   (explicit instantiations of upstream's template, by mangled name), and extern "C":
//   tf_mx_plan_small / tf_mx_plan_rank / tf_mx_plan_offsets / tf_mx_plan_scatter: the grouping (tensorfold
//       experts.cu's plan kernels, grouped.route: pairs by expert in pair order, items of <= 64 pairs)
//   tf_mx_combine       out[r] = sum over slots s (in order, fmaf chain from 0) of w[r, s] * y[r * slots + s]
//   tf_mx_shared_swiglu the shared expert's gate / up GEMM output (fp32) -> experts.cuh's swiglu -> bf16 into the
//                       pack's slot rows (main columns, and the rest columns where this pair computes them)
#include <cuda_bf16.h>
#include <stdint.h>

#include "../experts.cuh"

// ==== upstream: src/tensorfold/cuda/nvfp4/experts_prefill.cu (FMT 1 only; see above) ================================
namespace tf_mx {
// ---- copied from nvfp4/experts.cu:26-36 (fp4pair; e4m3f is FMT 0's, dropped); keep the two in step ----------------------------------
// bf16x2 of the e2m1 nibbles at bits [s, s + 4) and [16 + s, 20 + s): fields into bf16's, times 2^126 (exact).
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
  const uint32_t v = w >> s;
  const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
  return r;
}
// ---- end of the copy ----------------------------------------------------------------------------------------------

// ---- copied from nvfp4/experts.cu:69-70 (mx_scale); keep the two in step -----------------------------------------
// an mx scale byte b (0..17) -> 2^(b - 9), a normal fp32 built from its exponent field
__device__ __forceinline__ float mx_scale(uint32_t b) { return __uint_as_float((b + (127u - 9u)) << 23); }
// ---- end of the copy ----------------------------------------------------------------------------------------------

// A (32 columns, 32 inputs) block format (nvfp4/experts.py:3-9). Both start with 128 code words (lane gq * 4 + t
// holds n8 tiles 0-3 of column gq).
//   FMT 0 nvfp4: 144 int32 words; then 16 words of e4m3 scale bytes [t][h][tile][column pair].
//   FMT 1 mx:    136 int32 words; then 8 words of one byte b per (column, 32 inputs), byte 8t + 2 tile + c.
template <int FMT>
struct Fmt;

template <>
struct Fmt<1> {
  static constexpr int WORDS = 136;
  static constexpr int U4 = WORDS / 4;     // 34 uint4 a block: every block still 16-byte aligned
};

// The one place a block's scales are read: the block scales of this lane's columns (2t, 2t + 1 of each n8 tile) for
// each 16-input half h, as fp32. nvfp4: the e4m3 bytes, as nvfp4/experts.cu:141-143 reads them.
template <int FMT>
__device__ __forceinline__ void block_scales(const uint4* blk, int t, float (&s)[2][NTW][2]);

// (block_scales<0>, FMT 0's e4m3 reader, dropped)

// mx: one byte per column and 32 inputs serves both 16-input halves, as nvfp4/experts.cu:136-139 reads it: the
// quad's uint2 (words 128 + 2t, 129 + 2t): tiles 0-1 in .x, 2-3 in .y; tile j's columns 2t and 2t + 1 at bits
// 16 (j & 1) and 16 (j & 1) + 8.
template <>
__device__ __forceinline__ void block_scales<1>(const uint4* blk, int t, float (&s)[2][NTW][2]) {
  const uint2 q = reinterpret_cast<const uint2*>(blk + 32)[t];
#pragma unroll
  for (int j = 0; j < NTW; ++j) {
    const uint32_t sw = (j >> 1) ? q.y : q.x;
    const int sh = (j & 1) * 16;
    const float s0 = mx_scale((sw >> sh) & 0xFFu), s1 = mx_scale((sw >> (sh + 8)) & 0xFFu);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      s[h][j][0] = s0;
      s[h][j][1] = s1;
    }
  }
}

// p = A x B on a zero accumulator (the decode-form kernel's ``float p[4] = {0}; mma(p, ...)``)
__device__ __forceinline__ void mma0(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}

template <int FMT, int M, int RT, int WM, int WN, int CN, int SG, int ST>
struct Tile {
  static_assert(SG >= 2, "the X swizzle needs at least 8 chunks a row");
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, CB = WN * CN;   // CB: column blocks a CTA
  static constexpr int XC = SG * 4;                       // uint4 a row a stage: SG groups of 32 bf16 inputs
  static constexpr int WB = M * Fmt<FMT>::U4;             // uint4 a (column block, group): M matrices' blocks
  static constexpr int XU = BM * XC, WU = CB * SG * WB, SU = XU + WU;
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int SMEM = ST * SU * 16;
  // chunk c of row r: pairs of chunks move by row so a half-warp's 4 rows read 4 distinct 32-byte bank windows
  static __device__ __forceinline__ int xslot(int r, int c) { return r * XC + (c ^ ((r & 3) << 1)); }
};

// One CTA: one item (expert e, pairs members[first .. first + cnt), cnt <= BM) by WN * CN column blocks of 32.
// Warp (wm, wn) holds pairs [16 RT wm, 16 RT (wm + 1)) of the item and column blocks cbt * WN * CN + wn * CN + c.
template <int FMT, int M, int EPI, int RT, int WM, int WN, int CN, int SG, int ST>
__global__ void __launch_bounds__(WM * WN * 32)
    nvfp4_prefill_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                         const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                         const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out,
                         int N, float limit, int skip) {
  using P = Tile<FMT, M, RT, WM, WN, CN, SG, ST>;
  extern __shared__ uint4 sm[];
  // an item's column tiles side by side: they share its X rows in L2 (a column tile's items side by side, sharing
  // an expert's blocks instead, measured 1.1-1.9x slower at 2048-4096 rows)
  const int nbt = (NB + P::CB - 1) / P::CB, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 3, gq = lane >> 2, cb0 = cbt * P::CB + wn * CN, cbs = min(P::CB, NB - cbt * P::CB);
  const __nv_bfloat16* xsrc[P::XPT];
  int xdst[P::XPT];
#pragma unroll
  for (int i = 0; i < P::XPT; ++i) {
    const int q = tid + i * P::THREADS, r = q / P::XC, c = q - r * P::XC;
    xdst[i] = P::xslot(r, c);
    xsrc[i] = nullptr;
    if (q < P::XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = X + (size_t)(slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  const uint4* wsrc = W + ((size_t)e * NB + cbt * P::CB) * (size_t)KG * P::WB;
  auto stage = [&](int s, int g0) {                  // groups g0 .. g0 + ng - 1 (ng < SG only at the end of K)
    const int ng = min(SG, KG - g0);
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i] && ((tid + i * P::THREADS) % P::XC) < ng * 4) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * 32);
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * SG * P::WB; q += P::THREADS) {
      const int j = q / (SG * P::WB), o = q - j * SG * P::WB;
      if (o < ng * P::WB) cp16(ws + q, wsrc + ((size_t)j * KG + g0) * P::WB + o);
    }
  };
  const int base = 16 * RT * wm;
  const int nt = max(0, min(RT, (cnt - base + 15) >> 4));
  const int cn = max(0, min(CN, NB - cb0));         // this warp's column blocks (fewer only at the end of N)
  const bool active = nt > 0 && cn > 0;
  float acc[CN][M][RT][NTW][4];
#pragma unroll
  for (int c = 0; c < CN; ++c)
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
          acc[c][m][r][j][0] = acc[c][m][r][j][1] = acc[c][m][r][j][2] = acc[c][m][r][j][3] = 0.f;
  const int steps = (KG + SG - 1) / SG;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < steps) stage(s, s * SG);
    cp_commit();
  }
  for (int k = 0; k < steps; ++k) {
    cp_wait<ST - 2>();
    __syncthreads();
    if (k + ST - 1 < steps) stage((k + ST - 1) % ST, (k + ST - 1) * SG);
    cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k % ST) * P::SU;
    const uint2* x2 = reinterpret_cast<const uint2*>(xs);
#pragma unroll
    for (int gi = 0; gi < SG; ++gi) {
      if (k * SG + gi >= KG) break;
      // A: rows gq and gq + 8 of each row tile, inputs 16h + 4t .. 16h + 4t + 3 (nvfp4/experts.cu:55-59, 69)
      uint32_t a[RT][2][4];
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= nt) break;
        const int r0 = base + 16 * r + gq;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int c = gi * 4 + 2 * h + (t >> 1);
          const uint2 lo = x2[2 * P::xslot(r0, c) + (t & 1)], hi = x2[2 * P::xslot(r0 + 8, c) + (t & 1)];
          a[r][h][0] = lo.x;
          a[r][h][2] = lo.y;
          a[r][h][1] = hi.x;
          a[r][h][3] = hi.y;
        }
      }
#pragma unroll
      for (int cm = 0; cm < CN * M; ++cm) {          // column block c, matrix m: each B fragment feeds RT tiles
        const int c = cm / M, m = cm - c * M;
        if (c >= cn) break;
        const uint4* blk = xs + P::XU + ((wn * CN + c) * SG + gi) * P::WB + m * Fmt<FMT>::U4;
        const uint4 wv = blk[lane];
        float s[2][NTW][2];
        block_scales<FMT>(blk, t, s);
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          const uint32_t word = comp(wv, j);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t b0 = fp4pair(word, 8 * h), b1 = fp4pair(word, 8 * h + 4);
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              if (r >= nt) break;
              float p[4];
              mma0(p, a[r][h][0], a[r][h][1], a[r][h][2], a[r][h][3], b0, b1);
              float(&c4)[4] = acc[c][m][r][j];
              c4[0] = fmaf(p[0], s[h][j][0], c4[0]);
              c4[1] = fmaf(p[1], s[h][j][1], c4[1]);
              c4[2] = fmaf(p[2], s[h][j][0], c4[2]);
              c4[3] = fmaf(p[3], s[h][j][1], c4[3]);
            }
          }
        }
      }
    }
  }
  cp_wait<0>();
  if (!active) return;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float g = __ldg(scale + e * M + m);
#pragma unroll
    for (int c = 0; c < CN; ++c)
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[c][m][r][j][q] *= g;
  }
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    if (r >= nt) break;
    const int m0 = base + 16 * r + gq, m1 = m0 + 8;
    const bool v0 = m0 < cnt, v1 = m1 < cnt;
    const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
#pragma unroll
    for (int c = 0; c < CN; ++c)
      if (c < cn) epilogue<EPI, M, RT>(acc[c], r, out, N, (cb0 + c) * COLS + 2 * t, p0, p1, v0, v1, limit);
  }
}

}  // namespace tf_mx
// ==== end of upstream ===============================================================================================

// ==== the entries (upstream's TF_GEOS ids 2, 3, 4: (RT, WM, WN, CN, SG, stages)) ====================================
// geometry 2: gate / up's default at NB 16 (experts_prefill.choose), 3: six column-block warps (when 6 divides NB),
// 4: down's default. Every geometry gives the same bits (a pair's bits never depend on the tiling).
#define TF_MX_GEOS(X)          \
  X(2, 2, 2, 4, 1, 2, 2)       \
  X(3, 2, 2, 6, 1, 2, 2)       \
  X(4, 2, 2, 4, 2, 2, 2)

// The kernel is upstream's __global__ template itself (only its namespace is named, tf_mx, not anonymous, so its
// symbols are stable): explicit instantiations, loaded by their mangled names (mx_prompt.zig entry_symbols). A wrapper
// entry inlining it changed ptxas's register allocation (gu geometry 3: 168 registers and a stack frame against
// upstream's 138); the template instantiated as it is compiles to upstream's SASS (tools: sass-diff in the commit).
#define TF_MX_ENTRY(ID, RT, WM, WN, CN, SG, ST)                                                                     \
  template __global__ void tf_mx::nvfp4_prefill_kernel<1, 2, 2, RT, WM, WN, CN, SG, ST>(                            \
      const __nv_bfloat16*, int, int, const uint4*, const float*, int, int, const int*, const int*, const int*, void*, \
      int, float, int);                                                                                              \
  template __global__ void tf_mx::nvfp4_prefill_kernel<1, 1, 0, RT, WM, WN, CN, SG, ST>(                            \
      const __nv_bfloat16*, int, int, const uint4*, const float*, int, int, const int*, const int*, const int*, void*, \
      int, float, int);
TF_MX_GEOS(TF_MX_ENTRY)
#undef TF_MX_ENTRY

// Each entry's launch shape, in the order gu_g2, gu_g3, gu_g4, dn_g2, dn_g3, dn_g4: {threads, dynamic shared bytes,
// column blocks a CTA (CB), pairs an item holds (16 RT WM)}. A launch is max_items x ceil(NB / CB) CTAs (upstream's
// launch()); every entry opts into its shared memory (gu_g4's 51,200 B is over the 48 KiB default).
#define TF_MX_SHAPE(M_, RT, WM, WN, CN, SG, ST)                                                                    \
  {tf_mx::Tile<1, M_, RT, WM, WN, CN, SG, ST>::THREADS, tf_mx::Tile<1, M_, RT, WM, WN, CN, SG, ST>::SMEM,                       \
   tf_mx::Tile<1, M_, RT, WM, WN, CN, SG, ST>::CB, tf_mx::Tile<1, M_, RT, WM, WN, CN, SG, ST>::BM}
#define TF_MX_GU(ID, RT, WM, WN, CN, SG, ST) TF_MX_SHAPE(2, RT, WM, WN, CN, SG, ST),
#define TF_MX_DN(ID, RT, WM, WN, CN, SG, ST) TF_MX_SHAPE(1, RT, WM, WN, CN, SG, ST),
extern "C" __device__ int tf_mx_entry[6][4] = {TF_MX_GEOS(TF_MX_GU) TF_MX_GEOS(TF_MX_DN)};
#undef TF_MX_GU
#undef TF_MX_DN
#undef TF_MX_SHAPE
// The 136-word mx block and the item size every entry was built for (mx_prompt.zig refuses another image).
extern "C" __device__ int tf_mx_format[2] = {tf_mx::Fmt<1>::WORDS, 64};

// ==== the grouping: tensorfold src/tensorfold/cuda/experts.cu's plan section (experts_plan_cuda, grouped.route) ======
// copied as it is (the same code as zig/kernels/cuda/experts.cu's tf_experts plan kernels); keep in step. Pairs
// p = row * slots + slot grouped by expert in pair order; items (expert, first member, count) of <= T pairs in expert
// order; counts [items, distinct experts]. One block for <= 1024 pairs (tf_mx_plan_small), else rank / offsets /
// scatter with rank [P] and hist [ceil(P / 1024), E] scratch.
namespace tf_mx_plan {

constexpr int PLAN_THREADS = 1024;
constexpr int EMAX = 1024;
constexpr int PSMALL = 1024;       // pairs the one-block plan takes; wider calls rank in blocks of PLAN_THREADS

// Whole block: thread e gives expert e's pair count, gets its first member; items of <= T pairs land in expert order.
__device__ int place_items(int c, int E, int T, int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int wsum[3][32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tiles = (c + T - 1) / T;
  int a = c, b = tiles, u = c > 0;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int ua = __shfl_up_sync(0xffffffffu, a, o), ub = __shfl_up_sync(0xffffffffu, b, o);
    const int uu = __shfl_up_sync(0xffffffffu, u, o);
    if (lane >= o) {
      a += ua;
      b += ub;
      u += uu;
    }
  }
  if (lane == 31) {
    wsum[0][warp] = a;
    wsum[1][warp] = b;
    wsum[2][warp] = u;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      int v = wsum[s][lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int w = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += w;
      }
      wsum[s][lane] = v;
    }
  }
  __syncthreads();
  const int off = (warp ? wsum[0][warp - 1] : 0) + a - c;
  const int ioff = (warp ? wsum[1][warp - 1] : 0) + b - tiles;
  if (tid == 0) {
    counts[0] = wsum[1][31];
    counts[1] = wsum[2][31];
  }
  if (tid < E)
    for (int j = 0; j < tiles; ++j) {
      int* it = items + 3 * (ioff + j);
      it[0] = tid;
      it[1] = off + T * j;
      it[2] = min(T, c - T * j);
    }
  return off;
}

}  // namespace tf_mx_plan

using tf_mx_plan::EMAX;
using tf_mx_plan::PLAN_THREADS;
using tf_mx_plan::PSMALL;

// One block: pairs p = row * slots + slot grouped by expert (pair order within an expert).
extern "C" __global__ void __launch_bounds__(PLAN_THREADS)
    tf_mx_plan_small(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
                     int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int pk[PSMALL];
  __shared__ int cnt[EMAX];
  __shared__ int off[EMAX];
  const int tid = threadIdx.x;
  for (int e = tid; e < EMAX; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = picks[p];
    pk[p] = e;
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int o = tf_mx_plan::place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
  if (tid < E) off[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = pk[p];
    int rank = 0;
    for (int q = 0; q < p; ++q) rank += pk[q] == e;
    members[off[e] + rank] = p;
  }
}

// Wide calls, pass 1: block b ranks its pairs within each expert in pair order and writes its counts to hist[b][e].
extern "C" __global__ void __launch_bounds__(PLAN_THREADS)
    tf_mx_plan_rank(const int* __restrict__ picks, int P, int E, int* __restrict__ rank, int* __restrict__ hist) {
  __shared__ int cnt[EMAX];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  const int p = blockIdx.x * PLAN_THREADS + tid;
  const bool ok = p < P;
  const int e = ok ? picks[p] : -1 - lane;
  const unsigned same = __match_any_sync(0xffffffffu, e);
  const int below = __popc(same & ((1u << lane) - 1u));
  for (int w = 0; w < PLAN_THREADS / 32; ++w) {
    if (warp == w) {
      const int base = ok ? cnt[e] : 0;
      __syncwarp();
      if (ok && below == 0) cnt[e] = base + __popc(same);
      if (ok) rank[p] = base + below;
    }
    __syncthreads();
  }
  for (int x = tid; x < E; x += PLAN_THREADS) hist[(size_t)blockIdx.x * E + x] = cnt[x];
}

// Pass 2, one block: each block's first member for each expert in place of its count, and the items.
extern "C" __global__ void __launch_bounds__(PLAN_THREADS)
    tf_mx_plan_offsets(int nblk, int E, int T, int* __restrict__ hist, int* __restrict__ items,
                       int* __restrict__ counts) {
  const int tid = threadIdx.x;
  int c = 0;
  if (tid < E)
    for (int b = 0; b < nblk; ++b) {
      int* h = hist + (size_t)b * E + tid;
      const int v = *h;
      *h = c;
      c += v;
    }
  const int off = tf_mx_plan::place_items(c, E, T, items, counts);
  if (tid < E)
    for (int b = 0; b < nblk; ++b) hist[(size_t)b * E + tid] += off;
}

// Pass 3: every pair to its place.
extern "C" __global__ void tf_mx_plan_scatter(const int* __restrict__ picks, int P, int E, const int* __restrict__ rank,
                                              const int* __restrict__ hist, int* __restrict__ members) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < P) members[hist[(size_t)(p / PLAN_THREADS) * E + picks[p]] + rank[p]] = p;
}

// ==== the slot combine ===============================================================================================
// out[r, :] = sum_{s = 0 .. slots - 1} w[r, s] * y[r * slots + s, :]: fixed order s = 0 .. slots - 1, an fp32 fmaf
// chain from 0, float4 a thread (bench_mx_zigsplit.py's mx_combine_k with the slot count a parameter: 6 routed + the
// shared expert's slot = 7 here). out may have a row stride (ld4 float4s) wider than d4.
extern "C" __global__ void tf_mx_combine(const float4* __restrict__ y, const float* __restrict__ w,
                                         float4* __restrict__ out, long long n4, int d4, int slots, long long ld4) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const long long r = i / d4;
  const int c = (int)(i - r * d4);
  const float* wr = w + r * slots;
  const float4* yr = y + r * slots * (long long)d4 + c;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 7
  for (int s = 0; s < slots; ++s) {
    const float4 v = __ldcs(yr + (long long)s * d4);
    const float ws = wr[s];
    a.x = fmaf(ws, v.x, a.x);
    a.y = fmaf(ws, v.y, a.y);
    a.z = fmaf(ws, v.z, a.z);
    a.w = fmaf(ws, v.w, a.w);
  }
  out[r * ld4 + c] = a;
}

// ==== the shared expert's SwiGLU =====================================================================================
// g fp32 [n, 2 sw] (the cuBLAS GEMM of x against [gate rows; up rows]: columns [0, sw) gate, [sw, 2 sw) up) ->
// h = experts.cuh's swiglu(gate, up, limit) (the routed experts' epilogue: bf16(bf16(silu(bf16 g)) * bf16 u), clipped)
// rounded to bf16, into the pack: column j < main_w at xm[(r * slots + slot) * main_w + j], j >= main_w at
// xr[(r * slots + slot) * rest_w + j - main_w] (sw = main_w, or main_w + rest_w on the pair that computes the shared
// expert's rest). One block a row.
extern "C" __global__ void tf_mx_shared_swiglu(const float* __restrict__ g, int sw, int slots, int slot,
                                               __nv_bfloat16* __restrict__ xm, int main_w,
                                               __nv_bfloat16* __restrict__ xr, int rest_w, float limit) {
  const long long r = blockIdx.x;
  const float* gr = g + r * 2 * (long long)sw;
  const long long row = r * slots + slot;
  for (int j = threadIdx.x; j < sw; j += blockDim.x) {
    const __nv_bfloat16 h = __float2bfloat16_rn(swiglu(gr[j], gr[sw + j], limit));
    if (j < main_w)
      xm[row * main_w + j] = h;
    else
      xr[row * rest_w + (j - main_w)] = h;
  }
}
