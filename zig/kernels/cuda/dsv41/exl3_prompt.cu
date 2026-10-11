// TF_DS_EXL3_PROMPT=v2: a 2D node's prompt-chunk EXL3 expert GEMMs (exl3_prompt.zig). The device code computes every
// output element with the served build's arithmetic (TensorFold's exl3 experts_grouped.cuh grouped_mma_kernel for gate /
// up and grouped_rows_kernel for down, and experts.cu's rot_in_kernel for gate / up's input rows; after ExLlamaV3, MIT,
// Copyright (c) 2025 Turboderp): the same trellis decode (decode_tile, copied as it is), the same fp16 A rows, each K
// chain an fp32 mma.m16n8k16 chain from zero over the same k tiles in the same order, the chains of a split added in
// order and gate / up's splits added in order from 0. Only the schedule differs. A program is (expert place, 32 member
// rows, 256 output columns (16 warps) or 128 (8 warps), matrix); each warp owns one n16 column tile for all 32 rows, so
// each 16 x 16 weight tile is decoded once a program straight into the mma B registers. K goes in steps of one
// 128-column Hadamard block through a three-stage shared-memory ring filled by cp.async two steps ahead (each warp
// copying its own tiles' trellis words and its own rows, so a step needs one block barrier):
//   - gate / up: the rows are x's (bf16), rotated in place in the program (rot_in_kernel's arithmetic, its contraction
//     pinned by intrinsics), so the fp16 xg / xu rows (2 x 14,336 x 5,120 x 2 bytes at 2,048 rows) are never written or
//     read, and suh comes two steps ahead in registers; the parity split's rest block is a launch of its own.
//   - down: the rows are the assembled fp16 intermediate's, copied as they are.
// Output: the same fp32 partials (Z) at the same places; the served epilogues and combine run after it unchanged.
// (TF_DS_PROMPT_OVL=dn: a launch may cover a range of column blocks (cb0) and put its columns in a compact Z part
// (z_ld, z_c0); each element's operands and sums are the same, only where it lands moves.) The
// work list is walked from its end (the shared expert, every row and the widest trellises, first). The register entries
// (run / program: one step of words ahead in registers, 128 columns) cover K2 past the ring's and other widths.
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tf_ds_v2 {

// ---- experts_grouped.cuh's decode, copied as it is -------------------------------------------------------------------

template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

template <int K2>
struct Fmt {
    static constexpr int TW = 4 * K2;
    static constexpr int LW = (TW + 31) / 32;
    static constexpr int GV = (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return (p >> 1) * K2 + ((p & 1) ? K2 : (K2 >> 1)); }
    __host__ __device__ static constexpr int off(int j) { return end(GV - 1) - end(j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 128 * K2;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w[0] >> 16;
        st[4] = (w[0] >> 12) & 0xffffu;
        st[5] = (w[0] >> 8) & 0xffffu;
        st[6] = (w[0] >> 4) & 0xffffu;
        st[7] = w[0] & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = __ldg(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? __ldg(p + l * 32) : 0u;
    }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

// ---- rot_in_kernel's arithmetic (experts.cu), its contraction pinned ----------------------------------------------
// The served cubin's SASS (sm_121, nvcc 13.3, -O3): the first butterfly stage is contracted, a = fma(x0, s0, x1 s1),
// b = fma(x0, s0, -(x1 s1)) (and c / d from x2, x3), every later stage an FADD, then fp16(v * HAD_SCALE). Written here
// with explicit _rn intrinsics so no compiler choice can differ.
constexpr float HAD_SCALE = 0.08838834764831845f;

__device__ __forceinline__ void rot4(const float (&x)[4], const float (&s)[4], float (&v)[4], int lane) {
    const float p1 = __fmul_rn(x[1], s[1]);
    const float p3 = __fmul_rn(x[3], s[3]);
    const float a = __fmaf_rn(x[0], s[0], p1), b = __fmaf_rn(x[0], s[0], -p1);
    const float c = __fmaf_rn(x[2], s[2], p3), d = __fmaf_rn(x[2], s[2], -p3);
    v[0] = __fadd_rn(a, c);
    v[1] = __fadd_rn(b, d);
    v[2] = __fsub_rn(a, c);
    v[3] = __fsub_rn(b, d);
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
        }
    }
}

// ---- the program -----------------------------------------------------------------------------------------------------
#ifndef TF_V2_MT
#define TF_V2_MT 2                // m16 row tiles a program
#endif
#ifndef TF_V2_MINB
#define TF_V2_MINB 2              // programs an SM (the register budget: 65,536 / (256 x MINB))
#endif
constexpr int MT = TF_V2_MT;
constexpr int ROWS = 16 * MT;     // member rows a program
constexpr int KS = 8;             // k tiles a step: one 128-column Hadamard block of K
constexpr int LD = KS * 16 + 8;   // halfs a staged row (padded: ldmatrix rows in distinct banks)
constexpr int WARPS = 8;          // one n16 output tile a warp: 128 columns a program
constexpr int STAGES = 2;         // staged row blocks (gate / up: rotated in the program; down: cp.async)
constexpr int RPW = ROWS / WARPS; // rows a warp rotates a step

struct Args {
    // the rows: gate / up rotate x [rows, D] bf16 (row stride x_stride) with suh [E, D] fp16 per matrix; down stages
    // xa [P, K] fp16
    const __nv_bfloat16* x;
    int x_stride;
    int cb0;     // TF_DS_PROMPT_OVL=dn: the launch's first (128- or 256-column) block; 0: every block from the first
    const half* suh0;
    const half* suh1;
    const half* xa;
    // the weights: main (tp, k2 per matrix) and, for gate / up, the parity split's rest (0 pointer: another node's)
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int64_t* rtp0;
    const int64_t* rtp1;
    const int* rk2_0;
    const int* rk2_1;
    // the grouping (exl3 experts' group_place / work_list: the work list of ROWS-row member groups)
    const int* uids;
    const int* ucount;
    const int* members;
    const int* work;
    int maxm, slots;
    // outputs: Z [mats, P, N] fp32 (main), Zr [mats, P, NR] (rest)
    float* z;
    float* zr;
    int K, N, NR, P;
    int chain;   // k tiles a chain (K / 16 / (splits x warps of the served launch))
    int wk;      // chains a split
    int fold;    // 1: the splits' sums added in order from 0 (gate / up), 0: one split written as is (down)
    int k2_lo;   // a program runs only for a matrix of K2 in [k2_lo, the entry's HI] (one launch a K2 window)
    // TF_DS_PROMPT_OVL=dn: the main Z as a compact column part [mats, P, z_ld] holding columns [z_c0, z_c0 + z_ld)
    // (0: Z [mats, P, N], every column at its own place); only where the outputs land changes
    int z_ld;
    int z_c0;
};
// the layout exl3_prompt.zig's Args writes
static_assert(sizeof(Args) == 200 && offsetof(Args, cb0) == 12 && offsetof(Args, suh0) == 16 && offsetof(Args, maxm) == 136 &&
              offsetof(Args, z) == 144 && offsetof(Args, fold) == 184 && offsetof(Args, k2_lo) == 188 &&
              offsetof(Args, z_ld) == 192 && offsetof(Args, z_c0) == 196, "Args layout");

template <int CB, int K2, bool ROT>
__device__ __forceinline__ void run(const Args& g, const uint32_t* __restrict__ T, int NTILES, int nt0, int e,
                                    const half* __restrict__ suh, float* __restrict__ Z, int N, int col0,
                                    const int* rows_sh, const int* xrow_sh, int nlive,
                                    half (&As)[STAGES][ROWS][LD]) {
    constexpr bool FOLD = ROT;                                // gate / up folds its splits; down has one
    constexpr int LW = Fmt<K2>::LW, TW = Fmt<K2>::TW;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const LaneMap<K2> map(lane);
    const int KT = g.K >> 4, NSTEP = KT / KS;
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)nt0 + warp) * TW + lane;   // this warp's n16 tile, k tile 0

    // the staged rows: gate / up rotate RPW rows a warp (rows warp * RPW ..; a dead row is skipped, the warp
    // uniformly), each lane 4 columns of the 128-block; down copies 16 bytes a thread ROWS / 16 times a step
    uint2 xv[RPW];
    uint2 sv;
    auto fetch_rows = [&](int st) {
        if constexpr (ROT) {
            const int c = st * 128 + 4 * lane;
            sv = *reinterpret_cast<const uint2*>(suh + (size_t)e * g.K + c);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const int xr = xrow_sh[warp * RPW + i];
                if (xr >= 0) xv[i] = *reinterpret_cast<const uint2*>(g.x + (size_t)xr * g.x_stride + c);
            }
        }
    };
    auto stage_rows = [&](int st, int buf) {
        if constexpr (ROT) {
            const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
            const float s[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                if (xrow_sh[warp * RPW + i] < 0) continue;     // a dead row: its A rows are never written out
                const __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(&xv[i].x);
                const __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(&xv[i].y);
                const float x[4] = {__low2float(x01), __high2float(x01), __low2float(x23), __high2float(x23)};
                float v[4];
                rot4(x, s, v, lane);
                const __half2 o01 = __halves2half2(__float2half_rn(__fmul_rn(v[0], HAD_SCALE)), __float2half_rn(__fmul_rn(v[1], HAD_SCALE)));
                const __half2 o23 = __halves2half2(__float2half_rn(__fmul_rn(v[2], HAD_SCALE)), __float2half_rn(__fmul_rn(v[3], HAD_SCALE)));
                uint2 o;
                o.x = *reinterpret_cast<const uint32_t*>(&o01);
                o.y = *reinterpret_cast<const uint32_t*>(&o23);
                *reinterpret_cast<uint2*>(&As[buf][warp * RPW + i][4 * lane]) = o;
            }
        } else {
            constexpr int CPT = ROWS / 16;                    // 16-byte chunks a thread a step
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                const int idx = threadIdx.x + c * 256;        // chunk: row idx / 16, 16 bytes idx % 16
                const int row = idx >> 4, part = idx & 15;
                const int r = rows_sh[row];
                const half* src = g.xa + (size_t)(r < 0 ? 0 : r) * g.K + st * 128 + part * 8;
                cp_async16(&As[buf][row][part * 8], src, r >= 0);
            }
            cp_async_commit();
        }
    };

    uint32_t wcur[KS][LW], wnext[KS][LW];
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) load_words<K2>(wcur[kk], tp + (size_t)kk * kstride, lane);
    fetch_rows(0);
    stage_rows(0, 0);
    if constexpr (!ROT) {
        if (NSTEP > 1) stage_rows(1, 1);
        else cp_async_commit();
        cp_async_wait<1>();
    }
    __syncthreads();

    float acc[MT][2][4], sacc[MT][2][4], tot[FOLD ? MT : 1][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[m][h][c] = sacc[m][h][c] = 0.f;
                if constexpr (FOLD) tot[m][h][c] = 0.f;
            }
    int left = g.chain, wk = 0;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

    for (int st = 0; st < NSTEP; ++st) {
        const int buf = st & 1;
        // the next step's words and (gate / up) rows, in flight behind this step's mma
        if (st + 1 < NSTEP) {
#pragma unroll
            for (int kk = 0; kk < KS; ++kk)
                load_words<K2>(wnext[kk], tp + (size_t)((st + 1) * KS + kk) * kstride, lane);
            if constexpr (ROT) fetch_rows(st + 1);
        }
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(wcur[kk], map, lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < nlive) {
                    uint32_t a[4];
                    ldmatrix_x4(a, &As[buf][m * 16 + lrow][kk * 16 + lcol]);
                    mma16816(acc[m][0], a, b0);
                    mma16816(acc[m][1], a, b1);
                }
            }
            if (--left == 0) {                                // a chain ends: into its split's sum, in order
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            sacc[m][h][c] = wk == 0 ? acc[m][h][c] : __fadd_rn(sacc[m][h][c], acc[m][h][c]);
                            acc[m][h][c] = 0.f;
                        }
                left = g.chain;
                if (++wk == g.wk) {                           // a split ends: into the total, in order from 0
                    wk = 0;
                    if constexpr (FOLD) {
#pragma unroll
                        for (int m = 0; m < MT; ++m)
#pragma unroll
                            for (int h = 0; h < 2; ++h)
#pragma unroll
                                for (int c = 0; c < 4; ++c) tot[m][h][c] = __fadd_rn(tot[m][h][c], sacc[m][h][c]);
                    }
                }
            }
        }
        if (st + 1 < NSTEP) {
#pragma unroll
            for (int kk = 0; kk < KS; ++kk)
#pragma unroll
                for (int l = 0; l < LW; ++l) wcur[kk][l] = wnext[kk][l];
            if constexpr (ROT) {
                stage_rows(st + 1, buf ^ 1);
            } else {
                cp_async_wait<0>();                           // step st + 1's rows have landed (this thread's)
            }
        }
        __syncthreads();
        if constexpr (!ROT) {
            if (st + 2 < NSTEP) stage_rows(st + 2, buf);      // the buffer this step read, two steps ahead
        }
    }
    // Z [mats, P, N]: rows g8 and g8 + 8 of each live m16 tile, columns 2 t of each n8 half of the warp's tile
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < nlive) {
            const int r0 = rows_sh[m * 16 + g8], r1 = rows_sh[m * 16 + g8 + 8];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = col0 + warp * 16 + h * 8 + 2 * t;
                const float* o = FOLD ? tot[FOLD ? m : 0][h] : sacc[m][h];
                if (r0 >= 0) *reinterpret_cast<float2*>(Z + (size_t)r0 * N + col) = make_float2(o[0], o[1]);
                if (r1 >= 0) *reinterpret_cast<float2*>(Z + (size_t)r1 * N + col) = make_float2(o[2], o[3]);
            }
        }
    }
}

// grid: x = (128-column block, matrix) pairs (gate / up: main blocks then the rest block, each for both matrices;
// down: its blocks, one matrix), y = the ROWS-row work list's (expert place, member group) pairs.
template <int CB, int LO, int HI, bool ROT>
__device__ __forceinline__ void program(const Args& g) {
    const int u = g.work[2 * blockIdx.y];
    if (u >= g.ucount[0]) return;
    const int mb = g.work[2 * blockIdx.y + 1];
    const int mats = ROT ? 2 : 1;
    const int mat = (int)blockIdx.x % mats;
    const int cb = g.cb0 + (int)blockIdx.x / mats;
    const int nmain = g.N / 128;
    const bool rest = cb >= nmain;
    const int e = g.uids[u];
    const int64_t tp = rest ? (mat ? g.rtp1[e] : g.rtp0[e]) : (mat ? g.tp1[e] : g.tp0[e]);
    if (tp == 0) return;                                      // the rest block of an expert another node computes
    const int k2 = rest ? (mat ? g.rk2_1[e] : g.rk2_0[e]) : (mat ? g.k2_1[e] : g.k2_0[e]);
    if (k2 < g.k2_lo || k2 > HI) return;                      // another launch's K2 window
    const int NTILES = (rest ? g.NR : g.N) >> 4;
    const int nt0 = (rest ? cb - nmain : cb) * 8;
    const int zld = (rest || g.z_ld == 0) ? (rest ? g.NR : g.N) : g.z_ld;   // Z's row stride
    const int zc0 = (rest || g.z_ld == 0) ? 0 : g.z_c0;                   // Z's first column
    float* Z = (rest ? g.zr : g.z) + (size_t)mat * g.P * zld;

    __shared__ int rows_sh[ROWS];
    __shared__ int xrow_sh[ROWS];
    __shared__ __align__(16) half As[STAGES][ROWS][LD];
    if (threadIdx.x < ROWS) {
        const int m = mb * ROWS + threadIdx.x;
        const int code = m < g.maxm ? g.members[(size_t)u * g.maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * g.slots + (code & 31) : -1;
        xrow_sh[threadIdx.x] = code >= 0 ? (code >> 5) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                               // members come first: an empty group
    int nlive = 0;
#pragma unroll
    for (int m = 0; m < MT; ++m) nlive += rows_sh[m * 16] >= 0;
    const half* suh = ROT ? (mat ? g.suh1 : g.suh0) : nullptr;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(tp);
    switch (k2) {
#define TF_V2_CASE(K2_)                                                                                         \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            run<CB, K2_, ROT>(g, T, NTILES, nt0, e, suh, Z, zld, nt0 * 16 - zc0, rows_sh, xrow_sh, nlive, As);  \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_V2_CASE(2)
        TF_V2_CASE(3)
        TF_V2_CASE(4)
        TF_V2_CASE(5)
        TF_V2_CASE(6)
        TF_V2_CASE(7)
        TF_V2_CASE(8)
        TF_V2_CASE(9)
        TF_V2_CASE(10)
        TF_V2_CASE(11)
        TF_V2_CASE(12)
        TF_V2_CASE(13)
        TF_V2_CASE(14)
        TF_V2_CASE(15)
        TF_V2_CASE(16)
#undef TF_V2_CASE
        default:
            __trap();
    }
}


// ---- the ring pipeline (K2 to 10) ------------------------------------------------------------------------------------
// The same arithmetic as run(); the trellis words and the input rows of step st + 2 come by cp.async into a ring of three
// shared-memory stages while step st multiplies, each warp copying its own tiles' words and its own RPW rows (so only
// one block barrier a step: the rotated rows, rotated in place, visible to every warp). No prefetch registers.
constexpr int RS = 3;             // ring stages

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}

template <int CB, int K2, bool ROT, int TWM, int NW, int MTP>
__device__ __forceinline__ void run_ring(const Args& g, const uint32_t* __restrict__ T, int NTILES, int nt0, int e,
                                         const half* __restrict__ suh, float* __restrict__ Z, int N, int col0,
                                         const int* rows_sh, const int* xrow_sh, int nlive, int nact,
                                         half (&As)[RS][16 * MTP][LD], uint32_t (&Ws)[RS][NW][KS][TWM]) {
    constexpr int MT = MTP, ROWS = 16 * MTP;                 // this entry's m16 tiles and rows a program
    constexpr int RPW = ROWS / NW;                            // rows a warp copies and rotates a step
    // warps past nact (a program over fewer columns than its warps: the parity rest block in a 256-column launch)
    // only copy and rotate rows
    const bool act = (int)(threadIdx.x >> 5) < nact;
    constexpr bool FOLD = ROT;
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    static_assert(TW <= TWM, "ring: a tile's words in its slot");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const LaneMap<K2> map(lane);
    const int KT = g.K >> 4, NSTEP = KT / KS;
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)nt0 + warp) * TW;      // this warp's n16 tile, k tile 0

    // step st's copies into slot st % RS: this warp's eight tiles' words (K2 16-byte chunks a tile) and its RPW rows
    // (x bf16 or xa fp16, 128 columns), a 16-byte chunk a lane at a time; a dead row's chunk zero-filled
    auto issue = [&](int st) {
        if (st < NSTEP) {
            const int slot = st % RS;
#pragma unroll
            for (int c0 = 0; c0 < KS * K2; c0 += 32) {
                const int c = c0 + lane;
                if (act && c < KS * K2) {
                    const int kk = c / K2, q = c - kk * K2;
                    cp_async16(&Ws[slot][warp][kk][q * 4], tp + (size_t)(st * KS + kk) * kstride + q * 4, true);
                }
            }
#pragma unroll
            for (int c = 0; c < RPW * 16 / 32; ++c) {
                const int idx = lane + 32 * c;
                const int row = warp * RPW + (idx >> 4), part = idx & 15;
                if constexpr (ROT) {
                    const int xr = xrow_sh[row];
                    cp_async16(&As[slot][row][part * 8], g.x + (size_t)(xr < 0 ? 0 : xr) * g.x_stride + st * 128 + part * 8, xr >= 0);
                } else {
                    const int r = rows_sh[row];
                    cp_async16(&As[slot][row][part * 8], g.xa + (size_t)(r < 0 ? 0 : r) * g.K + st * 128 + part * 8, r >= 0);
                }
            }
        }
        cp_async_commit();
    };

    issue(0);
    issue(1);
    // suh for the rotations, two steps ahead in registers (svc: step st's, svn: step st + 1's)
    uint2 svc = make_uint2(0, 0), svn = make_uint2(0, 0);
    if constexpr (ROT) {
        svc = __ldg(reinterpret_cast<const uint2*>(suh + (size_t)e * g.K + 4 * lane));
        if (NSTEP > 1) svn = __ldg(reinterpret_cast<const uint2*>(suh + (size_t)e * g.K + 128 + 4 * lane));
    }

    float acc[MT][2][4], sacc[MT][2][4], tot[FOLD ? MT : 1][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[m][h][c] = sacc[m][h][c] = 0.f;
                if constexpr (FOLD) tot[m][h][c] = 0.f;
            }
    int left = g.chain, wk = 0;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

    for (int st = 0; st < NSTEP; ++st) {
        const int slot = st % RS;
        cp_async_wait<1>();                                   // this thread's copies of step st (step st + 1's may fly)
        __syncwarp();
        if constexpr (ROT) {                                  // this warp's rows of x, rotated in place into fp16
            const __half2 s01 = *reinterpret_cast<const __half2*>(&svc.x), s23 = *reinterpret_cast<const __half2*>(&svc.y);
            const float s[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const int row = warp * RPW + i;
                if (xrow_sh[row] < 0) continue;
                const uint2 xv = *reinterpret_cast<const uint2*>(&As[slot][row][4 * lane]);
                const __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(&xv.x);
                const __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(&xv.y);
                const float x[4] = {__low2float(x01), __high2float(x01), __low2float(x23), __high2float(x23)};
                float v[4];
                rot4(x, s, v, lane);
                const __half2 o01 = __halves2half2(__float2half_rn(__fmul_rn(v[0], HAD_SCALE)), __float2half_rn(__fmul_rn(v[1], HAD_SCALE)));
                const __half2 o23 = __halves2half2(__float2half_rn(__fmul_rn(v[2], HAD_SCALE)), __float2half_rn(__fmul_rn(v[3], HAD_SCALE)));
                uint2 o;
                o.x = *reinterpret_cast<const uint32_t*>(&o01);
                o.y = *reinterpret_cast<const uint32_t*>(&o23);
                *reinterpret_cast<uint2*>(&As[slot][row][4 * lane]) = o;
            }
        }
        __syncthreads();                                      // every warp's rows of step st; step st - 1's slot free
        issue(st + 2);
        if constexpr (ROT) {
            svc = svn;
            if (st + 2 < NSTEP) svn = __ldg(reinterpret_cast<const uint2*>(suh + (size_t)e * g.K + (st + 2) * 128 + 4 * lane));
        }
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            if (!act) break;
            uint32_t w[LW];
#pragma unroll
            for (int l = 0; l < LW; ++l) w[l] = (l * 32 + lane < TW) ? Ws[slot][warp][kk][l * 32 + lane] : 0u;
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w, map, lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < nlive) {
                    uint32_t a[4];
                    ldmatrix_x4(a, &As[slot][m * 16 + lrow][kk * 16 + lcol]);
                    mma16816(acc[m][0], a, b0);
                    mma16816(acc[m][1], a, b1);
                }
            }
            if (--left == 0) {
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            sacc[m][h][c] = wk == 0 ? acc[m][h][c] : __fadd_rn(sacc[m][h][c], acc[m][h][c]);
                            acc[m][h][c] = 0.f;
                        }
                left = g.chain;
                if (++wk == g.wk) {
                    wk = 0;
                    if constexpr (FOLD) {
#pragma unroll
                        for (int m = 0; m < MT; ++m)
#pragma unroll
                            for (int h = 0; h < 2; ++h)
#pragma unroll
                                for (int c = 0; c < 4; ++c) tot[m][h][c] = __fadd_rn(tot[m][h][c], sacc[m][h][c]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    if (!act) return;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < nlive) {
            const int r0 = rows_sh[m * 16 + g8], r1 = rows_sh[m * 16 + g8 + 8];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = col0 + warp * 16 + h * 8 + 2 * t;
                const float* o = FOLD ? tot[FOLD ? m : 0][h] : sacc[m][h];
                if (r0 >= 0) *reinterpret_cast<float2*>(Z + (size_t)r0 * N + col) = make_float2(o[0], o[1]);
                if (r1 >= 0) *reinterpret_cast<float2*>(Z + (size_t)r1 * N + col) = make_float2(o[2], o[3]);
            }
        }
    }
}

template <int CB, int LO, int HI, bool ROT, int NW, int MTP>
__device__ __forceinline__ void program_ring(const Args& g) {
    // the work list from its end: the shared expert (the last place, every row: the most groups and the widest
    // trellises) starts first and overlaps the routed experts' weight stream instead of trailing it
    constexpr int MT = MTP, ROWS = 16 * MTP;
    const int y = (int)(gridDim.y - 1 - blockIdx.y);
    const int u = g.work[2 * y];
    if (u >= g.ucount[0]) return;
    const int mb = g.work[2 * y + 1];
    const int mats = ROT ? 2 : 1;
    const int mat = (int)blockIdx.x % mats;
    const int cb = g.cb0 + (int)blockIdx.x / mats;
    const int nmain = g.N / (16 * NW);
    const bool rest = cb >= nmain;
    const int e = g.uids[u];
    const int64_t tp = rest ? (mat ? g.rtp1[e] : g.rtp0[e]) : (mat ? g.tp1[e] : g.tp0[e]);
    if (tp == 0) return;
    const int k2 = rest ? (mat ? g.rk2_1[e] : g.rk2_0[e]) : (mat ? g.k2_1[e] : g.k2_0[e]);
    if (k2 < g.k2_lo || k2 > HI) return;                      // another launch's K2 window
    const int NTILES = (rest ? g.NR : g.N) >> 4;
    const int nt0 = (rest ? cb - nmain : cb) * NW;
    const int nact = min(NW, NTILES - nt0);                   // the warps with a column tile
    const int zld = (rest || g.z_ld == 0) ? (rest ? g.NR : g.N) : g.z_ld;   // Z's row stride
    const int zc0 = (rest || g.z_ld == 0) ? 0 : g.z_c0;                   // Z's first column
    float* Z = (rest ? g.zr : g.z) + (size_t)mat * g.P * zld;
    constexpr int TWM = 4 * HI;

    __shared__ int rows_sh[ROWS];
    __shared__ int xrow_sh[ROWS];
    extern __shared__ __align__(16) unsigned char ring_dyn[];
    auto& As = *reinterpret_cast<half (*)[RS][ROWS][LD]>(ring_dyn);
    auto& Ws = *reinterpret_cast<uint32_t (*)[RS][NW][KS][TWM]>(ring_dyn + sizeof(half) * RS * ROWS * LD);
    if (threadIdx.x < ROWS) {
        const int m = mb * ROWS + threadIdx.x;
        const int code = m < g.maxm ? g.members[(size_t)u * g.maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * g.slots + (code & 31) : -1;
        xrow_sh[threadIdx.x] = code >= 0 ? (code >> 5) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;
    int nlive = 0;
#pragma unroll
    for (int m = 0; m < MT; ++m) nlive += rows_sh[m * 16] >= 0;
    const half* suh = ROT ? (mat ? g.suh1 : g.suh0) : nullptr;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(tp);
    switch (k2) {
#define TF_V2_CASE(K2_)                                                                                         \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            run_ring<CB, K2_, ROT, TWM, NW, MTP>(g, T, NTILES, nt0, e, suh, Z, zld, nt0 * 16 - zc0, rows_sh, xrow_sh, nlive, nact, As, Ws); \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_V2_CASE(2)
        TF_V2_CASE(3)
        TF_V2_CASE(4)
        TF_V2_CASE(5)
        TF_V2_CASE(6)
        TF_V2_CASE(7)
        TF_V2_CASE(8)
        TF_V2_CASE(9)
        TF_V2_CASE(10)
#undef TF_V2_CASE
        default:
            __trap();
    }
}

}  // namespace tf_ds_v2

// The member rows a program (the work list the caller passes: exl3 experts' 32-row or 64-row one).
extern "C" __device__ int tf_ds_v2_rows = tf_ds_v2::ROWS;

// The entry points by K2 range (exl3_experts.range: (8, 8), (2, 10), (2, 16)), codebook mul1 (the served pack's).
#define TF_V2_ENTRY(NAME, LO, HI, ROT)                                                                          \
    extern "C" __global__ void __launch_bounds__(256, TF_V2_MINB) NAME(const tf_ds_v2::Args g) {               \
        tf_ds_v2::program<2, LO, HI, ROT>(g);                                                                   \
    }
TF_V2_ENTRY(tf_ds_v2_gu_8_8, 8, 8, true)
TF_V2_ENTRY(tf_ds_v2_gu_2_10, 2, 10, true)
TF_V2_ENTRY(tf_ds_v2_gu_2_16, 2, 16, true)
TF_V2_ENTRY(tf_ds_v2_dn_8_8, 8, 8, false)
TF_V2_ENTRY(tf_ds_v2_dn_2_10, 2, 10, false)
TF_V2_ENTRY(tf_ds_v2_dn_2_16, 2, 16, false)
// K2 2..6 (the served pack's 2- and 3-bit experts): the ring pipeline, 128 or 256 columns a program (NW warps)
#define TF_V2_RING(NAME, LO, HI, ROT, NW, MTP, MINB)                                                            \
    extern "C" __global__ void __launch_bounds__(32 * NW, MINB) NAME(const tf_ds_v2::Args g) {                \
        tf_ds_v2::program_ring<2, LO, HI, ROT, NW, MTP>(g);                                                     \
    }
TF_V2_RING(tf_ds_v2_gu_2_6, 2, 6, true, 8, 2, TF_V2_MINB)
TF_V2_RING(tf_ds_v2_dn_2_6, 2, 6, false, 8, 2, TF_V2_MINB)
TF_V2_RING(tf_ds_v2_gu16_2_10, 2, 10, true, 16, 2, 1)
TF_V2_RING(tf_ds_v2_dn16_2_10, 2, 10, false, 16, 2, 1)

// their dynamic shared memory (bytes): 8 warps and K2 to 6, 16 warps and K2 to 10
extern "C" __device__ int tf_ds_v2_ring_smem[2] = {
    (int)(sizeof(half) * tf_ds_v2::RS * 32 * tf_ds_v2::LD + 4 * tf_ds_v2::RS * 8 * tf_ds_v2::KS * 24),
    (int)(sizeof(half) * tf_ds_v2::RS * 32 * tf_ds_v2::LD + 4 * tf_ds_v2::RS * 16 * tf_ds_v2::KS * 40)};
