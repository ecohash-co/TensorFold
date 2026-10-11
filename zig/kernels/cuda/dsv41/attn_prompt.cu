// TF_DS_PROMPT_ATTN=tsel (stage ZT1): the prompt indexer's top-k selection in one launch a row, in place of
// topk_select_pruned's chain (topk_select of the tile maxima -> _gather_tiles -> torch's top-k of the gathered keys ->
// _topk_finish) and of topk_select's torch path (torch's top-k -> _topk_finish).
//
// Exact by construction: what those chains write depends only on the SET of the row's k largest keys. The keys are
// unique (the score's ordered bits above, the inverted position below), so that set is one set however it is found;
// _topk_finish writes its positions (0xFFFFFFFF - the key's low word) sorted ascending, -1 where a position is at or
// past vis[r], and so does this kernel. The pruned chain's set is the row's k largest keys (kernels.py
// topk_select_pruned: every one of them lies in one of the k tiles with the largest maxima); here those tiles are the
// ones whose maximum is at least T, the k-th largest maximum (the maxima are keys too: unique), and the k largest keys
// are found among their keys. (The prompt's scores run without DEAD_PAD, so every key t < n is written and every tile
// maximum is a real key of its tile. Exactly k tiles have a maximum >= T, each holds at least one real key, so the
// chain's k gathered tiles hold at least k real keys and its PAD fill (positions >= n of the last tile) never enters
// the set; this kernel reads only positions < n.)
//
// The selection: a radix select from the top, 12 bits on the first pass over the keys in global memory (L2 / DRAM),
// then, once the bin that holds the k-th largest has at most kCand keys, one pass that keeps the keys above that bin
// (selected) and the bin's keys (candidates) in shared memory, where 8-bit passes finish it. Keys compare as unsigned
// after flipping the sign bit (their signed order).
#include <stdint.h>

namespace {

constexpr int kThreads = 1024;
constexpr int kK = 512;                 // index_topk (the only k the engine passes)
constexpr int kCand = 6144;             // candidates kept in shared memory
constexpr int kBins = 4096;             // first pass: 12 bits

typedef unsigned long long u64;

struct Smem {
  unsigned int hist[kBins];
  u64 cand[kCand];
  u64 sel[kK];
  unsigned int idx[kK];
  u64 prefix, mask;
  int want, ncand, nsel, shift, done;
  u64 thr;                              // tile mode: T (unsigned order)
};

__device__ __forceinline__ u64 ukey(long long k) { return (u64)k ^ 0x8000000000000000ull; }

// The row's keys: keys[0 .. n).
struct RowSrc {
  const long long* row;
  int n;
  template <class F> __device__ __forceinline__ void operator()(F f) const {
    for (int i = threadIdx.x; i < n; i += kThreads) f(ukey(row[i]));
  }
};

// The tile maxima: tmax[0 .. nt).
struct TmaxSrc {
  const long long* tm;
  int nt;
  template <class F> __device__ __forceinline__ void operator()(F f) const {
    for (int i = threadIdx.x; i < nt; i += kThreads) f(ukey(tm[i]));
  }
};

// The keys of the tiles whose maximum is at least thr: a warp a tile, two keys a lane.
struct TileSrc {
  const long long* row;
  const long long* tm;
  int n, nt;
  u64 thr;
  template <class F> __device__ __forceinline__ void operator()(F f) const {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int j = w; j < nt; j += kThreads / 32) {
      if (ukey(tm[j]) < thr) continue;
      const int t0 = j * 64 + l, t1 = t0 + 32;
      if (t0 < n) f(ukey(row[t0]));
      if (t1 < n) f(ukey(row[t1]));
    }
  }
};

// The shared candidates.
struct CandSrc {
  const u64* c;
  int n;
  template <class F> __device__ __forceinline__ void operator()(F f) const {
    for (int i = threadIdx.x; i < n; i += kThreads) f(c[i]);
  }
};

// A broken invariant (fewer values than k, or values not unique): stop the kernel with an error the stream reports
// (cudaErrorLaunchFailure at the next synchronization) instead of looping or writing past the shared arrays. Never
// taken on the indexer's keys (unique, n >= k: the launcher refuses n < k).
__device__ __forceinline__ void broken() { __trap(); }

// One pass: the histogram of the next `bits` bits (1 .. 12) below `shift` of the values matching the prefix, then the
// bin holding the want-th largest of them (warp 0: each lane a run of ceil(nb / 32) bins from the top, a prefix sum
// over lanes). Every pass narrows the prefix by `bits` bits (shift strictly falls) or traps: the callers' loops end.
//
// (Stage ZT2: ZT1's lanes took nb / 32 bins each, which is 0 for nb < 32. The 8-bit passes on the candidates run
// from shift 52 (64 - 12) down to 4, so every selection ended with a 4-bit pass (nb 16) in which no lane held a
// bin: nothing claimed the want-th value, shift stayed 4, and `while (s.shift > 0) pass(..)` spun forever. That is
// arm TA's hang: the first chunk whose indexer selection took tsel (4,224-token prompt: its second chunk, 1,056
// compressed keys, not a power of two) never finished, and every rank waited on it.)
template <class Src>
__device__ void pass(Smem& s, const Src& src, int bits) {
  const int nb = 1 << bits;
  for (int b = threadIdx.x; b < nb; b += kThreads) s.hist[b] = 0;
  __syncthreads();
  const u64 prefix = s.prefix, mask = s.mask;
  const int sh = s.shift - bits;
  src([&](u64 u) {
    if ((u & mask) == prefix) atomicAdd(&s.hist[(u >> sh) & (u64)(nb - 1)], 1u);
  });
  __syncthreads();
  if (threadIdx.x < 32) {
    const int l = threadIdx.x;
    const int per = (nb + 31) / 32;     // >= 1 for any nb (nb a power of two: lanes past nb / per hold none)
    const int hi = nb - 1 - l * per;    // this lane's bins: hi, hi - 1, .., max(hi - per + 1, 0)
    const int lo = hi - per + 1 > 0 ? hi - per + 1 : 0;
    unsigned int sum = 0;
    for (int b = hi; b >= lo; --b) sum += s.hist[b];
    unsigned int incl = sum;
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (l >= o) incl += v;
    }
    const unsigned int excl = incl - sum;
    const unsigned int want = (unsigned int)s.want;
    const bool here = excl < want && incl >= want;
    const unsigned int ballot = __ballot_sync(0xffffffffu, here);
    if (here && l == __ffs(ballot) - 1) {   // (incl rises with l: at most one lane, sum > 0 so hi >= lo)
      unsigned int w = want - excl;
      int b = hi;
      for (; b > lo; --b) {
        const unsigned int h = s.hist[b];
        if (h >= w) break;
        w -= h;
      }
      s.want = (int)w;
      s.prefix = prefix | ((u64)b << sh);
      s.mask = mask | ((u64)(nb - 1) << sh);
      s.shift = sh;
      s.ncand = (int)s.hist[b];
    }
  }
  __syncthreads();
  // no lane claimed a bin: fewer than want values match the prefix (s.shift is written again only after the next
  // pass's two barriers, so every thread reads this pass's value)
  if (s.shift != sh) broken();
}

// Keeps the values above the prefix's range (selected) and those in it (candidates) in shared memory.
template <class Src>
__device__ void collect(Smem& s, const Src& src, bool into_cand) {
  const u64 prefix = s.prefix, mask = s.mask;
  const u64 top = prefix | ~mask;
  if (threadIdx.x == 0) s.done = 0;
  __syncthreads();
  src([&](u64 u) {
    if (u > top) {
      const int i = atomicAdd(&s.nsel, 1);
      if (i < kK) s.sel[i] = u;
    } else if ((u & mask) == prefix) {
      if (into_cand) {
        const int i = atomicAdd(&s.done, 1);
        if (i < kCand) s.cand[i] = u;
      } else {
        const int i = atomicAdd(&s.nsel, 1);  // the last value (shift 0): the k-th largest itself
        if (i < kK) s.sel[i] = u;
      }
    }
  });
  __syncthreads();
  // the last collect holds exactly k values; before it fewer than k are above the candidates' bin (unique values:
  // never taken)
  if (into_cand ? (s.nsel >= kK || s.done > kCand) : s.nsel != kK) broken();
}

// The k largest values of src (unique) into s.sel[0 .. k).
template <class Src>
__device__ void selectTop(Smem& s, const Src& src, int k) {
  if (threadIdx.x == 0) {
    s.prefix = 0;
    s.mask = 0;
    s.want = k;
    s.nsel = 0;
    s.shift = 64;
  }
  __syncthreads();
  pass(s, src, 12);
  while (s.ncand > kCand && s.shift > 0) pass(s, src, s.shift >= 12 ? 12 : s.shift);
  if (s.shift == 0) {                   // (the bin is one value: want == 1)
    collect(s, src, false);
    return;
  }
  collect(s, src, true);
  if (threadIdx.x == 0) s.ncand = s.done;
  __syncthreads();
  const CandSrc cs{s.cand, s.ncand};
  while (s.shift > 0) pass(s, cs, s.shift >= 8 ? 8 : s.shift);
  collect(s, cs, false);
}

// The selected keys' positions sorted ascending (bitonic, kK entries), -1 at or past vis.
__device__ void finish(Smem& s, long long vis, long long* out) {
  for (int i = threadIdx.x; i < kK; i += kThreads) s.idx[i] = 0xFFFFFFFFu - (unsigned int)(s.sel[i] & 0xFFFFFFFFull);
  __syncthreads();
  for (int size = 2; size <= kK; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const int i = threadIdx.x;
      if (i < kK) {
        const int j = i ^ stride;
        if (j > i) {
          const bool up = (i & size) == 0;
          const unsigned int a = s.idx[i], b = s.idx[j];
          if ((a > b) == up) {
            s.idx[i] = b;
            s.idx[j] = a;
          }
        }
      }
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < kK; i += kThreads) {
    const long long v = (long long)s.idx[i];
    out[i] = v < vis ? v : -1;
  }
}

}  // namespace

// One row a block: keys [rows, ks] (row stride ks, n keys a row), tmax [rows, nt] (null: no pruning: the row's keys
// themselves), vis [rows], out [rows, 512].
extern "C" __global__ void __launch_bounds__(kThreads) tf_ds_tsel_kernel(const long long* keys, long long ks, int n,
                                                                         const long long* tmax, int nt,
                                                                         const long long* vis, long long* out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const long long r = blockIdx.x;
  const long long* row = keys + r * ks;
  if (tmax == nullptr) {
    selectTop(s, RowSrc{row, n}, kK);
  } else {
    const long long* tm = tmax + r * nt;
    selectTop(s, TmaxSrc{tm, nt}, kK);
    // T: the smallest of the k largest maxima
    if (threadIdx.x == 0) s.thr = ~0ull;
    __syncthreads();
    u64 mn = ~0ull;
    for (int i = threadIdx.x; i < kK; i += kThreads) mn = s.sel[i] < mn ? s.sel[i] : mn;
    for (int o = 16; o > 0; o >>= 1) {
      const u64 v = __shfl_xor_sync(0xffffffffu, mn, o);
      mn = v < mn ? v : mn;
    }
    if ((threadIdx.x & 31) == 0) atomicMin(&s.thr, mn);
    __syncthreads();
    selectTop(s, TileSrc{row, tm, n, nt, s.thr}, kK);
  }
  finish(s, vis[r], out + r * kK);
}

// The shared memory the kernel takes (the launcher reads it).
extern "C" __device__ const int tf_ds_tsel_smem = (int)sizeof(Smem);
