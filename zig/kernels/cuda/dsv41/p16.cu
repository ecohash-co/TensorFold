// TF_DS_PROMPT_P16 (tier 2, a numerics change; p16.zig): on an SP prompt chunk, the fp32 partials toOwners sends to
// the rows' owners go over the wire as bf16, and the owner widens them back to fp32 before the hc mix reads them.
//
//   tf_ds_p16_pack_kernel        dst bf16 [count] = round-to-nearest-even of src fp32 [count] (the sender, before NCCL)
//   tf_ds_p16_widen_rows_kernel  dst fp32 rows = the bf16 rows of src widened (exact: the bf16 bits are fp32's top
//                                half), `rows` rows of n4 * 4 values, src / dst rows src_ld4 / dst_ld4 four-value
//                                words apart (the owner, in place of toOwnersCopy's byte copy of a received part)
//
// Integer arithmetic only (no FTZ, no flags): tf_p16_rne is __float2bfloat16_rn on every non-NaN value (zero and its
// sign, subnormals, ties to even, overflow past bf16's largest finite value to infinity, infinities); a NaN stays a
// NaN of its sign (the quiet bit set), where the intrinsic gives the canonical 0x7fff. tools/dsv41-zig/p16_host_check.cu
// checks this file's function against the intrinsic's host implementation on all 2^32 inputs; p16.zig mirrors it for
// the host tests.
#include <stdint.h>

__host__ __device__ __forceinline__ unsigned short tf_p16_rne(unsigned u) {
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x0040u);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__host__ __device__ __forceinline__ unsigned tf_p16_widen(unsigned short h) { return (unsigned)h << 16; }

// count values, src 16-byte aligned, dst 8-byte aligned, count a multiple of 4 (p16.zig checks): four a thread
extern "C" __global__ void tf_ds_p16_pack_kernel(const uint4* src, uint2* dst, unsigned long long n4) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const uint4 v = src[i];
        dst[i] = make_uint2((unsigned)tf_p16_rne(v.x) | ((unsigned)tf_p16_rne(v.y) << 16),
                            (unsigned)tf_p16_rne(v.z) | ((unsigned)tf_p16_rne(v.w) << 16));
    }
}

// a row a blockIdx.y, four values a thread
extern "C" __global__ void tf_ds_p16_widen_rows_kernel(const uint2* src, unsigned long long src_ld4, uint4* dst,
                                                       unsigned long long dst_ld4, unsigned long long n4) {
    const unsigned long long r = blockIdx.y;
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const uint2 v = src[r * src_ld4 + i];
    dst[r * dst_ld4 + i] = make_uint4(tf_p16_widen((unsigned short)(v.x & 0xffffu)), tf_p16_widen((unsigned short)(v.x >> 16)),
                                      tf_p16_widen((unsigned short)(v.y & 0xffffu)), tf_p16_widen((unsigned short)(v.y >> 16)));
}
