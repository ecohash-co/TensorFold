// TF_DS_PROMPT_P16: p16.cu's conversion functions on the host, all 2^32 fp32 inputs (no GPU: nvcc's host compiler).
//   nvcc -O2 -std=c++17 -arch=sm_121 -o /tmp/p16_host_check tools/dsv41-zig/p16/host_check.cu && /tmp/p16_host_check
// Checks: tf_p16_rne == __float2bfloat16_rn (cuda_bf16.h's host path) on every non-NaN input; a NaN stays a NaN of its
// sign; tf_p16_widen(tf_p16_rne(x)) round-trips every bf16 value exactly (all 2^16).
#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include "../../../zig/kernels/cuda/dsv41/p16.cu"

int main() {
    unsigned long long bad = 0, nan = 0;
    for (unsigned long long i = 0; i <= 0xffffffffull; i++) {
        const unsigned u = (unsigned)i;
        const unsigned short mine = tf_p16_rne(u);
        float f;
        std::memcpy(&f, &u, 4);
        if ((u & 0x7fffffffu) > 0x7f800000u) {
            nan++;
            if ((mine & 0x7fffu) <= 0x7f80u || (mine >> 15) != (u >> 31)) bad++;
            continue;
        }
        const __nv_bfloat16 ref = __float2bfloat16_rn(f);
        unsigned short rb;
        std::memcpy(&rb, &ref, 2);
        if (rb != mine) {
            if (bad < 8) std::printf("mismatch %08x: ours %04x intrinsic %04x\n", u, mine, rb);
            bad++;
        }
    }
    unsigned long long rt = 0;
    for (unsigned h = 0; h < 0x10000u; h++) {
        const unsigned w = tf_p16_widen((unsigned short)h);
        if ((h & 0x7fffu) > 0x7f80u) continue; // NaN: checked above
        if (tf_p16_rne(w) != h) rt++;
    }
    std::printf("p16 host check: %llu inputs (%llu NaN), %llu mismatches, %llu round-trip failures\n", 1ull << 32, nan, bad, rt);
    return bad == 0 && rt == 0 ? 0 : 1;
}
