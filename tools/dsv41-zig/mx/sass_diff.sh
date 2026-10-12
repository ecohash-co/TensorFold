#!/bin/bash
# stage MX3: the six O2 kernel instantiations of zig/kernels/cuda/dsv41/mx_prompt.cu against the same instantiations of
# TensorFold's upstream nvfp4/experts_prefill.cu (its device code, torch includes dropped), both compiled with the Zig
# build's flags (torch's + -O3, sm_121): each kernel's SASS equal (addresses and symbols normalized), or else (ptxas
# scheduling the staging addresses differently) the same multiset of float / mma instructions. Host compile only, no GPU. usage: sass_diff.sh UPSTREAM_SRC_DIR (…/src/tensorfold/cuda) [NVCC]
set -eu
UP=${1:?upstream src/tensorfold/cuda dir}
NVCC=${2:-/usr/local/cuda/bin/nvcc}
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/nvfp4"
cp "$UP/experts.cuh" "$W/experts.cuh"
python3 - "$UP/nvfp4/experts_prefill.cu" "$W/nvfp4/up.cu" <<'PY'
import sys
src = open(sys.argv[1]).read().split("\n")
end = next(i for i, l in enumerate(src) if l.startswith("template <int FMT, int M, int EPI, int RT, int WM, int WN, int CN, int SG, int ST>") and src[i + 1].startswith("void launch"))
lines = [l for l in src[:end] if not l.startswith(("#include <ATen", "#include <c10", "#include <torch"))]
sig = "(const __nv_bfloat16*, int, int, const uint4*, const float*, int, int, const int*, const int*, const int*, void*, int, float, int);"
for m, epi in ((2, 2), (1, 0)):
    for geo in ("2, 2, 4, 1, 2, 2", "2, 2, 6, 1, 2, 2", "2, 2, 4, 2, 2, 2"):
        lines.append(f"template __global__ void nvfp4_prefill_kernel<1, {m}, {epi}, {geo}>{sig}")
lines.append("}  // namespace")
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
PY
F="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121"
$NVCC -cubin $F -diag-suppress 177 -o "$W/up.cubin" "$W/nvfp4/up.cu"
$NVCC -cubin $F -o "$W/mx.cubin" "$HERE/zig/kernels/cuda/dsv41/mx_prompt.cu"
CUOBJ=$(dirname "$NVCC")/cuobjdump
norm() {  # one file a kernel instantiation: its instructions only
  "$CUOBJ" -sass "$1" | python3 -c '
import re, sys, os
out = sys.argv[1]; os.makedirs(out, exist_ok=True); cur = None; buf = []
def flush():
    if cur: open(os.path.join(out, cur), "w").write("\n".join(buf) + "\n")
for line in sys.stdin:
    m = re.search(r"Function : \S*nvfp4_prefill_kernel(I\S+)", line)
    if m:
        flush(); cur = m.group(1); buf = []; continue
    if line.strip().startswith("Function :"):
        flush(); cur = None; continue
    if cur:
        i = re.search(r"/\*[0-9a-f]{4,}\*/\s+(.*?);", line)
        if i: buf.append(re.sub(r"_ZN\S+", "SYM", i.group(1)))
flush()' "$2"
}
norm "$W/up.cubin" "$W/up"
norm "$W/mx.cubin" "$W/mx"
n=0; bad=0
fpcount() { sed 's/^@[^ ]* //' "$1" | awk '{print $1}' | grep -E '^(HMMA|FFMA|FMUL|FADD|FMNMX|MUFU|F2F|F2FP|HFMA2|FSETP|FSEL)' | sort | uniq -c; }
for f in "$W"/up/*; do
  k=$(basename "$f"); n=$((n + 1))
  if cmp -s "$f" "$W/mx/$k"; then echo "same      $(wc -l < "$f") instructions  $k"
  elif [ "$(fpcount "$f")" = "$(fpcount "$W/mx/$k")" ]; then
    echo "schedule  $(wc -l < "$f") / $(wc -l < "$W/mx/$k") instructions, the same float / mma operations  $k"
  else echo "DIFF      float / mma operations differ  $k"; bad=$((bad + 1)); fi
done
echo "kernels $n, float / mma differing $bad"
[ "$n" = 6 ] && [ "$bad" = 0 ]
