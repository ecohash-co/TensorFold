#!/usr/bin/env python3
"""Stage MX3: bench_mx_zigsplit.py --dump files (torch.save zips: check-L{layer}-R{rows}.pt) -> the flat files
tf-dsv41-mx-bench --dump-check reads (check-L{layer}-R{rows}.bin), with the standard library alone (no torch: the zip's
pickle is read with stand-ins for torch's rebuild functions, the storages are the raw little-endian bytes beside it).

.bin layout (little endian): b"TFMXDMP1", u32 layer, node, rows, items, d, mw, i, dn; per item u32 pair, u32 e, then
f32 x[d], act[mw], xd[i], y[dn] (the sampled pairs' rows exactly as the Python bench saved them: fp32 copies of bf16 x,
bf16 act, bf16 xd and fp32 y); then u32 n and n bytes of JSON: the bench's recorded check dict.
usage: dump_to_bin.py DUMP_DIR OUT_DIR"""
import io
import json
import os
import pickle
import struct
import sys
import zipfile

DTYPES = {"FloatStorage": ("f", 4), "IntStorage": ("i", 4), "LongStorage": ("q", 8), "BFloat16Storage": ("H", 2)}


class StorageType:
    def __init__(self, name):
        self.name = name


class Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)

    def floats(self):
        kind, data = self.storage
        if kind != "FloatStorage":
            raise ValueError(f"expected an fp32 tensor, got {kind}")
        if len(self.size) != 1 or self.stride != (1,):
            raise ValueError(f"expected a contiguous vector, got {self.size} / {self.stride}")
        n = self.size[0]
        return data[4 * self.offset:4 * (self.offset + n)]


def rebuild_tensor_v2(storage, offset, size, stride, *rest):
    return Tensor(storage, offset, size, stride)


class Unpickler(pickle.Unpickler):
    def __init__(self, f, z, prefix):
        super().__init__(f)
        self.z, self.prefix = z, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unexpected class {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, st, key, _loc, _numel = pid
        return (st.name, self.z.read(f"{self.prefix}/data/{key}"))


def load(path):
    z = zipfile.ZipFile(path)
    pkl = next(n for n in z.namelist() if n.endswith("/data.pkl"))
    prefix = pkl[: -len("/data.pkl")]
    return Unpickler(io.BytesIO(z.read(pkl)), z, prefix).load()


def convert(src, dst):
    d = load(src)
    items = d["items"]
    x0 = items[0]
    dims = [len(x0[k].floats()) // 4 for k in ("x", "act", "xd", "y")]
    out = bytearray(b"TFMXDMP1")
    out += struct.pack("<8I", d["layer"], d["node"], d["rows"], len(items), *dims)
    for it in items:
        out += struct.pack("<II", it["pair"], it["e"])
        for k, n in zip(("x", "act", "xd", "y"), dims):
            b = it[k].floats()
            if len(b) != 4 * n:
                raise ValueError(f"{src}: item {it['pair']} {k} has {len(b) // 4} values, not {n}")
            out += b
    js = json.dumps(d["check"]).encode()
    out += struct.pack("<I", len(js)) + js
    with open(dst + ".part", "wb") as f:
        f.write(out)
    os.replace(dst + ".part", dst)
    return {"src": src, "dst": dst, "layer": d["layer"], "node": d["node"], "rows": d["rows"], "items": len(items),
            "dims": dims, "check_pass": d["check"].get("pass")}


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src)):
        if name.startswith("check-") and name.endswith(".pt"):
            print(json.dumps(convert(os.path.join(src, name), os.path.join(out, name[:-3] + ".bin"))))
            n += 1
    if n == 0:
        print(f"no check-*.pt in {src}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
