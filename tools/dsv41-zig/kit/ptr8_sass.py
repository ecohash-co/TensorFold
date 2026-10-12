#!/usr/bin/env python3
"""For every captured Triton variant (aot.json + cubins/) and every *i64 parameter it specialized as 16-aligned:
does the code read/write through that pointer only with accesses of at most 8 bytes (or not at all)?

Conservative taint analysis on cuobjdump -sass: registers loaded from the parameter's constant-bank slot are
tainted; taint flows (to a fixpoint, never cleared) through every non-memory instruction from any tainted source to
all its destinations, and through local/shared/generic spills (any tainted store taints every later load's
destination). A memory access whose address uses a tainted register must be <= 64 bits wide. Output: JSON
{hash: {"fn":..., "params": {name: "safe"|"unsafe:<insn>"|"unused"}}}."""
import json, os, re, subprocess, sys

# usage: ptr8_sass.py KIT_AOT_DIR [ZIG_OUT]  (JSON verdicts on stdout; with ZIG_OUT also the Zig table)
kit = sys.argv[1]
d = json.load(open(os.path.join(kit, "aot.json")))
PBASE = 0x380
SIZES = {"i32": 4, "fp32": 4, "u64": 8}
REG = re.compile(r"\b(U?R)(\d+)\b")
CB = re.compile(r"c\[0x0\]\[(0x[0-9a-f]+)\]")
MEM_LD = ("LDG", "LD", "LDL", "LDS", "LDGSTS", "ATOMG", "ATOM", "ATOMS", "RED", "REDG", "LDSM")  # shared ones: spill tracking only
MEM_ST = ("STG", "ST", "STL", "STS")

def layout(params):
    off, out = PBASE, {}
    for p in params:
        sz = SIZES.get(p["type"], 8)
        off = (off + sz - 1) // sz * sz
        out[p["name"]] = (off, sz)
        off += sz
    return out

def sass(path):
    txt = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", path], capture_output=True, text=True, check=True).stdout
    ins = []
    for line in txt.splitlines():
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
        if m: ins.append((int(m.group(1), 16), m.group(2)))
    return ins

def regs_of(tok, wide):
    out = []
    for kind, n in REG.findall(tok):
        n = int(n)
        out.append(f"{kind}{n}")
        if wide: out += [f"{kind}{n+k}" for k in range(1, wide)]
    return out

def width(op):
    for w, n in ((".128", 4), (".256", 8), (".64", 2)):
        if w in op: return n
    return 1

def decode(s):
    pred = re.match(r"^@!?U?P\w+\s+", s)
    s2 = s[pred.end():] if pred else s
    parts = s2.split(None, 1)
    op = parts[0]; rest = parts[1] if len(parts) > 1 else ""
    return bool(pred), op, rest

def analyze(ins, off, sz):
    """Flow-sensitive may-taint over the SASS control flow (gen/kill per instruction; a predicated write kills
    nothing; spills: any tainted store to local/shared/generic memory taints every load from them)."""
    addr_ix = {a: i for i, (a, _) in enumerate(ins)}
    n = len(ins)
    succ = []
    calls, rets = [], []
    for i, (a, s) in enumerate(ins):
        pr, op, rest = decode(s)
        base = op.split(".")[0]
        nx = [i + 1] if i + 1 < n else []
        if base in ("BRA", "JMP"):
            m = re.search(r"0x([0-9a-f]+)", rest)
            t = addr_ix.get(int(m.group(1), 16)) if m else None
            if t is None: return "unsafe:unresolved " + s   # an indirect branch: no claim
            nx = ([t] + nx) if pr else [t]
        elif base == "EXIT" and not pr:
            nx = []
        elif base == "CALL" and ".REL" in op:
            # a direct call (Triton's division subroutines): into the callee, and on after it (RET below)
            m = re.search(r"0x([0-9a-f]+)", rest)
            t = addr_ix.get(int(m.group(1), 16)) if m else None
            if t is None: return "unsafe:unresolved " + s
            calls.append(i)
            nx = [t] + nx
        elif base == "RET":
            rets.append((i, pr))
            nx = None
        elif base in ("BRX", "JMX", "CALL"):
            return "unsafe:indirect " + s
        succ.append(nx)
    for i, pr in rets:
        # a return goes back after every call site (a may-analysis: the union), or on when predicated off
        succ[i] = [c + 1 for c in calls if c + 1 < n] + ([i + 1] if pr and i + 1 < n else [])
    state = [None] * n
    state[0] = frozenset()
    work = [0]
    spilled = [False]
    verdict = {"bad": None, "used": False}
    def step(i, t):
        s = ins[i][1]
        pr, op, rest = decode(s)
        base = op.split(".")[0]
        ops = [o.strip() for o in rest.split(",")] if rest else []
        w = width(op)
        t = set(t)
        reads_slot = False
        for cm in CB.finditer(rest):
            a = int(cm.group(1), 16)
            span = 16 if ".128" in op else (8 if ".64" in op else 4)
            if a < off + sz and off < a + span: reads_slot = True
        if base in MEM_LD or base in MEM_ST:
            groups = re.findall(r"\[([^\[\]]*)\]", rest)
            addr_txt = groups[-1] if groups else ""
            if base in MEM_ST:
                addr_txt = groups[-1] if groups else ""
            shared = base in ("LDS", "STS", "LDSM", "ATOMS")
            if base == "LDGSTS" and len(groups) >= 2: addr_txt = groups[-1]   # its global source
            if not shared and any(r in t for r in regs_of(addr_txt, 0)):
                verdict["used"] = True
                if w > 2 and verdict["bad"] is None: verdict["bad"] = s
            if base in MEM_ST:
                if any(r in t for o in ops[1:] for r in regs_of(o, 0)): spilled[0] = True
                return frozenset(t)
            dest = ops[0] if ops else ""
            dr = regs_of(dest, w) if not dest.startswith("[") else []
            if not pr:
                for r in dr: t.discard(r)
            if base in ("LDL", "LDS", "LD") and spilled[0]:
                t.update(dr)
            return frozenset(t)
        if not ops: return frozenset(t)
        dest, srcs = ops[0], ops[1:]
        dw = 4 if ".128" in op else (2 if (".64" in op or "WIDE" in op) else 1)
        dr = regs_of(dest, dw)
        if base.startswith("ISETP") or base.startswith("FSETP") or base.startswith("PLOP") or base.startswith("UISETP"):
            dr = []
        src_t = reads_slot or any(r in t for o in srcs for r in regs_of(o, 0))
        if not pr:
            for r in dr: t.discard(r)
        if src_t: t.update(dr)
        return frozenset(t)
    rounds = 0
    while True:
        before = spilled[0]
        while work:
            i = work.pop()
            out = step(i, state[i])
            for j in succ[i]:
                if state[j] is None or not out <= state[j]:
                    state[j] = out if state[j] is None else (state[j] | out)
                    work.append(j)
        if spilled[0] == before: break
        work = [i for i in range(n) if state[i] is not None]
        rounds += 1
    # one final pass over every reached instruction (the verdicts of the fixpoint states)
    verdict["bad"] = None; verdict["used"] = False
    for i in range(n):
        if state[i] is not None: step(i, state[i])
    if verdict["bad"]: return "unsafe:" + verdict["bad"]
    return "safe" if verdict["used"] else "unused"

res = {}
for k in d["kernels"]:
    targets = [p for p in k["params"] if p["type"] == "*i64" and p["div16"]]
    if not targets: continue
    ins = sass(os.path.join(kit, "cubins", k["hash"] + ".cubin"))
    lay = layout(k["params"])
    res[k["hash"]] = {"fn": k["fn"], "params": {p["name"]: analyze(ins, *lay[p["name"]]) for p in targets}}
json.dump(res, sys.stdout, indent=1)

HDR = """//! Generated by tools/dsv41-zig/kit/ptr8_sass.py from the kit (aot.json + cubins, cuobjdump -sass of each cubin);
//! do not edit by hand. For each captured variant: the *i64 parameters it specialized as 16-aligned whose code reads
//! or writes through them only with accesses of at most 8 bytes, or not at all (a conservative flow-sensitive taint of
//! the registers loaded from the parameter's constant-bank slot through every non-memory instruction, direct calls
//! and spills; any wider global access through a tainted address, or an indirect branch, leaves the parameter out).
//! Triton's 16-alignment of a pointer only widens the accesses through it; with none wider than 8 bytes the same
//! code at an 8-aligned int64 address loads and stores the same bytes. aot.Relax.proven: a launch with no exact
//! variant and none in upstream's fallback may run such a variant with these pointers 8-aligned. Another kit's
//! cubins have other hashes: nothing here applies to them.
const aot = @import("cuda").aot;

pub const variants = [_]aot.Relax.Proven{
"""
if len(sys.argv) > 2:
    rows = []
    for h in sorted(res):
        v = res[h]
        ok = [p for p, s in v["params"].items() if s in ("safe", "unused")]
        if ok:
            rows.append('    .{ .hash = "%s", .ptrs8 = &.{ %s } }, // %s' % (h, ", ".join('"%s"' % p for p in ok), v["fn"]))
    with open(sys.argv[2], "w") as f:
        f.write(HDR + "\n".join(rows) + "\n};\n")
