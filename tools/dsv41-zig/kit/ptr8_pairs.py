#!/usr/bin/env python3
"""Cross-check of ptr8_sass.py (after it wrote /tmp/ptr8.json): captured variant pairs that differ only in int64 pointers 16-alignment;
where the analysis proves the pointers, the two cubins SASS must be identical (no wider access depends on the hint)."""
import json, os, re, subprocess, hashlib, itertools
kit=os.path.expanduser("~/zigdsv/kit/aot")
d=json.load(open(kit+"/aot.json")); r=json.load(open("/tmp/ptr8.json"))
def md5(h):
    t=subprocess.run(["/usr/local/cuda/bin/cuobjdump","-sass",f"{kit}/cubins/{h}.cubin"],capture_output=True,text=True).stdout
    ins=[re.sub(r"/\*[0-9a-f]+\*/","",l).split(";")[0].strip() for l in t.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/",l)]
    return hashlib.md5("\n".join(ins).encode()).hexdigest()
agree=dis=0
for a,b in itertools.combinations(d["kernels"],2):
    if a["fn"]!=b["fn"] or a["consts"]!=b["consts"] or len(a["params"])!=len(b["params"]): continue
    diff=[(pa["name"],pa["div16"]) for pa,pb in zip(a["params"],b["params"]) if pa!=pb]
    if not diff: continue
    if not all(pa["name"]==pb["name"] and pa["type"]=="*i64" and pa["div16"]!=pb["div16"] for pa,pb in zip(a["params"],b["params"]) if pa!=pb): continue
    A,B=(a,b) if a["params"][[p["name"] for p in a["params"]].index(diff[0][0])]["div16"] else (b,a)
    names=[n for n,_ in diff]
    same = md5(A["hash"])==md5(B["hash"])
    claim = all(r.get(A["hash"],{}).get("params",{}).get(n,"x") in ("safe","unused") for n in names)
    print(A["fn"],A["hash"][:10],B["hash"][:10],names,"sass-identical" if same else "sass-differs","proven" if claim else "not-proven")
    if same==claim: agree+=1
    else: dis+=1
print("agree",agree,"disagree",dis)
