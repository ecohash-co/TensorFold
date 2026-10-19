#!/usr/bin/env python3
"""Paired long-context replay harness (standard library only).

  run     send a corpus of chat prompts greedily to an OpenAI-compatible engine, record text/usage/latency and, when the
          server supports it, logprobs (top 20) and token ids.  Resumable, sequential, no retries.
  compare compare two run files (A vs B) per item and per bucket; optional pass/fail gate.

Corpus JSONL, one object per line: {id, messages, bucket, est_tokens, [grader], ...}.  Message contents are never printed.

  python3 replay_run.py run --base http://localhost:8080/v1 --model MODEL \
      --corpus corpus.jsonl --out A.jsonl --max-tokens 128 --buckets 8k,32k --limit 10
  python3 replay_run.py compare A.jsonl B.jsonl --baseline A2.jsonl --gate

Thinking is disabled with chat_template_kwargs {enable_thinking: false} (TensorFold also
hears "thinking").  Logprobs: the request is first tried with logprobs+top_logprobs=20 (+return_token_ids); if the server
refuses (HTTP 4xx naming logprobs / token ids) the harness drops those fields, remembers the working tier and records
"logprobs": null.  Refusals are rejected up front, so probing costs no prefill.
"""
import argparse, json, math, os, sys, time, urllib.request, urllib.error, difflib
from collections import defaultdict

TOP_K = 20
# request tiers, richest first: (label, extra body fields)
TIERS = [
    ("logprobs+ids", {"logprobs": True, "top_logprobs": TOP_K, "return_token_ids": True}),
    ("logprobs", {"logprobs": True, "top_logprobs": TOP_K}),
    ("ids", {"return_token_ids": True}),
    ("plain", {}),
]


def norm_base(b):
    b = b.rstrip("/")
    return b if b.endswith("/v1") else b + "/v1"


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass  # torn last line from an interrupted run
    return out


class Refused(Exception):
    pass


def stream_chat(base, body, timeout):
    """One streaming chat call.  Returns dict with text, finish_reason, usage, ttft, seconds, logprob entries, ids."""
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.time()
    ttft = None
    text, fin, usage, lp, ids = [], None, None, [], []
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise Refused(f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}") from None
    with resp:
        ctype = resp.headers.get("Content-Type", "")
        if "event-stream" not in ctype:  # server ignored stream=true
            d = json.load(resp)
            return _from_full(d, time.time() - t0)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ch = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ch.get("error"):
                raise Refused(f"stream error: {str(ch['error'])[:300]}")
            if ch.get("usage"):
                usage = ch["usage"]
            tf = ch.get("tensorfold") or {}  # the Zig engine reports the reply's ids once, on the final chunk
            if not ids and isinstance(tf.get("token_ids"), list):
                ids = list(tf["token_ids"])
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                piece = d.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
                if c.get("finish_reason"):
                    fin = c["finish_reason"]
                l = c.get("logprobs")
                if l and l.get("content"):
                    lp.extend(l["content"])
                for k in ("token_ids", "tokens"):
                    if isinstance(c.get(k), list) and c[k] and isinstance(c[k][0], int):
                        ids.extend(c[k])
    return {"text": "".join(text), "finish_reason": fin, "usage": usage, "ttft": ttft,
            "seconds": time.time() - t0, "lp_content": lp, "ids": ids}


def _from_full(d, secs):
    c = d["choices"][0]
    l = c.get("logprobs") or {}
    ids = c.get("token_ids") if isinstance(c.get("token_ids"), list) else []
    if not ids and isinstance((d.get("tensorfold") or {}).get("token_ids"), list):
        ids = d["tensorfold"]["token_ids"]
    return {"text": (c.get("message") or {}).get("content") or "", "finish_reason": c.get("finish_reason"),
            "usage": d.get("usage"), "ttft": None, "seconds": secs, "lp_content": l.get("content") or [], "ids": ids}


def pack_logprobs(content):
    """Compact form: [{"t": token, "id": int|None, "lp": float, "top": [[token, lp, id|None], ...]}]."""
    out = []
    for e in content:
        top = [[x.get("token"), x.get("logprob"), x.get("token_id", x.get("id"))] for x in (e.get("top_logprobs") or [])]
        out.append({"t": e.get("token"), "id": e.get("token_id", e.get("id")), "lp": e.get("logprob"), "top": top})
    return out


def is_logprob_refusal(msg):
    m = msg.lower()
    return any(k in m for k in ("logprob", "token_ids", "return_token", "top_logprobs", "unsupported", "not supported", "extra"))


def cmd_run(a):
    base = norm_base(a.base)
    corpus = read_jsonl(a.corpus)
    if a.buckets:
        want = set(a.buckets.split(","))
        corpus = [c for c in corpus if c.get("bucket") in want]
    if a.limit:
        corpus = corpus[:a.limit]
    done = {r["id"] for r in read_jsonl(a.out) if not r.get("error")}
    todo = [c for c in corpus if c["id"] not in done]
    print(f"corpus {len(corpus)} items, {len(done)} already done, {len(todo)} to run -> {a.out}", flush=True)
    tier = len(TIERS) - 1 if a.no_logprobs else 0
    force = {r["id"]: r["token_ids"] for r in read_jsonl(a.force_from) if r.get("token_ids")} if a.force_from else {}
    with open(a.out, "a") as out:
        for it in todo:
            body = {"model": a.model, "messages": it["messages"], "max_tokens": a.max_tokens, "temperature": 0,
                    "stream": True, "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}}
            rec = {"id": it["id"], "bucket": it.get("bucket"), "est_tokens": it.get("est_tokens"), "max_tokens": a.max_tokens}
            if a.force_from:  # teacher-forced: score the reference build's exact tokens (engine `forced_tokens`)
                if it["id"] not in force:
                    continue
                body.update(forced_tokens=force[it["id"]], max_tokens=len(force[it["id"]]), ignore_eos=True)
                rec.update(forced=True, max_tokens=len(force[it["id"]]))
            for k in ("task", "grader"):  # carried through so `longsynth.py score OUT.jsonl` works standalone
                if k in it:
                    rec[k] = it[k]
            res = None
            while True:
                try:
                    res = stream_chat(base, {**body, **TIERS[tier][1]}, a.timeout)
                    break
                except Refused as e:
                    if tier < len(TIERS) - 1 and is_logprob_refusal(str(e)) and str(e).startswith("HTTP 4"):
                        tier += 1
                        print(f"  server refused extras; falling back to tier '{TIERS[tier][0]}'", flush=True)
                        continue
                    rec["error"] = str(e)[:400]
                    break
                except Exception as e:  # noqa: BLE001 - timeouts, resets: record, no retry
                    rec["error"] = repr(e)[:400]
                    break
            if res:
                lp = pack_logprobs(res["lp_content"]) if res["lp_content"] else None
                ids = res["ids"] or ([e["id"] for e in lp] if lp and all(e["id"] is not None for e in lp) else None)
                rec.update(text=res["text"], finish_reason=res["finish_reason"], usage=res["usage"],
                           ttft=round(res["ttft"], 3) if res["ttft"] is not None else None,
                           seconds=round(res["seconds"], 3), token_ids=ids or None, logprobs=lp, tier=TIERS[tier][0])
            out.write(json.dumps(rec) + "\n"); out.flush()
            u = rec.get("usage") or {}
            print(f"{rec['id']} bucket={rec['bucket']} prompt_tokens={u.get('prompt_tokens')} "
                  f"completion={u.get('completion_tokens')} ttft={rec.get('ttft')} total={rec.get('seconds')}s "
                  f"tier={rec.get('tier')}" + (f" ERROR {rec['error'][:120]}" if rec.get("error") else ""), flush=True)


# ------------------------------------------------------------------ compare math

def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def first_diff(a, b):
    """Index of first differing element, or None if identical (a shorter prefix diverges at len(shorter))."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def pseudo_tokens(text):
    import re
    return re.findall(r"\w+|[^\w\s]", text or "")


def token_seq(r):
    """Best available token sequence: ids > logprob token strings > regex pseudo-tokens of the text."""
    if r.get("token_ids"):
        return list(r["token_ids"]), "ids"
    if r.get("logprobs"):
        return [e["t"] for e in r["logprobs"]], "lp"
    return pseudo_tokens(r.get("text")), "text"


def norm_edit(sa, sb, n=128):
    a, b = sa[:n], sb[:n]
    m = max(len(a), len(b))
    return 0.0 if m == 0 else levenshtein(a, b) / m


def pos_kl(top_a, top_b):
    """KL(A||B) on the overlapping top-k.  top_* = [[tok, logprob, ...], ...].  Keep tokens present in BOTH lists,
    renormalize each side's probabilities over that intersection, then sum pA*log(pA/pB).  Returns None if <2 shared
    tokens (nothing meaningful to compare).  Approximation: tail mass outside the top-k is ignored, so this is a lower
    bound-ish proxy for the true KL and is insensitive to disagreements about tokens only one side ranks in its top-k."""
    da = {t[0]: t[1] for t in top_a if t[1] is not None}
    db = {t[0]: t[1] for t in top_b if t[1] is not None}
    shared = [t for t in da if t in db]
    if len(shared) < 2:
        return None
    pa = [math.exp(da[t]) for t in shared]; pb = [math.exp(db[t]) for t in shared]
    sa, sb = sum(pa), sum(pb)
    if sa <= 0 or sb <= 0:
        return None
    return sum((x / sa) * math.log((x / sa) / (y / sb)) for x, y in zip(pa, pb) if x > 0 and y > 0)


def lp_stats(la, lb):
    """Positions compared = those where both runs shared the same prefix: 0..d inclusive, d = first sampled-token
    divergence (after d the contexts differ, so distributions are not comparable).  top1_agree = fraction of those
    positions whose argmax (sampled token under greedy) matches.  Returns (top1_rate, mean_kl, max_kl, n_kl) or None."""
    ta = [e["t"] for e in la]; tb = [e["t"] for e in lb]
    d = first_diff(ta, tb)
    last = min(len(la), len(lb)) - 1 if d is None else d
    if last < 0:
        return None
    agree, kls = 0, []
    for i in range(last + 1):
        # argmax of top_logprobs when present (under teacher forcing the emitted tokens are equal by construction)
        ta_, tb_ = la[i].get("top"), lb[i].get("top")
        agree += (max(ta_, key=lambda x: x[1])[0] == max(tb_, key=lambda x: x[1])[0]) if ta_ and tb_ else la[i]["t"] == lb[i]["t"]
        if la[i].get("top") and lb[i].get("top"):
            k = pos_kl(la[i]["top"], lb[i]["top"])
            if k is not None:
                kls.append(max(k, 0.0))
    return agree / (last + 1), (sum(kls) / len(kls) if kls else None), (max(kls) if kls else None), len(kls)


def compare_items(ra, rb):
    ea, eb = ra.get("error"), rb.get("error")
    if ea or eb or ra.get("text") is None or rb.get("text") is None:
        return {"id": ra["id"], "bucket": ra.get("bucket"), "error": True, "a_err": bool(ea), "b_err": bool(eb)}
    ta, tb = ra["text"], rb["text"]
    sa, kind_a = token_seq(ra); sb, kind_b = token_seq(rb)
    kind = kind_a if kind_a == kind_b else "text"
    if kind != kind_a or kind != kind_b:
        sa, sb = pseudo_tokens(ta), pseudo_tokens(tb)
    fc = first_diff(ta, tb)
    ft = first_diff(sa, sb) if kind != "text" else None
    # identical: by token ids when both runs have them (a teacher-forced run renders a forced EOS as text, so its text
    # is longer than the greedy reference's even when every token is the same), else by text
    ident = (list(ra["token_ids"]) == list(rb["token_ids"])) if ra.get("token_ids") and rb.get("token_ids") else ta == tb
    row = {"id": ra["id"], "bucket": ra.get("bucket"), "error": False, "identical": ident, "first_char": fc,
           "first_tok": ft, "tok_kind": kind, "edit128": norm_edit(sa, sb), "top1": None, "mean_kl": None, "max_kl": None}
    if ra.get("logprobs") and rb.get("logprobs"):
        s = lp_stats(ra["logprobs"], rb["logprobs"])
        if s:
            row["top1"], row["mean_kl"], row["max_kl"], _ = s
    return row


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)
    out = {}
    for b, rs in by.items():
        ok = [r for r in rs if not r["error"]]
        kl = [r["mean_kl"] for r in ok if r["mean_kl"] is not None]
        t1 = [r["top1"] for r in ok if r["top1"] is not None]
        out[b] = {"n": len(rs), "compared": len(ok), "errors": len(rs) - len(ok),
                  "identical_rate": (sum(r["identical"] for r in ok) / len(ok)) if ok else None,
                  "mean_edit128": (sum(r["edit128"] for r in ok) / len(ok)) if ok else None,
                  "top1": (sum(t1) / len(t1)) if t1 else None,
                  "mean_kl": (sum(kl) / len(kl)) if kl else None,
                  "max_kl": max([r["max_kl"] for r in ok if r["max_kl"] is not None], default=None)}
    return out


def pair(fa, fb):
    """Every id in either file. An id that one side lacks is an error row in its bucket (counted by the gate's
    missing-result rule), so a candidate that drops records, or a whole bucket, cannot PASS by omission."""
    A = {r["id"]: r for r in read_jsonl(fa)}
    B = {r["id"]: r for r in read_jsonl(fb)}
    rows = []
    for i in list(A) + [i for i in B if i not in A]:
        if i in A and i in B:
            rows.append(compare_items(A[i], B[i]))
        else:
            r = A.get(i) or B[i]
            rows.append({"id": i, "bucket": r.get("bucket"), "error": True, "a_err": i not in A, "b_err": i not in B,
                         "missing": "A" if i not in A else "B"})
    return rows


def gate(summary, baseline, max_drop, max_kl, max_errors=0):
    """Return list of failure strings.  baseline: bucket summary of A-vs-A2 (or None => assumed 1.0 identical)."""
    fails = []
    for b, s in sorted(summary.items(), key=lambda kv: str(kv[0])):
        base = ((baseline or {}).get(b) or {}).get("identical_rate")
        base = 1.0 if base is None else base
        if s["errors"] > max_errors:
            fails.append(f"bucket {b}: {s['errors']} item(s) errored/missing in A or B")
        if s["identical_rate"] is not None and s["identical_rate"] < base - max_drop - 1e-12:
            fails.append(f"bucket {b}: identical-rate {s['identical_rate']:.3f} < baseline {base:.3f} - {max_drop}")
        if s["mean_kl"] is not None and s["mean_kl"] > max_kl:
            fails.append(f"bucket {b}: mean KL {s['mean_kl']:.5f} > {max_kl}")
    return fails


def fmt(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def cmd_compare(a):
    rows = pair(a.a, a.b)
    if not rows:
        sys.exit("no ids in either file")
    if not any(not r["error"] for r in rows) and not a.gate:
        print("note: no id has a result in both files")
    if a.per_item:
        print(f"{'id':<28}{'bucket':<8}{'same':<6}{'1stchr':<8}{'1sttok':<8}{'edit128':<9}{'top1':<7}{'meanKL':<9}{'maxKL'}")
        for r in rows:
            if r["error"]:
                why = f"missing in {r['missing']}" if r.get("missing") else f"A:{r['a_err']} B:{r['b_err']}"
                print(f"{r['id']:<28}{str(r['bucket']):<8}ERROR ({why})"); continue
            print(f"{r['id']:<28}{str(r['bucket']):<8}{str(r['identical']):<6}{str(r['first_char']):<8}{str(r['first_tok']):<8}"
                  f"{fmt(r['edit128']):<9}{fmt(r['top1']):<7}{fmt(r['mean_kl'],5):<9}{fmt(r['max_kl'],4)}")
        print()
    summ = summarize(rows)
    base = None
    if a.baseline:
        base = summarize(pair(a.a, a.baseline))
    print(f"{'bucket':<10}{'n':<5}{'err':<5}{'identical':<11}{'base':<8}{'edit128':<9}{'top1':<7}{'meanKL':<10}{'maxKL'}")
    for b, s in sorted(summ.items(), key=lambda kv: str(kv[0])):
        bb = ((base or {}).get(b) or {}).get("identical_rate")
        print(f"{str(b):<10}{s['n']:<5}{s['errors']:<5}{fmt(s['identical_rate']):<11}{fmt(bb):<8}{fmt(s['mean_edit128']):<9}"
              f"{fmt(s['top1']):<7}{fmt(s['mean_kl'],5):<10}{fmt(s['max_kl'],4)}")
    if a.gate:
        fails = gate(summ, base, a.max_identical_drop, a.max_mean_kl)
        print("GATE:", "FAIL" if fails else "PASS")
        for f in fails:
            print("  -", f)
        sys.exit(1 if fails else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="replay a corpus against an engine")
    r.add_argument("--base", required=True, help="e.g. http://localhost:8080/v1 (/v1 added if missing)")
    r.add_argument("--model", required=True); r.add_argument("--corpus", required=True); r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--buckets", help="comma list of bucket names to include, e.g. 8k,32k")
    r.add_argument("--limit", type=int, help="first N items (after bucket filter)")
    r.add_argument("--timeout", type=float, default=1800, help="per-request socket timeout seconds (default 1800)")
    r.add_argument("--no-logprobs", action="store_true", help="never ask for logprobs / token ids")
    r.add_argument("--force-from", help="reference run JSONL: send each item's token_ids as forced_tokens (+ignore_eos, "
                   "max_tokens=len) so both builds score the SAME continuation; items without ids are skipped")
    r.set_defaults(fn=cmd_run)
    c = sp.add_parser("compare", help="compare two run files",
                      description="Per-bucket comparison of A vs B (every id in either file; one missing on a side is an error).\n\n"
                      "Default gate rule (--gate): FAIL (exit 1) if, for any bucket,\n"
                      "  (1) identical-rate(A,B) < baseline identical-rate - 0.05   (baseline = A vs --baseline A2 run;\n"
                      "      without --baseline the baseline is assumed to be 1.0, i.e. fully deterministic),\n"
                      "  (2) mean per-item KL > 0.01 nats (only when both files have logprobs), or\n"
                      "  (3) any item errored / is missing a result in A or B.\n"
                      "Tune with --max-identical-drop and --max-mean-kl.\n\n"
                      "KL approximation: per position, over positions sharing the same prefix (up to and including the first\n"
                      "diverging token), take tokens in BOTH runs' top-20, renormalize each side over that intersection and\n"
                      "compute KL(A||B). Tail mass outside the top-k is ignored; positions with <2 shared tokens are skipped.\n"
                      "top1 = fraction of those positions where the argmax token agrees.",
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_argument("a"); c.add_argument("b")
    c.add_argument("--baseline", help="second run of build A (A2.jsonl) to measure the engine's own nondeterminism")
    c.add_argument("--gate", action="store_true"); c.add_argument("--per-item", action="store_true")
    c.add_argument("--max-identical-drop", type=float, default=0.05)
    c.add_argument("--max-mean-kl", type=float, default=0.01)
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
