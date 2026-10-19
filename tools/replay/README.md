# Paired replay harness (long-context A/B and KL gate)

`replay_run.py` is standard-library Python. It sends a frozen corpus of real chat contexts greedily to any OpenAI-compatible server, then compares two builds item by item and per length bucket.

## Corpus

JSONL, one object per line:
- `id`
- `messages`: OpenAI chat format
- `bucket`: for example `8-16K` or `128-256K`
- `est_tokens`

The harness never prints message contents. Build the corpus from your own agent sessions and keep it private. Only the aggregates need to be shared.

## Run and compare

`compare` pairs every id in either file. An id that one side lacks is an error in its bucket, so `--gate` fails when a run drops records. `replay_run_test.py` covers this (`python -m unittest discover -s tools/replay -p replay_run_test.py`).

```
python3 tools/replay/replay_run.py run --base http://localhost:8080/v1 --model MODEL \
    --corpus corpus.jsonl --out A.jsonl --max-tokens 128 --buckets 8-16K,16-32K
python3 tools/replay/replay_run.py compare A.jsonl B.jsonl --baseline A2.jsonl --gate
```

- **Thinking** is turned off through `chat_template_kwargs`.
- **Logprobs** are requested in the richest tier the server accepts. On `tensorfold serve` today that is `logprobs`+`top_logprobs`. The tiers are, in order: `logprobs`+`top_logprobs=20`+`return_token_ids`, then `logprobs`, then `ids`, then plain. If the server refuses one, the harness falls back to the next.
- **Resumable:** an interrupted run picks up where it stopped.

## Teacher-forced KL (numerics changes)

```
python3 tools/replay/replay_run.py run ... --out B-forced.jsonl --force-from A.jsonl
```

- The candidate scores the reference build's exact greedy tokens. It does this with a non-standard `forced_tokens` request field plus `logprobs`. `tensorfold serve` does not have `forced_tokens`; our DeepSeek TP4 engine does (fork branch `dsv41-tp4-logprobs`). Without it, use `run` and `compare` alone.
- `compare` reports, per bucket, top-1 agreement and mean and max KL over the shared top-20.
- **Why not plain text identity:** greedy text identity is not a usable gate for numerics changes, because long continuations drift even when the per-token distributions agree.

## Our calibration

These are our reference numbers from 4× GB10 TP4, on 60 private agent contexts from 8K to 256K, 12 per bucket. They compare P16 (bf16 owner partials, adopted) against the exact pre-P16 build, teacher-forced.

| bucket | top-1 agreement | mean KL (nats) | max KL |
|---|---|---|---|
| 8–16K | 0.986 | 0.0065 | 1.07 |
| 16–32K | 0.986 | 0.0040 | 0.16 |
| 32–64K | 0.977 | 0.0104 | 1.34 |
| 64–128K | 0.982 | 0.0059 | 1.64 |
| 128–256K | 0.988 | 0.0053 | 0.55 |

The same two builds, run free (not forced), produced byte-identical greedy replies on only 25–42% of items per bucket (8–64K).

Our pre-declared gate for any later numerics change is a mean KL of ≤ 0.01 per bucket, plus identical-rate rules. The MXFP4-on-long-prompts path failed it: mean KL was 0.0257 at 32–64K and 0.0134 at 64–128K.
