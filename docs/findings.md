# Findings so far: RAG (Protocol B) on held-out Wikipedia

**Status: working notes, 2026-09-30.** Every number below comes from a run
whose `summary.json` records its command, commit and inputs, at the paths in
[Provenance](#provenance). The tables were copied from those files by hand.
CLAUDE.md asks for result tables generated from raw outputs, so a generated
results page is still to do.

## Setup

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, bf16.
- **Evaluation:** Protocol B ([`eval_protocol.md`](eval_protocol.md), math in
  [`protocol_b_math.md`](protocol_b_math.md)).
  - Windows are 1,024 tokens with a 512-token stride, and each window's last
    512 tokens are scored.
  - For RAG, each 64-token chunk is predicted with neighbors retrieved using
    the chunk before it.
- **Eval text:** Wikipedia articles created after the datastore's 2023-11-01
  dump ([`heldout_eval_set.md`](heldout_eval_set.md)). Main runs use the
  first 500 of the 1,000 articles: 2,165 windows and 1,108,480 scored tokens
  (4,034,153 bytes).
- **Datastores:**
  - `wikipedia_1k`: 1,000 articles, 11,251 chunks, 0.69M tokens.
  - `wikipedia_100k`: 100,000 articles, 1,088,070 chunks, 66.5M tokens.
  - Both are random samples of the 2023-11-01 dump, and the 1k articles are
    the first 1,000 of the 100k.
- **Retrieval:**
  - BGE-small with CLS pooling, and exact inner-product search.
  - Each neighbor is its chunk plus the next chunk of the same article (up
    to 128 tokens).
  - Neighbors sharing 32 or more consecutive tokens with the target are
    dropped.

## 1. RAG doesn't help, and a 100× bigger index doesn't change that

Main runs: 500 articles, commit `1311aee` (clean), GPUs 2–3 of
`blackwell-maxq-0`.

| Datastore | k | bpb | ppl | Δbpb vs no retrieval | Chunks improved | Run time |
|---|---|---|---|---|---|---|
| none (LM only) | 0 | **0.72149** | 6.172 | – | – | 3 min |
| `wikipedia_1k` | 1 | 0.72460 | 6.221 | +0.0031 | 39.7% | 18 min |
| `wikipedia_1k` | 4 | 0.72588 | 6.241 | +0.0044 | 37.6% | 25 min |
| `wikipedia_100k` | 1 | 0.72461 | 6.221 | +0.0031 | 39.9% | 18 min |
| `wikipedia_100k` | 4 | 0.72594 | 6.242 | +0.0045 | 37.4% | 28 min |

"Chunks improved" is the share of the 17,320 scored 64-token chunks whose
loss went down with retrieval.

The same pattern shows on all 1,000 articles:

| Datastore | k | bpb | ppl | Δbpb |
|---|---|---|---|---|
| none | 0 | 0.71958 | 5.740 | – |
| `wikipedia_1k` | 4 | 0.72369 | 5.798 | +0.0041 |
| `wikipedia_100k` | 4 | 0.72360 | 5.797 | +0.0040 |

The no-retrieval baseline on 1,000 articles was rerun at `1311aee` and gave
exactly the same numbers, 0.71958 bpb and 5.7403 ppl. On 1,000 articles, 38%
of chunks improved with RAG at either index size. The per-chunk change
(10th to 90th percentile) spread from −2.4 to +4.0 nats with the 1k index,
and from −2.6 to +4.2 with the 100k index. Numbers from the 500
and 1,000 sets are not comparable with each other.

**What the k = 1 runs add.**
- Cutting the retrieved text 4× (from 4 neighbors to 1) cuts the penalty by
  only about 30%, from +0.0044 to +0.0031. So most of the cost comes from the
  first neighbor, not from the amount of retrieved text. Even the single best
  match hurts on average.
- At k = 1 the two index sizes agree to 5 decimal places. The 100k index
  doesn't find better top matches for these articles.

## 2. Retrieval itself works

How each metric is computed is in [`retrieval_metrics.md`](retrieval_metrics.md).

**The search is exact.** It is brute force over every vector, so its recall
of the true nearest neighbors is 100% by construction. Compared with FAISS
`IndexFlatIP` on `wikipedia_100k` (5,000 eval-style queries, k = 16):

- the same top-1 in 99.9% of queries,
- the same top-16 set in 99.7%,
- and every difference was a float32 near-tie (score gap ≤ 6×10⁻⁷).

**Retrieved text is related when related text exists.** On `wikipedia_1k`,
with 1,000 queries per task:

| Task | Recall@1 | Recall@4 | Recall@10 | Random @4 |
|---|---|---|---|---|
| Same article | 88.4% | 95.7% | 98.0% | 4.3% |
| Next chunk | 13.4% | 36.4% | 53.6% | 0.04% |
| Title → article | 99.2% | 99.6% | 99.7% | 0.4% |

**For the held-out eval queries there is no ground truth:** those articles
aren't in the index, by design. As a proxy, only 11.1% (1k) and 14.7% (100k)
of their neighbors come from an article whose title shares a word with the
query article's title. That was measured on the 1,000-article, k = 4 runs.

## 3. Controls: retrieval works when related text exists

Runs at commit `eca9cf9` (clean), GPUs 1–3 of `blackwell-maxq-0`, k = 4.
Intervals are 95% paired-bootstrap intervals over articles
(`scripts/bootstrap_ci.py`, 10,000 resamples): each resample redraws the 500
articles with replacement and recomputes every run's bpb on that same draw.

**Held-out articles (post-2023), 500 articles:**

| Run | bpb | Δbpb vs none | 95% CI |
|---|---|---|---|
| none | 0.72149 | – | – |
| random chunks from `wikipedia_100k` | 0.72651 | +0.00501 | [+0.00459, +0.00542] |
| dense `wikipedia_1k` | 0.72588 | +0.00439 | [+0.00365, +0.00505] |
| dense `wikipedia_100k` | 0.72594 | +0.00444 | [+0.00368, +0.00511] |

- The RAG penalty is real: its interval excludes zero.
- `wikipedia_100k` − `wikipedia_1k` = +0.00005 [−0.00033, +0.00043]. The two
  index sizes are indistinguishable.
- Retrieved text beats random text by only 0.0006 [0.0000, 0.0012]. Almost
  all of the penalty is the cost of inserting *any* text, and the retrieved
  text carries almost no information about these articles.

**In-dump articles (positive control), 500 articles.** These articles are in
`wikipedia_100k` but not in `wikipedia_1k` (`data/eval/wiki_indump`, built by
`scripts/build_indump_eval.py`). The leakage filter still drops any neighbor
sharing 32 tokens with the window being scored, so what `wikipedia_100k` can
offer is the article's *other* sections.

| Run | bpb | Δbpb vs none | 95% CI |
|---|---|---|---|
| none | 0.69711 | – | – |
| random chunks from `wikipedia_100k` | 0.70209 | +0.00497 | [+0.00462, +0.00533] |
| dense `wikipedia_1k` (article absent) | 0.70133 | +0.00422 | [+0.00381, +0.00463] |
| dense `wikipedia_100k` (article present, filtered) | **0.68909** | **−0.00802** | [−0.01018, −0.00606] |
| dense `wikipedia_100k`, no leakage filter | 0.01094 | −0.68618 | [−0.70788, −0.66214] |

- **RAG helps when the index holds related text:** −0.008 bpb (−1.2%) against
  no retrieval, and −0.013 against random text.
- **The pipeline is sound:** with the filter off, the top neighbor contains the
  target itself and the model copies it (0.011 bpb). So the prompt layout,
  the neighbor order and the instruct model all work.
- The filter dropped 29,376 of the 82,071 candidate spans examined for
  `wikipedia_100k`, mostly the article's own chunks overlapping the scored
  window. 38 of the 13,192 scored chunks had fewer than 4 neighbors left.
- With the article absent (`wikipedia_1k`), the same text behaves like the
  held-out set: +0.0042, close to random.

## Interpretation

The controls in section 3 separate the two explanations:

1. **Coverage is the problem.** On held-out articles, retrieved text is barely
   better than random text, because the datastores hold almost nothing on
   these topics. When related text is present (in-dump), RAG gives a clear gain.
2. **Disruption is real but small.** Inserting four unrelated spans costs about
   +0.005 bpb on either eval set. Retrieval has to beat this before it shows
   a net gain.

For the scaling study this means the held-out curve can only rise once the
datastore starts to cover post-2023 topics. A datastore built from the same
2023 dump may never do that, however large it gets.

Our datastores (0.7M and 66.5M tokens) are far smaller than MassiveDS's
smallest (billions of tokens), where gains began. MassiveDS also evaluated on
text from the same domain as its datastore.

## Next: what the two plots need

### bpb against datastore size

Already available: bpb per (index size, k) plus the no-retrieval line. The x
axis is articles; chunks and tokens are in each index manifest.

Still needed:

1. **More index sizes.** Two points aren't a curve. 10⁴ and 10⁶ articles
   would give four points across three orders of magnitude, and all 6.4M
   articles would be the endpoint. Nested builds keep each smaller index a
   prefix of the next.
2. **Confidence intervals.** Done (section 3): differences under about
   ±0.0004 bpb between runs on the same 500 articles are noise.
3. **Optionally, several datastore samples per size.** MassiveDS used 3
   seeds. Our indexes are one nested sample.

### Recall against bpb

The catch is that recall needs ground truth, and the held-out queries have
none. Workable options:

1. **Intrinsic recall at each index size.** Run `retrieval_quality.py` on
   every index, then plot Δbpb against Recall@k with one point per index
   size. It's cheap, but gives only as many points as there are sizes.
2. **Per-chunk retrieval score.** For every scored chunk, compute the cosine
   similarity of its neighbors and plot the chunk's loss change against it
   (binned). The queries are deterministic, so the scores can be recomputed
   from the saved neighbor IDs without rerunning the language model.
3. **Per-chunk overlap with the target.** Measure how much of the target
   chunk's text appears in its neighbors: the longest common token run, as
   in RETRO's leakage analysis. This can be computed post hoc from saved
   outputs.
4. **Title-overlap proxy per chunk.** A cruder relevance label, also post
   hoc.

Options 2–4 turn the ~17,000 scored chunks into points, which is enough to
show whether better retrieval means lower loss. Option 1 connects the plot to
datastore size.

## Provenance

All runs are on `blackwell-maxq-0` under `/home/normal/sai/retro-vs-rag-scaling`.
Each output directory holds `summary.json` (command, git state, code hash,
inputs) and `windows.jsonl`.

| Result | Commit | Output |
|---|---|---|
| 500 articles: none, RAG 1k/100k at k = 1 and k = 4 | `1311aee`, clean | `data/lm_eval/1311aee/eval500/qwen2.5-7b-instruct/{none,rag_wikipedia_1k,rag_wikipedia_1k_k1,rag_wikipedia_100k,rag_wikipedia_100k_k1}` |
| 1,000 articles: none (rerun) | `1311aee`, clean | `data/lm_eval/1311aee/qwen2.5-7b-instruct/none` |
| 1,000 articles: none (original) | `f6c1653` + uncommitted `eval_lm.py` with SHA-256 `3a233912…`, which is the file committed in `73a09c5` | `data/lm_eval/qwen2.5-7b-instruct/none` |
| 1,000 articles: RAG 1k | recorded as `73a09c5`, but started before that commit, with the same `eval_lm.py` hash | `data/lm_eval/qwen2.5-7b-instruct/rag_wikipedia_1k` |
| 1,000 articles: RAG 100k | `73a09c5`, clean | `data/lm_eval/qwen2.5-7b-instruct/rag_wikipedia_100k` |
| Retrieval recall | `73a09c5` + uncommitted script (hash in output) | `data/verification/retrieval_quality_wikipedia_1k.json` |
| Controls: random neighbors (held-out), all in-dump runs | `eca9cf9`, clean | `data/lm_eval/eca9cf9/{eval500,indump500}/qwen2.5-7b-instruct/*` |
| In-dump eval set | `eca9cf9`, clean | `data/eval/wiki_indump` |
| Bootstrap intervals | `eca9cf9`, clean | `data/analysis/bootstrap_{eval500_1311aee,eval500_random,indump500}.json` |
| GPU search vs FAISS | `7102b4f` + uncommitted script (hash in output) | `data/verification/exact_search_wikipedia_100k.json` |

Command for the main runs, with `--arm`, `--index` and `--k` varied:

```bash
.venv/bin/python -m scripts.eval_lm --arm rag --k 1 --index data/wiki_index/wikipedia_100k \
  --model Qwen/Qwen2.5-7B-Instruct --eval data/eval/wiki_postdump --max-articles 500 \
  --batch-size 32 --out data/lm_eval/1311aee/eval500/qwen2.5-7b-instruct/rag_wikipedia_100k_k1
```
