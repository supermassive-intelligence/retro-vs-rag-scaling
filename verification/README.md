# Verification

Internal checks that our measurement code reproduces published numbers. These
aren't experiment results. They show that `scripts/eval_lm.py` computes
perplexity correctly, so its numbers can be trusted.

## WikiText-2 perplexity (`wikitext_ppl.py`)

Runs `eval_lm`'s own `Scorer` class under the GPTQ protocol (Frantar et al.
2023), which AAAC ([arXiv 2605.08692](https://arxiv.org/abs/2605.08692)) uses
for its reference numbers:

- The full `wikitext-2-raw-v1` test set joined with `"\n\n"`, tokenized
  without filtering.
- Split into non-overlapping 2,048-token segments. Each segment scores its
  last 2,047 tokens.
- ppl = exp(mean of per-segment mean loss), in BF16.

```bash
python -m verification.wikitext_ppl --model Qwen/Qwen2.5-0.5B --model Qwen/Qwen2.5-7B \
  --model Qwen/Qwen2.5-7B-Instruct --out data/verification/wikitext2.json
```

**Result, 2026-09-25**

- **Code:** `f6c1653` plus uncommitted `eval_lm.py` and `wikitext_ppl.py`.
  Their source hashes are in the output file.
- **Hardware:** `blackwell-maxq-0`, GPU 1.
- **Data:** 299,078 tokens, 146 segments.

| Model | Ours | Published (AAAC Table 6, BF16) | Difference |
|---|---|---|---|
| Qwen2.5-0.5B | 13.077 | 13.03 | +0.4% |
| Qwen2.5-7B | 6.850 | 6.80 | +0.7% |
| Qwen2.5-7B-Instruct | 7.459 | none published | 9% above the base model |

Both references are reproduced to within 1%. Small positive gaps like this
typically come from BF16 kernel and attention-implementation differences.
The instruct model scoring 9% above its base model is expected: instruction
tuning moves the model away from raw web text.

**How this relates to our held-out set.** The same Qwen2.5-7B-Instruct scores
ppl 5.740 (0.7196 bpb) on our held-out post-2023 Wikipedia articles, lower
than its 7.46 on WikiText-2. The two setups differ in several ways:

- WikiText-2's raw text keeps artifacts such as ` @-@ ` and spaces before
  punctuation, which raise perplexity for BPE models.
- Our eval gives each scored token 512–1,023 tokens of context, against
  WikiText's 0–2,047.
- The texts themselves are different.

So a lower perplexity on our set doesn't point to an error. The WikiText check
above is the one that validates the code.

A first attempt at the 7B check on GPU 3 ran out of memory while sharing the
GPU with the main RAG run (log: `data/verification/wikitext2.oom-gpu3.log`).

## Retrieval quality (`retrieval_quality.py`)

Checks that dense retrieval finds related text, using relations the index
already knows as ground truth. Chunk queries reuse the stored chunk vectors;
title queries are encoded with the index's own encoder. Each recall is shown
next to what random retrieval would get.

```bash
python -m verification.retrieval_quality --index data/wiki_index/wikipedia_1k \
  --out data/verification/retrieval_quality_wikipedia_1k.json
```

**Result, 2026-09-25**

- **Index:** `wikipedia_1k` (11,251 vectors, 1,000 articles).
- **Queries:** 1,000 per task, seed 0, CPU.

| Task | Recall@1 | Recall@4 | Recall@10 | Random @4 |
|---|---|---|---|---|
| Same article: another chunk of the query's article | 88.4% | 95.7% | 98.0% | 4.3% |
| Next chunk: the exact following chunk | 13.4% | 36.4% | 53.6% | 0.04% |
| Title: the article's title finds one of its chunks | 99.2% | 99.6% | 99.7% | 0.4% |

Retrieval works: every task is far above random. For reference, maailma
reports next-chunk recall of 25.6% at k = 4 for the same encoder, but with
mean pooling on a different corpus. The title task is easy here with only
1,000 articles to choose between.

This rules out a broken retriever as the reason RAG doesn't help on the
held-out set. When related text is in the index, it comes back. The held-out
articles cover post-2023 topics, which the indexes rarely contain.

## Exact GPU search against FAISS (`exact_search_equivalence.py`)

Checks that `scripts/exact_search.py` returns the same neighbors as FAISS
`IndexFlatIP` on the CPU. Queries are built the way `eval_lm` builds them:
the 64 tokens before a target chunk of a held-out article.

```bash
python -m verification.exact_search_equivalence --index data/wiki_index/wikipedia_100k \
  --eval data/eval/wiki_postdump --out data/verification/exact_search_wikipedia_100k.json
```

**Result, 2026-09-28**

- **Index:** `wikipedia_100k` (1,088,070 vectors).
- **Queries:** 5,000, with k = 16.
- **Hardware:** `blackwell-maxq-0`, GPU 3 (RTX PRO 6000 Blackwell Max-Q),
  driver 590.48.01, CUDA 13.1, 64 CPU threads for FAISS. torch 2.14.0+cu130,
  faiss 1.15.1.

| Measure | Value |
|---|---|
| Same top-1 | 99.9% |
| Same top-16 set | 99.7% |
| Same top-16 order | 98.0% |
| Largest score difference (sorted lists) | 1.0e-6 |
| Differing positions | 280 of 80,000 |
| Largest score gap at a differing position | 6.0e-7 |
| FAISS search on CPU | 75.2 s |
| GPU search | 0.18 s (plus 5.5 s to load the vectors once) |

Every differing position is a float32 near-tie: the two candidates' scores
differ by at most 6e-7, a rounding effect of summing in a different order. So
both are exact search, and they differ only in how near-ties are broken. The
GPU version is about 400× faster.

Running `eval_lm --arm rag` on 5 articles with the GPU search gave the same
neighbors in 8 of 8 windows as an earlier FAISS run. Its bpb differed by
8e-5, which comes from bf16 batching, not retrieval: that FAISS run predates
the length-sorted batching.
