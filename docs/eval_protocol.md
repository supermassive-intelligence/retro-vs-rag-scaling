# Language-modeling evaluation protocol

How we measure whether retrieval helps a language model predict held-out
text, and how that changes with datastore size. **We use Protocol B.**
Protocol A is described for context, because it is what the closest
datastore-scaling paper (MassiveDS) uses.

Running example: one held-out article of 2,048 tokens, numbered 0–2047.

## Protocol A — MassiveDS (not used)

Shao et al. 2024, *Scaling Retrieval-Based Language Models with a
Trillion-Token Datastore* ([arXiv 2407.12854](https://arxiv.org/abs/2407.12854)),
§4.1 and Appendix B.2.

**1. Slide a 1,024-token window along the article, moving 512 tokens each step.**

```
tokens:    0 ------ 511 | 512 ----- 1023 | 1024 ---- 1535 | 1536 ---- 2047
window 1:  [  prefix    |    target     ]
window 2:               [    prefix     |     target     ]
window 3:                                [     prefix     |     target     ]
```

**2. In each window, retrieve once.** The first 512 tokens (the prefix) are
the query, and we take the top-k chunks. MassiveDS uses k = 3.

**3. Run one forward pass over:**

```
[chunk 3][chunk 2][chunk 1][ prefix: 512 tokens ][ target: 512 tokens ]
```

The best chunk goes closest to the text.

**4. The loss covers only the 512 target tokens.** Each target token is
predicted from everything before it: the retrieved chunks, the prefix, and
the earlier target tokens. The retrieved chunks and the prefix are context
only and are never scored.

Because the window moves 512 tokens at a time, each target half is scored
exactly once: window 1 scores 512–1023, window 2 scores 1024–1535, and
window 3 scores 1536–2047. Tokens 0–511 are never scored, for every arm
alike. The sliding exists so that every scored token has at least 512 tokens
of real text before it.

## Protocol B — RETRO (chosen)

Borgeaud et al. 2022, *Improving language models by retrieving from trillions
of tokens* ([arXiv 2112.04426](https://arxiv.org/abs/2112.04426)).

This protocol has **two separate strides**:

- The **evaluation window** is the same as in Protocol A: 1,024 tokens moving
  512 at a time. The first 512 tokens (the prefix) are context, and the last
  512 (the target) are scored.
- The **retrieval stride** is 64 tokens. Inside the target, retrieval happens
  again for every 64-token chunk.

Split the target into 8 chunks, T1…T8. Call the last 64 tokens of the prefix T0.

```
... prefix ...[ T0 ][ T1 ][ T2 ][ T3 ] ... [ T8 ]
                 │     ▲
                 └─────┘  query with T0 → retrieve top-k → used while predicting T1
                       │     ▲
                       └─────┘  query with T1 → retrieve → used while predicting T2
```

To predict chunk Ti, we query with the chunk just before it (Ti-1). That text
is already known when Ti is being predicted, so nothing from the future leaks
in.

**How the loss is computed:**

- **RAG arm:** each target chunk gets its own forward pass, because its
  retrieved chunks are different:

  ```
  [chunks retrieved with Ti-1][ prefix + T1 … Ti-1 ][ Ti ]   → loss on the 64 tokens of Ti only
  ```

  That's 8 forward passes per window, one per target chunk.
- **GCCA arm:** a single forward pass over the whole window. The tokens of
  each Ti cross-attend to the chunks retrieved with Ti-1, which is exactly
  maailma's `retrieval_plan`. The loss covers the same 512 target tokens.
- **LM-only baseline:** the same windows and target tokens, with no retrieval.

**Why B.** Both arms score the same tokens with the same retrieved chunks. The
only difference is whether the chunks sit in the prompt or go through
cross-attention, so a RAG-vs-GCCA gap measures the injection mechanism rather
than how often each arm retrieves. This is the reasoning behind maailma's arm C
(`evals/arms.py`). Protocol A retrieves once per window while GCCA retrieves
every 64 tokens, which would confound the two.

## Metric

Sum the loss over every scored token, across all windows and articles:

- **bits-per-byte (bpb)**, the main number:
  total loss in nats ÷ (ln 2 × UTF-8 bytes of all scored text).
  Bytes don't depend on the tokenizer, so the number is comparable to RETRO,
  REPLUG and the Pile (Gao et al. 2020, [arXiv 2101.00027](https://arxiv.org/abs/2101.00027)).
- **perplexity**, reported alongside: exp(total loss ÷ number of scored tokens).

## Held-out evaluation text

The datastore is `wikimedia/wikipedia` `20231101.en`, built from the official
Wikimedia dump of **2023-11-01**. That date is the datastore's knowledge
cutoff.

Evaluation uses **English Wikipedia articles created after 2023-11-01**, so no
eval article can be in any index, at any size. If eval articles were drawn
from the same dump, retrieval would just find the exact continuation, and the
scaling curve would only track the chance that the article was sampled.

- Source: a newer snapshot of the same kind of data, `omarkamali/wikipedia-monthly`
  (official dump, parsed with `mwparserfromhell`).
- Candidates: page IDs greater than the largest page ID in `20231101.en`.
  Wikipedia assigns page IDs at creation, so these pages are newer.
- Verification: each sampled article's creation date (its first revision) is
  checked with the Wikipedia API, and articles created before the cutoff are
  dropped.

Built by `scripts/build_heldout_eval.py` into `data/eval/wiki_postdump/`.
Articles need at least 1,024 tokens, one full evaluation window.

What the build produced, with its checks, is in
[`heldout_eval_set.md`](heldout_eval_set.md).

Its caveats (base-model exposure, text-cleaning differences, longer articles
only) are listed there too.

Fallback, if post-dump articles prove unusable: cut the datastore off five
years earlier. Index only articles created before 2018-11-01, and evaluate on
articles created between then and 2023-11-01, all from the same dump.

## Leakage control

A new article can still copy text from older ones, for example when it is
split off from an existing article. Following MassiveDS, drop any retrieved
chunk that shares 80% or more of its 13-grams with the target (Jaccard
similarity) or contains 32 or more consecutive tokens of it. Log how many
chunks are dropped at each index size.

## Other protocols in the literature

| Paper | Retrieval query | How often it retrieves | Tokens scored | Metric |
|---|---|---|---|---|
| RETRO (2112.04426) | previous 64-token chunk | every 64 tokens | whole document (2,048 windows, 1,024 stride) | bpb |
| In-Context RALM (2302.00083) | last 32 tokens | every 4 tokens | whole sequence, 1,024 tokens | perplexity |
| REPLUG (2301.12652) | 128-token context | once | the continuation | bpb |
| SILO (2308.04430) | prefix (BM25) | once per window | 1,024 windows, 512 stride | perplexity |
| MassiveDS (2407.12854) | 512-token prefix | once per window | last 512 tokens | perplexity |
