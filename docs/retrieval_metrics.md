# How retrieval recall is computed

Three different quantities get called "recall" in this project. They answer
different questions, and only the first two are measured today.

| Quantity | Question it answers | Script |
|---|---|---|
| Search exactness | Did the search return the true nearest neighbors? | `verification/exact_search_equivalence.py` |
| Retrieval recall@k | Is the nearest text actually related? | `verification/retrieval_quality.py` |
| Eval-query relatedness (proxy) | For the held-out queries we score, are the neighbors on topic? | post-hoc analysis of `eval_lm` outputs |

Results are in [`findings.md`](findings.md) and [`../verification/README.md`](../verification/README.md).

## Notation

| Symbol | Meaning |
|---|---|
| $c_1, \dots, c_M$ | the index's chunks, numbered in article order (`chunk_id`) |
| $\operatorname{doc}(c)$ | the article a chunk belongs to |
| $e(\cdot)$ | BGE-small, CLS pooling, L2-normalized |
| $R_k(q)$ | the $k$ chunks with the highest $e(q)^\top e(c)$, best first |

## 1. Search exactness

Exact search compares the query with every vector, so it returns the true
top $k$ by definition: recall of the true nearest neighbors is **100%**.
What we check is that our GPU implementation computes the same thing as
FAISS `IndexFlatIP`, the reference exact search, on the same queries:

- **Identical top-1:** the share of queries where both return the same best chunk.
- **Identical set / order:** the share of queries where the top-$k$ lists
  contain the same chunks, or the same chunks in the same order.
- **Score gap at a mismatch:** wherever the two lists differ at a position,
  the absolute difference of the two scores there.

A mismatch with a gap near float32 rounding (about 10⁻⁶) is a tie broken
differently, not a missed neighbor. The queries are built the way `eval_lm`
builds them: the 64 tokens before a target chunk of a held-out article,
decoded and encoded.

This check is unrelated to approximate search (HNSW, IVF, PQ). For those,
"recall@k" would mean $|R^{\text{approx}}_k \cap R^{\text{exact}}_k| / k$, and it
would have to be reported at every index size. We don't use them.

## 2. Retrieval recall@k

Recall needs a known correct answer for each query. Our indexes have no
relevance labels, so we use relations the index already knows. Each task
defines a query and a set $G(q)$ of chunks that count as correct.

| Task | Query $q$ | Correct set $G(q)$ | Excluded from results |
|---|---|---|---|
| `next_chunk` | chunk $c_p$ (its stored vector) | $\{c_{p+1}\}$, the chunk that follows it in the same article | $c_p$ itself |
| `same_article` | chunk $c_p$ | every other chunk of the same article: $\{c : \operatorname{doc}(c) = \operatorname{doc}(c_p),\ c \ne c_p\}$ | $c_p$ itself |
| `title` | the article's title, encoded with $e$ | every chunk of that article | nothing |

A query **hits** at $k$ if any correct chunk is among the top $k$ results,
after removing the query chunk itself for the chunk tasks:

$$
\text{hit}_k(q) = \mathbb{1}\big[\, R_k(q) \cap G(q) \neq \varnothing \,\big],
\qquad
\text{Recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \text{hit}_k(q).
$$

$Q$ is the query set. Recall@k is simply the share of queries with at least
one correct chunk in the top $k$. This is sometimes called hit rate or
success@k. Because `next_chunk` has exactly one correct chunk, its Recall@k
is also the standard single-target recall.

**Query sampling** (seed 0, 1,000 queries per task):

- **Chunk tasks:** random chunks that have a following chunk in the same
  article, so `next_chunk` always has an answer.
- **Title task:** random articles, using each article's title.

**Stored vectors as queries.** Chunk queries use the chunk's stored vector.
This is the same as re-encoding its text, because queries and chunks share
the encoder and no query prefix is used.

**Random baseline.** Next to each recall is the recall that $k$ uniformly
random picks would get. With $N$ candidates, of which $r = |G(q)|$ are
correct, drawn without replacement:

$$
P_{\text{random}}(\text{hit}_k) = 1 - \prod_{j=0}^{k-1} \frac{N - r - j}{N - j},
$$

averaged over queries. $N = M - 1$ for the chunk tasks (the query chunk is
excluded) and $N = M$ for titles. For `next_chunk` this is about $k / N$.
For `same_article` it grows with article length, since long articles have
more correct chunks.

**Reading the numbers.**

- **`same_article`** tests whether retrieval stays on topic. It is the most
  direct evidence that the embedding works.
- **`next_chunk`** is the relation RETRO relies on, and it is much harder.
  The following chunk continues the text but often covers different content.
- **`title`** is entity lookup, closest to how a question-style query
  behaves. It gets harder as the index grows and more articles have similar
  titles.

All three get harder with more distractors, so they should be measured at
every index size.

**Limitation.** These tasks measure retrieval on text that is *in* the
index. They say nothing directly about the held-out eval queries, whose
articles are deliberately absent from every index.

## 3. Eval-query relatedness (proxy)

For the queries we actually score, from post-2023 articles, no chunk in the
index is known to be correct, so true recall can't be computed. The current
proxy asks whether a neighbor comes from an article whose title shares a word
with the query article's title. The candidate words are those in the query
article's title with 5 or more characters, after removing parentheses. The
match is case-insensitive, against the whitespace-separated words of the
neighbor's title. We report the share of all retrieved neighbors that
pass: 11.1% with `wikipedia_1k` and 14.7% with `wikipedia_100k`, on the
1,000-article k = 4 runs.

This is a crude label. It misses related articles with different titles and
accepts unrelated ones that share a common word. Better per-chunk signals,
all computable from saved outputs, are listed in
[`findings.md`](findings.md#recall-against-bpb):

- the neighbors' cosine similarity to the query,
- the longest token run shared between a neighbor and the target chunk
  (RETRO's leakage overlap).
