# Protocol B: the math

This is the exact computation behind `scripts/eval_lm.py`: which tokens are
scored, what each one is conditioned on, and how the per-token losses become
the language-modeling loss, bits-per-byte (bpb) and perplexity (ppl). The
motivation and design choices are in [`eval_protocol.md`](eval_protocol.md).

## 1. Notation

| Symbol | Meaning |
|---|---|
| $x_1, \dots, x_N$ | one held-out article, tokenized with the Qwen2.5 tokenizer |
| $m = 64$ | chunk size, the retrieval stride |
| $W = 1024$ | window length |
| $S = 512$ | window stride, which is also the number of scored tokens per window |
| $k = 4$ | neighbors per scored chunk |
| $p_\theta$ | the language model's next-token distribution |
| $x_{a:b}$ | tokens $x_a, \dots, x_{b-1}$ (end excluded) |

## 2. Windows and scored tokens

Window $w = 0, 1, 2, \dots$ starts at $s_w = S \cdot w$ and covers $x_{s_w : s_w + W}$.
Only windows that fit entirely inside the article are used ($s_w + W \le N$), so an
article of $N$ tokens has

$$
n_{\text{win}}(N) = \left\lfloor \frac{N - S}{S} \right\rfloor \quad (N \ge W)
$$

windows. In each window, the first $S$ tokens are **context** and the last $S$
are the **target**:

$$
\text{context}_w = x_{s_w : s_w + S}, \qquad \text{target}_w = x_{s_w + S : s_w + W}.
$$

Because windows advance by exactly $S$, the targets tile the article: every
token from position $S$ up to the last full window is scored **exactly once**.
The first $S$ tokens of each article are never scored, and neither is any tail
after the last full window. Both apply identically to every arm.

The target splits into $S/m = 8$ chunks. Chunk $T_i$ ($i = 1, \dots, 8$) starts at

$$
c_i = s_w + S + (i - 1)\, m, \qquad T_i = x_{c_i : c_i + m}.
$$

Write $T_0 = x_{s_w + S - m : s_w + S}$ for the last chunk of the context.

## 3. Retrieval for chunk $T_i$

**Query.** The query is the chunk just before it, $q_i = T_{i-1} = x_{c_i - m : c_i}$,
decoded to text. It is fully known before any token of $T_i$ is predicted, so
retrieval is causal.

**Scoring.** Let $e(\cdot)$ be the BGE-small encoder with CLS pooling and L2
normalization, and let $d_1, \dots, d_M$ be the index's chunks. Retrieval is
exact maximum inner product search:

$$
\operatorname{score}(q, d_j) = e(q)^\top e(d_j) = \cos\big(e(q), e(d_j)\big).
$$

The 16 highest-scoring chunks are the candidates, ranked $r_1, r_2, \dots$.

**Span.** Candidate $r$ is expanded to $\operatorname{span}(r)$: chunk $r$ plus
the next chunk of the same article, truncated to $2m = 128$ tokens.

**Leakage filter.** Let $G_{32}(y)$ be the set of 32-token n-grams of $y$. A
candidate is dropped if

$$
G_{32}\big(\operatorname{span}(r)\big) \cap G_{32}(\text{target}_w) \neq \varnothing,
$$

meaning it shares 32 or more consecutive tokens with the window's target. The
neighbor set $\mathcal{N}_i$ is the first $k$ candidates that survive, in rank
order $n_1, \dots, n_k$.

## 4. What each scored token is conditioned on

Every scored token $x_t$ receives a loss

$$
\ell_t = -\ln p_\theta\big(x_t \mid \text{input before } x_t\big) \quad \text{(nats)}.
$$

The arms differ only in that input.

**LM-only (`--arm none`).** One forward pass over the whole window:

$$
\ell^{\text{none}}_t = -\ln p_\theta\big(x_t \mid x_{s_w : t}\big), \qquad t \in [s_w + S,\ s_w + W).
$$

**RAG (`--arm rag`).** Each chunk $T_i$ gets its own forward pass. The
neighbors come first in reverse rank order, so the best one sits closest to
the text, and each is followed by the separator $\sigma$ (the tokens of
`"\n\n"`):

$$
R_i = \operatorname{span}(n_k) \oplus \sigma \oplus \cdots \oplus \operatorname{span}(n_1) \oplus \sigma,
$$

$$
\ell^{\text{rag}}_t = -\ln p_\theta\big(x_t \mid R_i \oplus x_{s_w : t}\big), \qquad t \in [c_i,\ c_i + m).
$$

Only the $m$ tokens of $T_i$ are scored in that pass. $R_i$, the context
tokens and $T_1, \dots, T_{i-1}$ are conditioning only.

**Check.** With $k = 0$, $R_i$ is empty and $\ell^{\text{rag}}_t = \ell^{\text{none}}_t$
exactly. The implementation reproduces this to within $1.8 \times 10^{-4}$ nats
per chunk in float32.

**GCCA (later).** One forward pass like the LM-only arm, with no neighbors in
the input. The tokens of $T_i$ instead cross-attend to the same $\mathcal{N}_i$.

## 5. From token losses to the metrics

Sum over every scored token of every window of every article, for one arm:

$$
L = \sum_{\text{articles}} \sum_{w} \sum_{i=1}^{8} \sum_{t \in T_i} \ell_t \quad \text{(total nats)}
$$

$$
T_{\text{tok}} = 512 \times (\text{number of windows}) \quad \text{(scored tokens)}
$$

$$
B = \sum_{\text{articles}} \sum_{w} \big|\operatorname{utf8}\big(\operatorname{decode}(\text{target}_w)\big)\big| \quad \text{(scored bytes)}
$$

$T_{\text{tok}}$ and $B$ depend only on the eval set and the windowing, so they
are **identical across arms and index sizes**. Every comparison is therefore a
comparison of $L$ alone.

**Language-modeling loss** (mean cross-entropy per token):

$$
\bar{\ell} = \frac{L}{T_{\text{tok}}} \quad \text{nats/token}.
$$

**Perplexity:**

$$
\text{ppl} = \exp\!\left(\frac{L}{T_{\text{tok}}}\right) = e^{\bar{\ell}}.
$$

**Bits per token:** $\bar{\ell} / \ln 2$.

**Bits-per-byte:**

$$
\text{bpb} = \frac{L}{B \ln 2} = \frac{T_{\text{tok}}}{B} \cdot \frac{\bar{\ell}}{\ln 2}.
$$

The second form is the Pile paper's definition, $\text{BPB} = (L_T / L_B)\,\ell / \ln 2$
(Gao et al. 2020). Dividing by $\ln 2$ converts nats to bits, and dividing by
bytes instead of tokens makes the number independent of the tokenizer. That
is why RETRO, REPLUG and the Pile report bpb.

The two metrics are tied together by

$$
\text{ppl} = 2^{\,\text{bpb} \cdot B / T_{\text{tok}}}.
$$

**Aggregation is over all tokens, not per article.** A long article
contributes more scored tokens and so more weight. Averaging per-article bpb
values would give a different, length-unweighted number.

## 6. Comparing arms

Since $B$ and $T_{\text{tok}}$ are shared across arms:

$$
\Delta\text{bpb} = \frac{L_{\text{rag}} - L_{\text{none}}}{B \ln 2}, \qquad
\frac{\text{ppl}_{\text{rag}}}{\text{ppl}_{\text{none}}} = \exp\!\left(\frac{L_{\text{rag}} - L_{\text{none}}}{T_{\text{tok}}}\right).
$$

The relative improvement from retrieval is $1 - \text{bpb}_{\text{rag}} / \text{bpb}_{\text{none}}$,
the quantity RETRO plots. On the scaling plot, the x-axis is the index size in
articles, and each RAG point is compared with the LM-only line.

## 7. Worked example

The 5-article smoke run of the LM-only arm (`Qwen2.5-0.5B-Instruct`, float32):
8 windows, $T_{\text{tok}} = 4{,}096$, $B = 15{,}804$ bytes, bpb $= 1.08878$.

| Quantity | Computation | Value |
|---|---|---|
| Total loss $L$ | $1.08878 \times 15{,}804 \times \ln 2$ | 11,927.04 nats |
| Mean loss $\bar{\ell}$ | $11{,}927.04 / 4{,}096$ | 2.91187 nats/token |
| Perplexity | $e^{2.91187}$ | 18.391 |
| Bits per token | $2.91187 / \ln 2$ | 4.20095 |
| Bytes per token | $15{,}804 / 4{,}096$ | 3.858 |
| Perplexity from bpb | $2^{1.08878 \times 3.858}$ | 18.391 |

This matches the ppl the run reported (18.391).

## 8. Edge cases

- **Window boundaries and multi-byte characters.** $B$ decodes each window's
  target separately. A multi-byte UTF-8 character split across a window
  boundary decodes to U+FFFD (3 bytes) instead of its true bytes, a tiny
  miscount at a rare event. It is identical across arms, so it cannot affect
  comparisons.
- **Fewer than $k$ neighbors.** If the leakage filter removes too many of the
  16 candidates, $\mathcal{N}_i$ has fewer than $k$ spans. The run counts how
  often that happens (`chunks_short_of_k`).
- **Padding.** Sequences in a batch are right-padded with an attention mask.
  Scored positions are real tokens only, so padding never enters $L$.
