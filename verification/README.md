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
