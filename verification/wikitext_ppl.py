"""Internal check: reproduce published WikiText-2 perplexities with eval_lm's own Scorer.

GPTQ protocol (Frantar et al. 2023), as used by AAAC (arXiv 2605.08692):
the full wikitext-2-raw-v1 test set joined with "\\n\\n", tokenized without
filtering, split into non-overlapping 2,048-token segments. Each segment scores
its last 2,047 tokens, and ppl = exp(mean of per-segment mean losses).

    python -m verification.wikitext_ppl --model Qwen/Qwen2.5-0.5B --model Qwen/Qwen2.5-7B \\
        --out data/verification/wikitext2.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from scripts.eval_lm import Scorer
from scripts.provenance import git_state, sha256_file

log = logging.getLogger("wikitext_ppl")

# BF16 WikiText-2 perplexities, AAAC (arXiv 2605.08692) Table 6.
REFERENCE = {"Qwen/Qwen2.5-0.5B": 13.03, "Qwen/Qwen2.5-7B": 6.80, "Qwen/Qwen2.5-14B": 5.25}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    results = []
    for name in args.model:
        tokenizer = AutoTokenizer.from_pretrained(name)
        ids = tokenizer(text)["input_ids"]
        n_seg = len(ids) // args.seqlen
        segments = [ids[i * args.seqlen : (i + 1) * args.seqlen] for i in range(n_seg)]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        scorer = Scorer(name, "bfloat16", pad_id)
        seg_means = []
        for b in range(0, n_seg, args.batch_size):
            batch = segments[b : b + args.batch_size]
            for nll in scorer.per_token_nll(batch, [args.seqlen - 1] * len(batch)):
                seg_means.append(float(nll.mean()))
        ppl = math.exp(float(np.mean(seg_means)))
        ref = REFERENCE.get(name)
        results.append({"model": name, "tokens": len(ids), "segments": n_seg, "ppl": ppl, "reference": ref,
                        "rel_diff": None if ref is None else (ppl - ref) / ref})
        log.info("%s: ppl %.3f (reference %s) over %d segments", name, ppl, ref, n_seg)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "command": " ".join([sys.executable, "-m", "verification.wikitext_ppl", *sys.argv[1:]]),
        "git": git_state(),
        "code_sha256": {"verification/wikitext_ppl.py": sha256_file(Path(__file__)),
                        "scripts/eval_lm.py": sha256_file(Path(__file__).parents[1] / "scripts" / "eval_lm.py")},
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "protocol": {"seqlen": args.seqlen, "dtype": "bfloat16", "aggregation": "exp(mean of per-segment mean NLL)"},
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
