"""Internal check: does ExactIndex (PyTorch, GPU) return the same neighbors as FAISS IndexFlatIP (CPU)?

Queries are built the way eval_lm builds them: the 64 tokens just before a
target chunk of a held-out article, decoded to text and encoded with the index's
encoder. Near-ties can legitimately swap order between two exact
implementations, so every mismatch is reported with its score gap.

    python -m verification.exact_search_equivalence --index data/wiki_index/wikipedia_100k \\
        --eval data/eval/wiki_postdump --out data/verification/exact_search_wikipedia_100k.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from scripts.exact_search import METHOD, ExactIndex
from scripts.provenance import git_state, sha256_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--n-queries", type=int, default=5000)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    git = git_state()

    manifest = json.loads((args.index / "manifest.json").read_text())
    chunk = manifest["chunking"]["chunk_size"]
    tokenizer = AutoTokenizer.from_pretrained(manifest["chunking"]["tokenizer"])
    rng = random.Random(args.seed)
    articles = [json.loads(line)["text"] for line in (args.eval / "articles.jsonl").open()]
    queries = []
    while len(queries) < args.n_queries:
        ids = tokenizer(rng.choice(articles), add_special_tokens=False)["input_ids"]
        c = rng.randrange(512, len(ids) - chunk + 1, chunk) if len(ids) >= 512 + chunk else None
        if c is not None:
            queries.append(tokenizer.decode(ids[c - chunk : c]))
    encoder = SentenceTransformer(manifest["embedding"]["model"], device="cuda")
    q = encoder.encode(queries, batch_size=512, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    del encoder

    faiss_index = faiss.read_index(str(args.index / "index.faiss"))
    t0 = time.time()
    f_scores, f_ids = faiss_index.search(q, args.k)
    faiss_s = time.time() - t0
    del faiss_index

    t0 = time.time()
    exact = ExactIndex(args.index / "index.faiss", device="cuda")
    torch.cuda.synchronize()
    load_s = time.time() - t0
    exact.search(q[:64], args.k)  # warm-up: CUDA context and kernels
    t0 = time.time()
    g_scores, g_ids = exact.search(q, args.k)
    torch.cuda.synchronize()
    gpu_s = time.time() - t0

    same_order = (f_ids == g_ids).all(axis=1)
    same_set = np.array([set(a) == set(b) for a, b in zip(f_ids, g_ids)])
    # For each differing position, how far apart were the two scores FAISS saw? Tiny gaps mean near-ties.
    gaps = [abs(float(fs[j]) - float(gs[j])) for fs, gs, fi, gi in zip(f_scores, g_scores, f_ids, g_ids)
            for j in range(args.k) if fi[j] != gi[j]]
    top1_same = float(np.mean(f_ids[:, 0] == g_ids[:, 0]))
    report = {
        "command": " ".join([sys.executable, "-m", "verification.exact_search_equivalence", *sys.argv[1:]]),
        "git": git,
        "code_sha256": {"scripts/exact_search.py": sha256_file(Path(__file__).parents[1] / "scripts" / "exact_search.py"),
                        "verification/exact_search_equivalence.py": sha256_file(Path(__file__))},
        "index": str(args.index), "n_vectors": exact.ntotal, "n_queries": len(queries), "k": args.k,
        "method": METHOD,
        "identical_order": float(same_order.mean()),
        "identical_set": float(same_set.mean()),
        "top1_identical": top1_same,
        "max_abs_score_diff_sorted": float(np.abs(f_scores - g_scores).max()),
        "mismatched_positions": len(gaps),
        "max_score_gap_at_mismatch": max(gaps) if gaps else 0.0,
        "timing_s": {"faiss_cpu_search": round(faiss_s, 2), "gpu_load": round(load_s, 2), "gpu_search": round(gpu_s, 3)},
        "speedup": round(faiss_s / gpu_s, 1),
        "env": {"gpu": torch.cuda.get_device_name(0), "cpu_threads": faiss.omp_get_max_threads(),
                "torch": torch.__version__, "faiss": faiss.__version__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for key in ("n_vectors", "n_queries", "identical_order", "identical_set", "top1_identical",
                "max_abs_score_diff_sorted", "mismatched_positions", "max_score_gap_at_mismatch", "timing_s", "speedup", "env"):
        print(f"{key}: {report[key]}")


if __name__ == "__main__":
    main()
