"""Internal check: is dense retrieval finding related text? Recall@k on relations the index already knows.

Three tasks, each with a known correct answer:
  next_chunk    query = chunk p of an article (its stored vector); hit = chunk p+1 of the same article
  same_article  query = chunk p; hit = any other chunk of the same article
  title         query = the article's title (encoded); hit = any chunk of that article

Each recall is reported next to the recall random retrieval would get.

    python -m verification.retrieval_quality --index data/wiki_index/wikipedia_1k \\
        --out data/verification/retrieval_quality_wikipedia_1k.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.provenance import git_state, sha256_file

KS = (1, 4, 10)


def p_random_hit(n_relevant: int, n_candidates: int, k: int) -> float:
    """Chance that k uniformly random picks (without replacement) include one of n_relevant items."""
    miss = 1.0
    for j in range(k):
        miss *= max(0, n_candidates - n_relevant - j) / (n_candidates - j)
    return 1.0 - miss


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    git = git_state()

    docs, titles = [], []
    with (args.index / "chunks.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            docs.append(r["doc_id"])
            titles.append(r["title"])
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        by_doc[d].append(i)

    index = faiss.read_index(str(args.index / "index.faiss"))
    M = index.ntotal
    assert M == len(docs), (M, len(docs))
    k_max = max(KS)
    rng = random.Random(args.seed)
    hits: dict[tuple[str, int], list[bool]] = defaultdict(list)
    chance: dict[tuple[str, int], list[float]] = defaultdict(list)

    # Chunk queries reuse the stored vectors: queries and chunks share the encoder and no prefix is used.
    with_next = [i for i in range(M - 1) if docs[i + 1] == docs[i]]
    chunk_queries = rng.sample(with_next, min(args.n_queries, len(with_next)))
    q_vecs = np.stack([index.reconstruct(i) for i in chunk_queries])
    _, ids = index.search(q_vecs, k_max + 1)
    for qi, row in zip(chunk_queries, ids):
        ranked = [int(c) for c in row if c >= 0 and c != qi][:k_max]
        same = set(by_doc[docs[qi]]) - {qi}
        for k in KS:
            hits[("next_chunk", k)].append(qi + 1 in ranked[:k])
            chance[("next_chunk", k)].append(p_random_hit(1, M - 1, k))
            hits[("same_article", k)].append(bool(same & set(ranked[:k])))
            chance[("same_article", k)].append(p_random_hit(len(same), M - 1, k))

    manifest = json.loads((args.index / "manifest.json").read_text())
    encoder = SentenceTransformer(manifest["embedding"]["model"], device="cpu")
    title_docs = rng.sample(sorted(by_doc), min(args.n_queries, len(by_doc)))
    t_vecs = encoder.encode([titles[by_doc[d][0]] for d in title_docs], normalize_embeddings=True, convert_to_numpy=True)
    _, ids = index.search(t_vecs.astype(np.float32), k_max)
    for d, row in zip(title_docs, ids):
        for k in KS:
            hits[("title", k)].append(bool(set(by_doc[d]) & {int(c) for c in row[:k]}))
            chance[("title", k)].append(p_random_hit(len(by_doc[d]), M, k))

    metrics = {f"{task}@{k}": {"recall": float(np.mean(v)), "random": float(np.mean(chance[(task, k)])), "n": len(v)}
               for (task, k), v in hits.items()}
    print(f"{args.index} ({M:,} vectors, {len(by_doc):,} articles)")
    for name, m in metrics.items():
        print(f"  {name:16s} recall {m['recall']:6.1%}   random {m['random']:.3%}   (n={m['n']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "command": " ".join([sys.executable, "-m", "verification.retrieval_quality", *sys.argv[1:]]),
        "git": git,
        "code_sha256": {"verification/retrieval_quality.py": sha256_file(Path(__file__))},
        "index": str(args.index),
        "index_faiss_sha256": manifest["files"]["index.faiss"],
        "encoder": manifest["embedding"]["model"],
        "seed": args.seed,
        "metrics": metrics,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
