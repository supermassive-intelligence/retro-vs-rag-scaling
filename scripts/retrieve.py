"""Dense top-k retrieval over a built Wikipedia chunk index.

The basic retriever of DPR (Karpukhin et al. 2020) and RAG (Lewis et al.
2020): embed the query with the same bi-encoder that embedded the chunks, then
run exact maximum-inner-product search (brute force, FAISS IndexFlatIP) for
the top k. No reranking, no approximate search, no query rewriting.

The query encoder is read from the index manifest, so queries and chunks can
never be embedded by different models.

    python -m scripts.retrieve --index data/wiki_index/wikipedia_1k \\
        --queries queries.jsonl --k 5 --out data/retrieval/wikipedia_1k/run
    python -m scripts.retrieve --index data/wiki_index/wikipedia_1k \\
        --query "When was the tower built?" --out data/retrieval/wikipedia_1k/adhoc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from scripts.provenance import git_state, sha256_file

log = logging.getLogger("retrieve")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, required=True, help="index dir from build_wiki_index")
    p.add_argument("--queries", type=Path, help='JSONL with {"id": ..., "query": ...} per line')
    p.add_argument("--query", action="append", default=[], help="ad-hoc query; repeatable")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--query-prefix", default="", help="prepended to every query before encoding")
    p.add_argument("--out", type=Path, required=True, help="output dir for results.jsonl + manifest.json")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    args = p.parse_args()
    if not args.queries and not args.query:
        p.error("give --queries and/or --query")
    return args


def load_queries(args: argparse.Namespace) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    if args.queries:
        with args.queries.open() as f:
            queries += [json.loads(line) for line in f if line.strip()]
    queries += [{"id": f"adhoc-{i}", "query": q} for i, q in enumerate(args.query)]
    return queries


def load_chunks(path: Path, wanted: set[int]) -> dict[int, dict]:
    # chunk_id is the line number in chunks.jsonl, so one streaming pass fetches only the hits.
    found: dict[int, dict] = {}
    with path.open() as f:
        for i, line in enumerate(f):
            if i in wanted:
                rec = json.loads(line)
                assert rec["chunk_id"] == i, f"chunk_id {rec['chunk_id']} on line {i}"
                found[i] = rec
    missing = wanted - found.keys()
    assert not missing, f"{len(missing)} hit ids not in {path}"
    return found


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    started = time.time()
    git = git_state()  # at start: the tree can change while a long run is going

    index_manifest_path = args.index / "manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text())
    queries = load_queries(args)
    log.info("%d queries against %s (%d vectors)", len(queries), args.index, index_manifest["index"]["n_vectors"])

    encoder = SentenceTransformer(index_manifest["embedding"]["model"], device=args.device)
    q_emb = encoder.encode(
        [args.query_prefix + q["query"] for q in queries],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    index = faiss.read_index(str(args.index / "index.faiss"))
    assert index.d == q_emb.shape[1], f"index dim {index.d} != query dim {q_emb.shape[1]}"
    t0 = time.time()
    scores, ids = index.search(q_emb, args.k)
    search_s = time.time() - t0

    chunks = load_chunks(args.index / "chunks.jsonl", {int(i) for i in ids.flatten() if i >= 0})

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    with results_path.open("w") as f:
        for q, row_scores, row_ids in zip(queries, scores, ids):
            hits = [
                {"rank": r, "score": float(s), **{k: chunks[int(i)][k] for k in ("chunk_id", "doc_id", "title", "position", "text")}}
                for r, (s, i) in enumerate(zip(row_scores, row_ids))
                if i >= 0
            ]
            f.write(json.dumps({"id": q["id"], "query": q["query"], "hits": hits}, ensure_ascii=False) + "\n")

    query_blob = json.dumps(queries, sort_keys=True).encode()
    manifest = {
        "command": " ".join([sys.executable, "-m", "scripts.retrieve", *sys.argv[1:]]),
        "git": git,
        "index": {
            "path": str(args.index),
            "manifest_sha256": sha256_file(index_manifest_path),
            "index_faiss_sha256": index_manifest["files"]["index.faiss"],
            "n_docs": index_manifest["sampling"]["n_docs"],
            "n_vectors": index_manifest["index"]["n_vectors"],
        },
        "retrieval": {
            "method": "dense exact MIPS (FAISS IndexFlatIP, cosine on L2-normalized vectors)",
            "encoder": index_manifest["embedding"]["model"],
            "query_prefix": args.query_prefix,
            "k": args.k,
            "device": str(encoder.device),
        },
        "queries": {"n": len(queries), "sha256": hashlib.sha256(query_blob).hexdigest(), "source": str(args.queries) if args.queries else None},
        "files": {"results.jsonl": sha256_file(results_path)},
        "search_s": round(search_s, 4),
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s (search %.3fs)", args.out, search_s)


if __name__ == "__main__":
    main()
