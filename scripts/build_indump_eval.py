"""Build the in-dump eval set: articles that ARE in the larger index but not the smaller one.

This is the positive control for Protocol B. On held-out articles the index
holds almost nothing on topic, so a flat result can't tell "retrieval can't
help here" from "retrieval never helps in this setup". Here related text is
guaranteed to exist: every eval article's own chunks are in the index. The
eval's leakage filter still drops any neighbor that shares 32 tokens with the
window being scored, so what remains is the article's other sections, which is
RETRO's test-set-overlap setting.

Articles are drawn from the index with the most articles (--index) and must be
absent from --exclude-index, so the same text can be scored against an index
that contains it and one that doesn't. Text comes from the dump's parquet files,
and each article's first chunk is checked against the index's chunks.jsonl.

    python -m scripts.build_indump_eval --index data/wiki_index/wikipedia_100k \\
        --exclude-index data/wiki_index/wikipedia_1k --n-articles 500 --out data/eval/wiki_indump
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scripts.build_heldout_eval import parquet_files
from scripts.build_wiki_index import chunk_document
from scripts.provenance import git_state, sha256_file

log = logging.getLogger("build_indump_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, required=True, help="index the articles are drawn from")
    p.add_argument("--exclude-index", type=Path, action="append", default=[], help="indexes the articles must not be in")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-articles", type=int, default=500)
    p.add_argument("--min-tokens", type=int, default=1024, help="one full Protocol B window")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool", type=int, default=20_000, help="shuffled candidates whose text is looked up")
    p.add_argument("--repo", default="wikimedia/wikipedia")
    p.add_argument("--revision", default="b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
    p.add_argument("--glob", default="20231101.en/train-*.parquet")
    return p.parse_args()


def index_docs(index_dir: Path) -> dict[str, str]:
    """doc_id -> text of the doc's first chunk, in index order."""
    first: dict[str, str] = {}
    with (index_dir / "chunks.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            first.setdefault(r["doc_id"], r["text"])
    return first


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    started = time.time()
    git = git_state()

    manifest_in = json.loads((args.index / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest_in["chunking"]["tokenizer"])
    chunk_size = manifest_in["chunking"]["chunk_size"]

    first_chunk = index_docs(args.index)
    excluded: set[str] = set()
    for ex in args.exclude_index:
        excluded |= set(index_docs(ex))
    candidates = [d for d in first_chunk if d not in excluded]
    log.info("%d docs in %s, %d after excluding %d", len(first_chunk), args.index, len(candidates), len(excluded))
    random.Random(args.seed).shuffle(candidates)
    pool = candidates[: args.pool]

    texts: dict[str, dict] = {}
    wanted = pa.array(pool)
    for path in parquet_files(args.repo, args.revision, args.glob):
        t = pq.read_table(path, columns=["id", "url", "title", "text"])
        texts.update({r["id"]: r for r in t.filter(pc.is_in(t.column("id"), value_set=wanted)).to_pylist()})
    missing = [d for d in pool if d not in texts]
    assert not missing, f"{len(missing)} pool docs not found in {args.repo}@{args.revision[:8]}, e.g. {missing[:3]}"

    kept: list[dict] = []
    too_short = 0
    for d in pool:
        if len(kept) == args.n_articles:
            break
        r = texts[d]
        n_tokens = len(tokenizer(r["text"], add_special_tokens=False)["input_ids"])
        if n_tokens < args.min_tokens:
            too_short += 1
            continue
        # The text must tokenize into the very chunks the index holds, or "in the index" isn't true.
        assert chunk_document(r["text"], tokenizer, chunk_size)[0][0] == first_chunk[d], f"doc {d}: text differs from index"
        kept.append({**r, "n_tokens": n_tokens, "n_bytes": len(r["text"].encode("utf-8"))})
    assert len(kept) == args.n_articles, f"only {len(kept)} of {args.n_articles}; raise --pool"

    args.out.mkdir(parents=True, exist_ok=True)
    articles_path = args.out / "articles.jsonl"
    with articles_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "command": " ".join([sys.executable, "-m", "scripts.build_indump_eval", *sys.argv[1:]]),
        "git": git,
        "source": {"repo": args.repo, "revision": args.revision, "files": args.glob},
        "in_index": {"path": str(args.index), "index_faiss_sha256": manifest_in["files"]["index.faiss"]},
        "excluded_indexes": [{"path": str(ex), "index_faiss_sha256": json.loads((ex / "manifest.json").read_text())["files"]["index.faiss"]}
                             for ex in args.exclude_index],
        "selection": {"seed": args.seed, "pool": args.pool, "min_tokens": args.min_tokens,
                      "tokenizer": manifest_in["chunking"]["tokenizer"], "n_candidates": len(candidates),
                      "n_requested": args.n_articles, "n_kept": len(kept), "rejected": {"too_short": too_short}},
        "totals": {"tokens": sum(r["n_tokens"] for r in kept), "bytes": sum(r["n_bytes"] for r in kept)},
        "files": {"articles.jsonl": sha256_file(articles_path)},
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s: %d articles, %d tokens", args.out, len(kept), manifest["totals"]["tokens"])


if __name__ == "__main__":
    main()
