"""Build an index from Wikipedia articles created after the language model was trained.

The 2023 indexes hold text Qwen2.5 has almost certainly seen, and almost nothing
on the held-out articles' post-2023 topics. This index draws only from articles
created on or after --created-after (default 2024-10-01, after Qwen2.5's
September 2024 release) in a 2026 snapshot, leaving out every held-out eval
article.

Creation date comes from page IDs, which Wikipedia assigns in creation order.
The cutoff ID is the smallest ID among the held-out eval articles whose first
revision (checked with the Wikipedia API when the eval set was built) is on or
after --created-after; every page with a larger ID was created later still. A
sample of the chosen articles is re-checked against the API.

Chunking and embedding are build_wiki_index's own, so the indexes are directly
comparable. Articles are taken in one seeded shuffle, so a smaller index's
articles are a prefix of a larger one's.

    python -m scripts.build_unseen_index --n-docs 1000 --out data/wiki_index/wikipedia_unseen_1k --device cuda
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

from scripts.build_heldout_eval import first_revision, parquet_files
from scripts.build_wiki_index import index_documents
from scripts.provenance import git_state, sha256_file

log = logging.getLogger("build_unseen_index")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-docs", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval", type=Path, default=Path("data/eval/wiki_postdump"),
                   help="held-out eval set: its articles are excluded and its creation dates set the cutoff ID")
    p.add_argument("--created-after", default="2024-10-01", help="UTC date")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", type=int, default=100, help="chosen articles whose creation date is re-checked via the API")
    p.add_argument("--api-delay", type=float, default=0.5)
    p.add_argument("--repo", default="omarkamali/wikipedia-monthly")
    p.add_argument("--revision", default="9cd30b1feefeeedeb4e629b221d9b4c55469d523")
    p.add_argument("--glob", default="20260301/en/train/train_part_*.parquet")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    git = git_state()

    cutoff = f"{args.created_after}T00:00:00Z"
    eval_articles = [json.loads(line) for line in (args.eval / "articles.jsonl").open()]
    after = [int(r["id"]) for r in eval_articles if r["created"] >= cutoff]
    before = [int(r["id"]) for r in eval_articles if r["created"] < cutoff]
    min_id = min(after)
    # Page IDs rise with creation time; an eval article below the cutoff date with a higher ID would break that.
    assert all(i < min_id for i in before), "page IDs are not in creation order around the cutoff"
    eval_ids = {r["id"] for r in eval_articles}
    eval_titles = {r["title"] for r in eval_articles}
    log.info("cutoff %s -> page id >= %d (from %d eval articles)", cutoff, min_id, len(eval_articles))

    files = parquet_files(args.repo, args.revision, args.glob)
    candidates: list[str] = []
    excluded = 0
    for path in files:
        t = pq.read_table(path, columns=["id", "title"])
        t = t.filter(pc.greater_equal(t.column("id").cast("int64"), min_id))
        for i, title in zip(t.column("id").to_pylist(), t.column("title").to_pylist()):
            if i in eval_ids or title in eval_titles:
                excluded += 1
            else:
                candidates.append(i)
    candidates.sort(key=int)  # a fixed order before the seeded shuffle
    random.Random(args.seed).shuffle(candidates)
    chosen = candidates[: args.n_docs]
    assert len(chosen) == args.n_docs, f"only {len(candidates)} candidates"
    log.info("%d candidates (%d eval articles excluded), taking %d", len(candidates), excluded, len(chosen))

    rows: dict[str, dict] = {}
    wanted = pa.array(chosen)
    for path in files:
        t = pq.read_table(path, columns=["id", "title", "text"])
        rows.update({r["id"]: r for r in t.filter(pc.is_in(t.column("id"), value_set=wanted)).to_pylist()})
    docs = [rows[i] for i in chosen]

    checked = []
    for i in chosen[: args.verify]:
        checked.append({"id": i, "created": first_revision(int(i))})
        time.sleep(args.api_delay)
    early = [c for c in checked if c["created"] is not None and c["created"] < cutoff]
    if early:
        log.warning("%d of %d checked articles were created before %s: %s", len(early), len(checked), cutoff, early[:5])

    stats = index_documents(docs, args.out, args, total=len(docs))
    manifest = {
        "command": " ".join([sys.executable, "-m", "scripts.build_unseen_index", *sys.argv[1:]]),
        "git": git,
        "dataset": {"repo": args.repo, "revision": args.revision, "files": args.glob},
        "sampling": {"seed": args.seed, "n_docs_requested": args.n_docs, "n_docs": stats.pop("n_docs"),
                     "created_after": args.created_after, "min_page_id": min_id, "n_candidates": len(candidates),
                     "excluded_eval": {"path": str(args.eval), "articles_sha256": sha256_file(args.eval / "articles.jsonl"),
                                       "n_excluded": excluded}},
        "creation_check": {"n_checked": len(checked), "n_missing": sum(c["created"] is None for c in checked),
                           "n_before_cutoff": len(early), "min_created": min((c["created"] for c in checked if c["created"]), default=None)},
        **stats,
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
