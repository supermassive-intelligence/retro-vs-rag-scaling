"""Build the held-out eval set: English Wikipedia articles created after the datastore's dump.

The datastore is wikimedia/wikipedia 20231101.en. Page IDs are assigned at
creation, so any article in a later snapshot whose ID exceeds the dump's
largest ID was created after the dump. Each sampled article's creation date
(its first revision) is then confirmed with the Wikipedia API.

    python -m scripts.build_heldout_eval --n-articles 1000 --out data/eval/wiki_postdump
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from scripts.provenance import git_state, sha256_file

log = logging.getLogger("build_heldout_eval")

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "retro-vs-rag-scaling/0.1 (https://github.com/supermassive-intelligence/retro-vs-rag-scaling)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-articles", type=int, default=1000)
    p.add_argument("--min-tokens", type=int, default=1024, help="one full Protocol B window")
    p.add_argument("--min-chars", type=int, default=4000, help="cheap pre-filter before tokenizing")
    p.add_argument("--created-after", default="2023-11-01", help="UTC date; first revision must be on/after it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--api-delay", type=float, default=0.5, help="seconds between Wikipedia API calls")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--old-repo", default="wikimedia/wikipedia")
    p.add_argument("--old-revision", default="b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
    p.add_argument("--old-glob", default="20231101.en/train-*.parquet")
    p.add_argument("--new-repo", default="omarkamali/wikipedia-monthly")
    p.add_argument("--new-revision", default="9cd30b1feefeeedeb4e629b221d9b4c55469d523")
    p.add_argument("--new-glob", default="20260301/en/train/train_part_*.parquet")
    return p.parse_args()


def parquet_files(repo: str, revision: str, pattern: str) -> list[Path]:
    # Whole-file download: ranged reads of one column over HTTP measured ~5x slower than fetching the file.
    root = Path(snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=[pattern]))
    files = sorted(root.glob(pattern))
    assert files, f"no files match {repo}@{revision}/{pattern}"
    log.info("%s@%s: %d files", repo, revision[:8], len(files))
    return files


def max_page_id(files: list[Path]) -> tuple[int, int]:
    best, n = -1, 0
    for path in files:
        ids = pq.read_table(path, columns=["id"]).column("id").cast("int64")
        n += len(ids)
        best = max(best, pc.max(ids).as_py())
    return best, n


def newer_articles(files: list[Path], min_id: int, min_chars: int) -> tuple[list[dict], dict[str, int]]:
    """Articles with page ID > min_id and at least min_chars of text, reading text only where needed."""
    stats = {"rows_scanned": 0, "newer_ids": 0, "newer_long_enough": 0}
    out: list[dict] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            ids = pf.read_row_group(rg, columns=["id"]).column("id").cast("int64")
            stats["rows_scanned"] += len(ids)
            mask = pc.greater(ids, min_id)
            n_new = pc.sum(mask).as_py() or 0
            if not n_new:
                continue
            stats["newer_ids"] += n_new
            rows = pf.read_row_group(rg, columns=["id", "url", "title", "text"]).filter(mask).to_pylist()
            out.extend(r for r in rows if len(r["text"]) >= min_chars)
        log.info("%s: %d candidates so far", path.name, len(out))
    stats["newer_long_enough"] = len(out)
    return out, stats


def first_revision(page_id: int) -> str | None:
    """UTC timestamp of the page's first revision, or None if the page no longer exists."""
    query = {
        "action": "query", "format": "json", "formatversion": "2", "prop": "revisions",
        "rvdir": "newer", "rvlimit": "1", "rvprop": "timestamp", "pageids": str(page_id),
    }
    req = urllib.request.Request(f"{WIKI_API}?{urllib.parse.urlencode(query)}", headers={"User-Agent": USER_AGENT})
    attempts = 8
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                page = json.load(resp)["query"]["pages"][0]
            if page.get("missing") or not page.get("revisions"):
                return None
            return page["revisions"][0]["timestamp"]
        except OSError as e:
            if attempt == attempts - 1:
                raise
            wait = min(2**attempt, 60)
            if isinstance(e, urllib.error.HTTPError) and e.code in (429, 503):
                wait = max(wait, int(e.headers.get("Retry-After") or 10))
            log.warning("wikipedia api: %s; retrying in %ds", e, wait)
            time.sleep(wait)
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    started = time.time()

    old_files = parquet_files(args.old_repo, args.old_revision, args.old_glob)
    old_max_id, old_rows = max_page_id(old_files)
    log.info("old dump: %d articles, max page id %d", old_rows, old_max_id)

    new_files = parquet_files(args.new_repo, args.new_revision, args.new_glob)
    candidates, scan = newer_articles(new_files, old_max_id, args.min_chars)
    log.info("new snapshot: %s", scan)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    random.Random(args.seed).shuffle(candidates)
    cutoff = f"{args.created_after}T00:00:00Z"
    rejected = {"too_short": 0, "created_before_cutoff": 0, "page_missing": 0}
    kept: list[dict] = []
    for r in candidates:
        if len(kept) == args.n_articles:
            break
        n_tokens = len(tokenizer(r["text"], add_special_tokens=False)["input_ids"])
        if n_tokens < args.min_tokens:
            rejected["too_short"] += 1
            continue
        created = first_revision(int(r["id"]))
        time.sleep(args.api_delay)
        if created is None:
            rejected["page_missing"] += 1
            continue
        if created < cutoff:
            rejected["created_before_cutoff"] += 1
            continue
        kept.append({**r, "created": created, "n_tokens": n_tokens, "n_bytes": len(r["text"].encode("utf-8"))})
        if len(kept) % 100 == 0:
            log.info("kept %d / %d (rejected %s)", len(kept), args.n_articles, rejected)

    if len(kept) < args.n_articles:
        log.warning("only %d of %d requested articles passed", len(kept), args.n_articles)

    args.out.mkdir(parents=True, exist_ok=True)
    articles_path = args.out / "articles.jsonl"
    with articles_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    created = sorted(r["created"] for r in kept)
    manifest = {
        "command": " ".join([sys.executable, "-m", "scripts.build_heldout_eval", *sys.argv[1:]]),
        "git": git_state(),
        "datastore_dump": {"repo": args.old_repo, "revision": args.old_revision, "files": args.old_glob,
                           "n_articles": old_rows, "max_page_id": old_max_id},
        "source": {"repo": args.new_repo, "revision": args.new_revision, "files": args.new_glob, **scan},
        "selection": {"seed": args.seed, "min_chars": args.min_chars, "min_tokens": args.min_tokens,
                      "tokenizer": args.tokenizer, "created_after": args.created_after,
                      "n_requested": args.n_articles, "n_kept": len(kept), "rejected": rejected},
        "created_range": {"min": created[0], "median": created[len(created) // 2], "max": created[-1]} if created else None,
        "totals": {"tokens": sum(r["n_tokens"] for r in kept), "bytes": sum(r["n_bytes"] for r in kept)},
        "files": {"articles.jsonl": sha256_file(articles_path)},
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
