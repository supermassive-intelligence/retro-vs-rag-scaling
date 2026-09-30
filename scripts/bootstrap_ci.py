"""Paired bootstrap over articles: confidence intervals for bpb differences between eval runs.

Every run scores the same windows of the same articles, so runs can be compared
article by article. Each resample draws the eval's articles with replacement,
and each run's bpb is recomputed on that same draw (total nats / total bytes).
Because all runs share the draw, article difficulty cancels out of the
differences, which are far tighter than the runs' own intervals.

    python -m scripts.bootstrap_ci --base data/lm_eval/<sha>/eval500/<model>/none \\
        --run data/lm_eval/<sha>/eval500/<model>/rag_wikipedia_1k \\
        --run data/lm_eval/<sha>/eval500/<model>/rag_wikipedia_100k \\
        --out data/analysis/bootstrap_eval500.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.provenance import git_state, sha256_file


def load(run: Path) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Per-article (ids in file order, nats, bytes) and the run's summary."""
    nats: dict[str, float] = defaultdict(float)
    nbytes: dict[str, int] = defaultdict(int)
    keys = []
    with (run / "windows.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            keys.append((r["article_id"], r["window_start"]))
            nats[r["article_id"]] += sum(r["nll_chunks"])
            nbytes[r["article_id"]] += r["target_bytes"]
    ids = list(nats)
    summary = json.loads((run / "summary.json").read_text())
    summary["_window_keys"] = keys
    return ids, np.array([nats[a] for a in ids]), np.array([nbytes[a] for a in ids]), summary


def interval(x: np.ndarray, level: float) -> list[float]:
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
    return [float(lo), float(hi)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="reference run; differences are run - base")
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    runs = [args.base, *args.run]
    loaded = [load(r) for r in runs]
    ids, _, nbytes, base_summary = loaded[0]
    for r, (ids_r, _, nbytes_r, s) in zip(runs[1:], loaded[1:]):
        # Pairing is only valid on identical windows of identical text.
        assert s["_window_keys"] == base_summary["_window_keys"], f"{r}: windows differ from base"
        assert ids_r == ids and np.array_equal(nbytes_r, nbytes), f"{r}: articles or bytes differ from base"
        assert s["eval"]["articles_sha256"] == base_summary["eval"]["articles_sha256"], f"{r}: different eval set"
    nats = np.stack([n for _, n, _, _ in loaded])  # (runs, articles)

    rng = np.random.default_rng(args.seed)
    n = len(ids)
    counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(args.resamples)])  # (B, articles)
    bpb = (counts @ nats.T) / (math.log(2) * (counts @ nbytes))[:, None]  # (B, runs)
    point = nats.sum(1) / (math.log(2) * nbytes.sum())

    def describe(i: int) -> dict:
        s = loaded[i][3]
        out = {"path": str(runs[i]), "git": s["git"], "command": s["command"],
               "windows_sha256": sha256_file(runs[i] / "windows.jsonl"),
               "bpb": float(point[i]), "bpb_ci": interval(bpb[:, i], args.level)}
        if i:
            d = bpb[:, i] - bpb[:, 0]
            per_article = nats[i] - nats[0]
            out |= {"delta_bpb": float(point[i] - point[0]), "delta_bpb_ci": interval(d, args.level),
                    "p_delta_le_0": float((d <= 0).mean()),
                    "articles_improved": float((per_article < 0).mean())}
        return out

    pairs = {}
    for i in range(1, len(runs)):
        for j in range(i + 1, len(runs)):
            d = bpb[:, j] - bpb[:, i]
            pairs[f"{runs[j].name} - {runs[i].name}"] = {"delta_bpb": float(point[j] - point[i]),
                                                         "delta_bpb_ci": interval(d, args.level)}

    result = {
        "command": " ".join([sys.executable, "-m", "scripts.bootstrap_ci", *sys.argv[1:]]),
        "git": git_state(),
        "method": "paired bootstrap over articles; bpb = resampled total nats / (ln 2 * resampled total bytes)",
        "resamples": args.resamples, "level": args.level, "seed": args.seed, "n_articles": n,
        "base": describe(0), "runs": [describe(i) for i in range(1, len(runs))], "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'run':40s} {'bpb':>8s}  {'Δ vs base':>9s}  {int(args.level * 100)}% CI of Δ")
    print(f"{runs[0].name:40s} {point[0]:8.5f}")
    for r in result["runs"]:
        lo, hi = r["delta_bpb_ci"]
        print(f"{Path(r['path']).name:40s} {r['bpb']:8.5f}  {r['delta_bpb']:+9.5f}  [{lo:+.5f}, {hi:+.5f}]")
    for name, v in pairs.items():
        lo, hi = v["delta_bpb_ci"]
        print(f"{name:60s} {v['delta_bpb']:+.5f}  [{lo:+.5f}, {hi:+.5f}]")


if __name__ == "__main__":
    main()
