"""Protocol B language-modeling eval: bits-per-byte on held-out articles (docs/eval_protocol.md).

Each article is cut into 1,024-token windows with a 512-token stride. The first
512 tokens of a window are context and the last 512 (T1..T8, 64 tokens each)
are scored, so every scored token is scored exactly once.

    --arm none   LM-only baseline: one forward pass per window.
    --arm rag    For each Ti, retrieve with T(i-1), prepend the neighbors, and
                 score Ti in its own forward pass.

    python -m scripts.eval_lm --arm none --model Qwen/Qwen2.5-0.5B-Instruct \\
        --eval data/eval/wiki_postdump --out data/lm_eval/<model>/none
    python -m scripts.eval_lm --arm rag --index data/wiki_index/wikipedia_1k \\
        --model Qwen/Qwen2.5-0.5B-Instruct --eval data/eval/wiki_postdump \\
        --out data/lm_eval/<model>/rag_wikipedia_1k
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.provenance import git_state, sha256_file

log = logging.getLogger("eval_lm")

WINDOW = 1024
TARGET = 512
LEAK_NGRAM = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["none", "rag"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--eval", type=Path, required=True, help="dir with articles.jsonl + manifest.json")
    p.add_argument("--index", type=Path, help="index dir (required for --arm rag)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=4, help="neighbors prepended per scored chunk")
    p.add_argument("--neighbor-span", choices=["chunk", "chunk+next"], default="chunk+next")
    p.add_argument("--search-depth", type=int, default=16, help="candidates retrieved before the leakage filter")
    p.add_argument("--max-articles", type=int, default=None, help="first N articles only (smoke runs)")
    p.add_argument("--batch-size", type=int, default=16, help="sequences per forward pass")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = p.parse_args()
    if args.arm == "rag" and args.index is None:
        p.error("--arm rag needs --index")
    return args


def ngrams(ids: list[int], n: int) -> set[tuple[int, ...]]:
    return {tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)}


class Scorer:
    """Summed per-token NLL (nats) of the last n tokens of each sequence."""

    def __init__(self, model_name: str, dtype: str, pad_id: int) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype)).to("cuda").eval()
        self.decoder = self.model.get_decoder()
        self.head = self.model.get_output_embeddings()
        self.pad_id = pad_id

    @torch.no_grad()
    def per_token_nll(self, seqs: list[list[int]], n_scored: list[int]) -> list[np.ndarray]:
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), length), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        hidden = self.decoder(input_ids=ids, attention_mask=mask).last_hidden_state
        out = []
        for j, s in enumerate(seqs):
            n, m = len(s), n_scored[j]
            logits = self.head(hidden[j, n - m - 1 : n - 1]).float()
            lp = torch.log_softmax(logits, dim=-1).gather(1, ids[j, n - m : n, None])
            out.append((-lp.squeeze(1)).cpu().numpy())
        return out


class NeighborStore:
    """Chunk texts from an index, turned into token spans the way the GCCA arm reads them."""

    def __init__(self, index_dir: Path, tokenizer, span: str, chunk_size: int) -> None:
        self.texts: list[str] = []
        self.doc_ids: list[str] = []
        with (index_dir / "chunks.jsonl").open() as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                assert rec["chunk_id"] == i, f"chunk_id {rec['chunk_id']} on line {i}"
                self.texts.append(rec["text"])
                self.doc_ids.append(rec["doc_id"])
        self.tokenizer = tokenizer
        self.span = span
        self.max_tokens = 2 * chunk_size if span == "chunk+next" else chunk_size
        self._cache: dict[int, list[int]] = {}

    def tokens(self, cid: int) -> list[int]:
        if cid not in self._cache:
            text = self.texts[cid]
            nxt = cid + 1
            if self.span == "chunk+next" and nxt < len(self.texts) and self.doc_ids[nxt] == self.doc_ids[cid]:
                # No separator: the continuation's first token already carries its leading space.
                text += self.texts[nxt]
            self._cache[cid] = self.tokenizer(text, add_special_tokens=False)["input_ids"][: self.max_tokens]
        return self._cache[cid]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    sep = tokenizer("\n\n", add_special_tokens=False)["input_ids"]

    articles = [json.loads(line) for line in (args.eval / "articles.jsonl").open()]
    if args.max_articles:
        articles = articles[: args.max_articles]
    tokens = [tokenizer(a["text"], add_special_tokens=False)["input_ids"] for a in articles]
    windows = [(ai, s) for ai, ids in enumerate(tokens) for s in range(0, len(ids) - WINDOW + 1, TARGET)]
    log.info("%d articles -> %d windows (%d scored tokens)", len(articles), len(windows), len(windows) * TARGET)

    index_manifest = None
    chunk_size = 64
    if args.arm == "rag":
        index_manifest = json.loads((args.index / "manifest.json").read_text())
        chunk_size = index_manifest["chunking"]["chunk_size"]
        assert index_manifest["chunking"]["tokenizer"].split("/")[-1].startswith("Qwen2.5"), \
            "eval tokenizer must match the tokenizer the index was chunked with"
    assert TARGET % chunk_size == 0
    n_chunks = TARGET // chunk_size

    neighbors: dict[tuple[int, int], list[int]] = {}
    store = None
    if args.arm == "rag":
        # Query for Ti is T(i-1): the 64 tokens just before it, already known when Ti is predicted.
        keys, queries = [], []
        for wi, (ai, s) in enumerate(windows):
            for i in range(n_chunks):
                c = s + TARGET + i * chunk_size
                keys.append((wi, i))
                queries.append(tokenizer.decode(tokens[ai][c - chunk_size : c]))
        encoder = SentenceTransformer(index_manifest["embedding"]["model"], device="cuda")
        q_emb = encoder.encode(queries, batch_size=512, normalize_embeddings=True, convert_to_numpy=True)
        del encoder
        index = faiss.read_index(str(args.index / "index.faiss"))
        assert index.d == q_emb.shape[1]
        t0 = time.time()
        _, ids = index.search(q_emb, args.search_depth)
        log.info("retrieved %d queries x %d in %.1fs", len(queries), args.search_depth, time.time() - t0)
        neighbors = {key: [int(c) for c in row if c >= 0] for key, row in zip(keys, ids)}
        store = NeighborStore(args.index, tokenizer, args.neighbor_span, chunk_size)

    scorer = Scorer(args.model, args.dtype, pad_id)

    # Build every scored sequence up front: (window, chunk index or None for all chunks, token ids, n scored).
    jobs: list[tuple[int, int | None, list[int], int]] = []
    records = []
    for wi, (ai, s) in enumerate(windows):
        ids = tokens[ai]
        target = ids[s + TARGET : s + WINDOW]
        rec = {
            "article_id": articles[ai]["id"],
            "window_start": s,
            "target_bytes": len(tokenizer.decode(target).encode("utf-8")),
            "nll_chunks": [0.0] * n_chunks,
        }
        if args.arm == "none":
            jobs.append((wi, None, ids[s : s + WINDOW], TARGET))
        else:
            leak = ngrams(target, LEAK_NGRAM)
            rec["retrieved"], rec["dropped"] = [], []
            for i in range(n_chunks):
                kept, dropped = [], 0
                for cid in neighbors[(wi, i)]:
                    if len(kept) == args.k:
                        break
                    if ngrams(store.tokens(cid), LEAK_NGRAM) & leak:
                        dropped += 1
                        continue
                    kept.append(cid)
                rec["retrieved"].append(kept)
                rec["dropped"].append(dropped)
                context: list[int] = []
                for cid in reversed(kept):  # best neighbor last, closest to the text
                    context += store.tokens(cid) + sep
                c = s + TARGET + i * chunk_size
                jobs.append((wi, i, context + ids[s : c + chunk_size], chunk_size))
        records.append(rec)

    jobs.sort(key=lambda j: len(j[2]))  # similar lengths per batch: less padding
    for b in range(0, len(jobs), args.batch_size):
        batch = jobs[b : b + args.batch_size]
        nlls = scorer.per_token_nll([j[2] for j in batch], [j[3] for j in batch])
        for (wi, i, _, _), nll in zip(batch, nlls):
            if i is None:
                records[wi]["nll_chunks"] = [float(x) for x in nll.reshape(n_chunks, chunk_size).sum(1)]
            else:
                records[wi]["nll_chunks"][i] = float(nll.sum())
        if (b // args.batch_size) % 200 == 0:
            log.info("scored %d / %d sequences", b + len(batch), len(jobs))

    args.out.mkdir(parents=True, exist_ok=True)
    windows_path = args.out / "windows.jsonl"
    with windows_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    total_nll = sum(sum(r["nll_chunks"]) for r in records)
    total_tokens = len(records) * TARGET
    total_bytes = sum(r["target_bytes"] for r in records)
    eval_manifest = args.eval / "manifest.json"
    summary = {
        "command": " ".join([sys.executable, "-m", "scripts.eval_lm", *sys.argv[1:]]),
        "git": git_state(),
        "code_sha256": {"scripts/eval_lm.py": sha256_file(Path(__file__))},
        "arm": args.arm,
        "model": {"name": args.model, "revision": getattr(scorer.model.config, "_commit_hash", None), "dtype": args.dtype},
        "eval": {"path": str(args.eval), "manifest_sha256": sha256_file(eval_manifest),
                 "articles_sha256": json.loads(eval_manifest.read_text())["files"]["articles.jsonl"],
                 "n_articles": len(articles), "max_articles": args.max_articles},
        "protocol": {"window": WINDOW, "target": TARGET, "chunk_size": chunk_size, "leak_ngram": LEAK_NGRAM},
        "retrieval": None if args.arm == "none" else {
            "index": str(args.index),
            "index_faiss_sha256": index_manifest["files"]["index.faiss"],
            "index_n_docs": index_manifest["sampling"]["n_docs"],
            "index_n_vectors": index_manifest["index"]["n_vectors"],
            "encoder": index_manifest["embedding"]["model"],
            "k": args.k, "neighbor_span": args.neighbor_span, "search_depth": args.search_depth,
            "spans_used": sum(len(x) for r in records for x in r["retrieved"]),
            "spans_dropped_leak": sum(sum(r["dropped"]) for r in records),
            "chunks_short_of_k": sum(len(x) < args.k for r in records for x in r["retrieved"]),
        },
        "totals": {"windows": len(records), "scored_tokens": total_tokens, "scored_bytes": total_bytes, "nll_nats": total_nll},
        "bpb": total_nll / (math.log(2) * total_bytes),
        "ppl": math.exp(total_nll / total_tokens),
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__},
        "files": {"windows.jsonl": sha256_file(windows_path)},
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("bpb %.4f  ppl %.3f  (%s)", summary["bpb"], summary["ppl"], args.out)


if __name__ == "__main__":
    main()
