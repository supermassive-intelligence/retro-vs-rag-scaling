"""Build a FAISS index over N raw Wikipedia articles chunked into 64-token pieces.

Chunking matches ../maailma's ingestion.chunk_document (Qwen2.5 tokenizer,
non-overlapping windows, trailing partial chunk kept) and embedding matches its
default encoder (BAAI/bge-small-en-v1.5), so chunks are interchangeable between
the RAG baseline and the GCCA arm.

    python -m scripts.build_wiki_index --n-docs 1000 --out data/wiki_index/wikipedia_1k
    python -m scripts.build_wiki_index --n-docs 100000 --out data/wiki_index/wikipedia_100k --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.provenance import git_state, sha256_file

log = logging.getLogger("build_wiki_index")

DATASET = "wikimedia/wikipedia"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-docs", type=int, required=True, help="number of Wikipedia articles to index")
    p.add_argument("--out", type=Path, default=None, help="output dir (default: data/wiki_index/n<N>)")
    p.add_argument("--dataset-config", default="20231101.en")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    return p.parse_args()


def chunk_document(text: str, tokenizer, m: int) -> list[tuple[str, int]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [(tokenizer.decode(ids[s : s + m], skip_special_tokens=True), len(ids[s : s + m])) for s in range(0, len(ids), m)]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = args.out or Path("data/wiki_index") / f"n{args.n_docs}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    git = git_state()  # at start: the tree can change while a long run is going

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Streaming shuffle is buffer-based: deterministic for a fixed (seed, buffer), not uniform over all of Wikipedia.
    ds = load_dataset(DATASET, args.dataset_config, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer).take(args.n_docs)

    chunks_path = out / "chunks.jsonl"
    texts: list[str] = []
    n_docs = 0
    n_tokens = 0
    with chunks_path.open("w") as f:
        for doc in tqdm(ds, total=args.n_docs, desc="chunking"):
            n_docs += 1
            for pos, (text, n_tok) in enumerate(chunk_document(doc["text"], tokenizer, args.chunk_size)):
                if not text.strip():
                    continue
                rec = {
                    "chunk_id": len(texts),
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "position": pos,
                    "n_tokens": n_tok,
                    "text": text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                texts.append(text)
                n_tokens += n_tok

    if n_docs < args.n_docs:
        log.warning("dataset exhausted: got %d of %d requested docs", n_docs, args.n_docs)
    log.info("%d docs -> %d chunks, %d tokens", n_docs, len(texts), n_tokens)

    encoder = SentenceTransformer(args.embedding_model, device=args.device)
    emb = encoder.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    # Inner product over L2-normalized vectors == cosine similarity; exact search keeps small-N results noise-free.
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    index_path = out / "index.faiss"
    faiss.write_index(index, str(index_path))

    manifest = {
        "command": " ".join([sys.executable, "-m", "scripts.build_wiki_index", *sys.argv[1:]]),
        "git": git,
        "dataset": {"name": DATASET, "config": args.dataset_config, "split": "train"},
        "sampling": {"seed": args.seed, "shuffle_buffer": args.shuffle_buffer, "n_docs_requested": args.n_docs, "n_docs": n_docs},
        "chunking": {"tokenizer": args.tokenizer, "chunk_size": args.chunk_size, "n_chunks": len(texts), "n_tokens": n_tokens},
        "embedding": {"model": args.embedding_model, "dim": int(emb.shape[1]), "normalized": True, "device": str(encoder.device)},
        "index": {"type": "IndexFlatIP", "n_vectors": int(index.ntotal)},
        "files": {"chunks.jsonl": sha256_file(chunks_path), "index.faiss": sha256_file(index_path)},
        "elapsed_s": round(time.time() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
