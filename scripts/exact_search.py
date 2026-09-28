"""Exact inner-product top-k with PyTorch: the same neighbors as FAISS IndexFlatIP, run on the GPU.

The index's vectors are read once from the FAISS file and kept on the device in
float32. Each query batch is multiplied against the database block by block with
a running top-k, so the score matrix never exceeds query_batch x block_rows.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import torch

METHOD = "exact MIPS: float32 matmul + top-k in PyTorch, TF32 off"


class ExactIndex:
    def __init__(self, index_path: Path, device: str | None = None, block_rows: int = 1_000_000) -> None:
        index = faiss.read_index(str(index_path))
        assert isinstance(index, faiss.IndexFlat) and index.metric_type == faiss.METRIC_INNER_PRODUCT, \
            f"{index_path} is not an exact inner-product index"
        # TF32 would round the float32 products and could reorder near-tied neighbors.
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ntotal, self.d = index.ntotal, index.d
        vectors = index.reconstruct_n(0, self.ntotal)
        self.starts = list(range(0, self.ntotal, block_rows))
        self.blocks = [torch.from_numpy(vectors[s : s + block_rows]).to(self.device) for s in self.starts]

    @torch.no_grad()
    def search(self, queries: np.ndarray, k: int, query_batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """``(scores, ids)`` of shape ``(n, k)``, best first, like ``faiss.Index.search``."""
        assert queries.shape[1] == self.d, f"query dim {queries.shape[1]} != index dim {self.d}"
        assert k <= self.ntotal, f"k={k} exceeds the {self.ntotal} vectors in the index"
        out_s, out_i = [], []
        for b in range(0, len(queries), query_batch):
            q = torch.from_numpy(np.ascontiguousarray(queries[b : b + query_batch], dtype=np.float32)).to(self.device)
            best_s = best_i = None
            for start, block in zip(self.starts, self.blocks):
                s, i = torch.topk(q @ block.T, min(k, block.shape[0]), dim=1)
                i = i + start
                if best_s is None:
                    best_s, best_i = s, i
                else:
                    best_s, pos = torch.topk(torch.cat([best_s, s], dim=1), k, dim=1)
                    best_i = torch.gather(torch.cat([best_i, i], dim=1), 1, pos)
            out_s.append(best_s.cpu().numpy())
            out_i.append(best_i.cpu().numpy())
        return np.concatenate(out_s), np.concatenate(out_i).astype(np.int64)

    def close(self) -> None:
        self.blocks = []
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
