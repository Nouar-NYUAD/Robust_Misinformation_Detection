"""Dense / sparse retrieval over the verified-claim archive, aggregated to claim level.

Encoders
  me5   intfloat/multilingual-e5-large (dense; the preprint's open-source encoder)
  bgem3 BAAI/bge-m3 (dense + sparse lexical weights; ours)
  air   BGE-M3 with the dense encoder replaced by the attack-invariant fine-tuned one (train_air.py);
        sparse lexical weights still come from the original BGE-M3

build: python retrieval.py --dataset pubhealth --encoder bgem3
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from common import ROOT, kb

IDX = ROOT / "index"
import os
AIR_PATH = os.environ.get("AIR_PATH", "/scratch/naa9497/misinfo/models/bgem3-air")


class Encoder:
    def __init__(self, name, device="cuda"):
        self.name = name
        if name == "me5":
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
            self.m.max_seq_length = 512
        elif name in ("bgem3", "air"):
            from FlagEmbedding import BGEM3FlagModel
            self.m = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=[device])
            if name == "air":
                from sentence_transformers import SentenceTransformer
                self.air = SentenceTransformer(AIR_PATH, device=device)
                self.air.max_seq_length = 1024
        else:
            raise ValueError(name)

    def encode(self, texts, is_query, max_length=1024, bs=64):
        if self.name == "me5":
            pre = "query: " if is_query else "passage: "
            d = self.m.encode([pre + t for t in texts], batch_size=bs, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=len(texts) > 5000)
            return d.astype(np.float16), None
        out = self.m.encode(texts, batch_size=bs, max_length=max_length, return_dense=True,
                            return_sparse=True, return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float16)
        rows, cols, vals = [], [], []
        for i, lw in enumerate(out["lexical_weights"]):
            for tok, w in lw.items():
                rows.append(i); cols.append(int(tok)); vals.append(float(w))
        sparse = sps.csr_matrix((vals, (rows, cols)), shape=(len(texts), 250002), dtype=np.float32)
        if self.name == "air":
            self.air.max_seq_length = max_length
            dense = self.air.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=len(texts) > 5000).astype(np.float16)
        return dense, sparse


class Index:
    """Claim-level index: KB rows are contiguous per claim (prep_data.py), so segment-max works."""

    def __init__(self, ds, encoder_name):
        d = IDX / ds
        self.rows = kb(ds)
        self.dense = np.load(d / f"{encoder_name}_dense.npy")
        sp = d / f"{encoder_name}_sparse.npz"
        self.sparse = sps.load_npz(sp).T.tocsr() if sp.exists() else None  # V x D
        cid = [r["claim_id"] for r in self.rows]
        self.starts = np.array([0] + [i for i in range(1, len(cid)) if cid[i] != cid[i - 1]])
        self.claim_ids = [cid[s] for s in self.starts]
        self.claim_pos = {c: i for i, c in enumerate(self.claim_ids)}
        first = {}
        for i, r in enumerate(self.rows):
            first.setdefault(r["claim_id"], r)
        self.claim_rec = [first[c] for c in self.claim_ids]
        import torch
        self.torch = torch
        self.dense_t = torch.from_numpy(self.dense).cuda()

    def doc_scores(self, qd, qs=None, w_sparse=0.3):
        s = (self.torch.from_numpy(qd).cuda() @ self.dense_t.T).float().cpu().numpy()
        if qs is not None and self.sparse is not None:
            s = s + w_sparse * (qs @ self.sparse).toarray()
        return s

    def claim_scores(self, doc_scores):
        """returns (Q x C) claim scores: max over the claim record and its evidence passages."""
        cs = np.maximum.reduceat(doc_scores, self.starts, axis=1)
        return cs

    def best_evidence(self, q_doc_scores, claim_idx):
        """Best-scoring EVIDENCE passage of a claim. Row a is the claim record itself (prep_data.kb_rows puts it
        first); it carries the claim's wording, not evidence, so it is excluded. Falls back to the claim record
        only if the claim has no evidence passages."""
        a = self.starts[claim_idx]
        b = self.starts[claim_idx + 1] if claim_idx + 1 < len(self.starts) else len(self.rows)
        if b - a <= 1:
            return a
        return a + 1 + int(np.argmax(q_doc_scores[a + 1:b]))


def fuse_views(idx, qd, qsp, owner, n_queries, per_view=50, keep=30, k_rrf=60, bs=256, exclude=None):
    """Multi-view retrieval with reciprocal rank fusion.

    qd/qsp: dense / sparse encodings of all views (rows), owner[v] = query index of view v.
    For each view, archived claims are ranked by claim score (max over the claim record and its evidence
    passages); each claim in a view's top `per_view` gets 1/(k_rrf + rank) (rank from 0). For each claim we also
    keep the evidence passage that scores best under the view where the claim scores highest.
    exclude[q] (optional): archive claim index that query q must never retrieve (its own record, used only when
    building verifier training data from training claims, which ARE in the archive; test claims never are).
    Returns, per query, up to `keep` (claim_idx, evidence_row) pairs sorted by fused score (desc)."""
    fused = [dict() for _ in range(n_queries)]
    for s in range(0, len(owner), bs):
        ds_ = idx.doc_scores(qd[s:s + bs], None if qsp is None else qsp[s:s + bs])
        cs = idx.claim_scores(ds_)
        for j in range(cs.shape[0]):
            if exclude is not None and exclude[owner[s + j]] is not None:
                cs[j, exclude[owner[s + j]]] = -np.inf
            f = fused[owner[s + j]]
            top = np.argsort(-cs[j], kind="stable")[:per_view]
            top = [ci for ci in top if np.isfinite(cs[j, ci])]  # an excluded claim is never retrieved
            for rank, ci in enumerate(top):
                ci = int(ci)
                e = f.setdefault(ci, [0.0, None, -np.inf])
                e[0] += 1.0 / (k_rrf + rank)
                if cs[j, ci] > e[2]:
                    e[2] = cs[j, ci]; e[1] = idx.best_evidence(ds_[j], ci)
    return [[(ci, v[1]) for ci, v in sorted(f.items(), key=lambda x: (-x[1][0], x[0]))[:keep]] for f in fused]


def build(ds, enc_name):
    rows = kb(ds)
    enc = Encoder(enc_name)
    dense, sparse = enc.encode([r["text"] for r in rows], is_query=False, max_length=512)
    out = IDX / ds
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{enc_name}_dense.npy", dense)
    if sparse is not None:
        sps.save_npz(out / f"{enc_name}_sparse.npz", sparse)
    print(ds, enc_name, dense.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--encoder", nargs="+", default=["me5", "bgem3"], help="me5 bgem3 air")
    a = ap.parse_args()
    for ds in a.dataset:
        for e in a.encoder:
            build(ds, e)
