"""BM25 with Lucene's scoring formula (as used by Pyserini in ProgramFC: k1=0.9, b=0.4), vectorised with scipy.

score(q, d) = sum_{t in q} idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))                     (Lucene BM25Similarity)
Query terms are counted with multiplicity (each occurrence is a clause in Lucene's boolean query).
Analysis approximates Lucene's English analyser: lower-casing, alphanumeric tokens, English stop words removed,
Porter stemming.
"""
import re
import numpy as np
import scipy.sparse as sps

_TOK = re.compile(r"[a-z0-9]+")


def _stemmer():
    from nltk.stem.porter import PorterStemmer
    return PorterStemmer()


def _stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return ENGLISH_STOP_WORDS


class BM25:
    def __init__(self, docs, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.stem, self.stop = _stemmer(), _stopwords()
        self._cache = {}
        toks = [self.analyse(d) for d in docs]
        vocab = {}
        rows, cols, vals = [], [], []
        for i, ts in enumerate(toks):
            counts = {}
            for t in ts:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                rows.append(i); cols.append(vocab.setdefault(t, len(vocab))); vals.append(c)
        self.vocab = vocab
        N = len(docs)
        tf = sps.csr_matrix((np.array(vals, float), (rows, cols)), shape=(N, len(vocab)))
        dl = np.array([len(t) for t in toks], float)
        avgdl = dl.mean() if N else 1.0
        df = np.bincount(np.array(cols, int), minlength=len(vocab)) if cols else np.zeros(len(vocab))
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        tfc = tf.tocoo()
        norm = k1 * (1 - b + b * dl[tfc.row] / avgdl)
        w = self.idf[tfc.col] * tfc.data * (k1 + 1) / (tfc.data + norm)
        self.W = sps.csr_matrix((w, (tfc.row, tfc.col)), shape=tf.shape).T.tocsr()  # V x N

    def analyse(self, text):
        out = []
        for t in _TOK.findall((text or "").lower()):
            if t in self.stop:
                continue
            s = self._cache.get(t)
            if s is None:
                s = self._cache[t] = self.stem.stem(t)
            out.append(s)
        return out

    def query_matrix(self, queries):
        rows, cols, vals = [], [], []
        for i, q in enumerate(queries):
            for t in self.analyse(q):
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i); cols.append(j); vals.append(1.0)
        return sps.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)))

    def scores(self, queries):
        """dense (Q x N) score matrix"""
        return (self.query_matrix(queries) @ self.W).toarray()

    def topk(self, queries, k, bs=512):
        out = []
        for s in range(0, len(queries), bs):
            S = self.scores(queries[s:s + bs])
            for row in S:  # descending score, ties broken by document index (deterministic)
                out.append([int(i) for i in np.lexsort((np.arange(len(row)), -row))[:k]])
        return out
