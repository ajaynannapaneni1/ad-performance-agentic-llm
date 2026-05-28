"""
RAG Vector Store — FAISS-backed semantic retrieval over ad performance knowledge base.
Chunks include: industry benchmarks, campaign strategy docs, historical trend summaries.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = Path(__file__).parents[2] / "data" / "vector_store.pkl"

# ── Knowledge base documents ──────────────────────────────────────────────────
# In production these are loaded from S3 / a document pipeline.
# Here we embed them inline for portability.

KNOWLEDGE_BASE = [
    {
        "id": "bench-001",
        "text": "Industry average CTR for Search campaigns is 2–5%. CTR below 1.5% typically signals weak ad copy or poor keyword-to-ad alignment.",
        "tags": ["benchmark", "ctr", "search"],
    },
    {
        "id": "bench-002",
        "text": "A healthy ROAS for eCommerce display campaigns is 3–6x. ROAS below 2x often indicates broad targeting or high CPCs outpacing revenue.",
        "tags": ["benchmark", "roas", "display"],
    },
    {
        "id": "bench-003",
        "text": "CPC benchmarks vary by vertical: Retail averages $0.40–$0.80, Finance $2–$5, SaaS $1.50–$4. Comparing CPC without vertical context is misleading.",
        "tags": ["benchmark", "cpc", "vertical"],
    },
    {
        "id": "strat-001",
        "text": "When CTR drops >20% week-over-week, the top causes are: ad fatigue (high frequency), keyword competition spike, or seasonal intent shift. Check impression share lost to rank vs. budget.",
        "tags": ["strategy", "ctr", "troubleshooting"],
    },
    {
        "id": "strat-002",
        "text": "Bid strategy transition from Manual CPC to Target ROAS typically requires 30–50 conversions/month per campaign to exit the learning phase within 7 days.",
        "tags": ["strategy", "bidding", "automation"],
    },
    {
        "id": "strat-003",
        "text": "Dayparting analysis: B2C campaigns typically see peak CTR 7–9 AM and 6–9 PM local time. B2B peaks 9 AM–12 PM Tuesday–Thursday. Adjusting bids ±20% during peak windows improves ROAS by 10–25%.",
        "tags": ["strategy", "dayparting", "optimization"],
    },
    {
        "id": "trend-001",
        "text": "Q4 (Oct–Dec) historically increases CPCs by 20–40% industry-wide due to holiday advertiser competition. Budget planning should account for elevated spend-to-revenue ratios.",
        "tags": ["trend", "seasonality", "q4"],
    },
    {
        "id": "trend-002",
        "text": "Mobile traffic share crossed 65% for retail queries in 2023. Campaigns with mobile bid adjustments below -30% risk significant impression loss on high-intent mobile searches.",
        "tags": ["trend", "mobile", "retail"],
    },
    {
        "id": "metric-001",
        "text": "Conversion Rate (CVR) = Conversions / Clicks. Average CVR for Search: 3–5%. Low CVR with high CTR points to landing page friction, not ad quality issues.",
        "tags": ["metric", "cvr", "landing-page"],
    },
    {
        "id": "metric-002",
        "text": "Impression Share (IS) = Impressions received / Eligible impressions. IS lost to budget indicates underfunding; IS lost to rank indicates Quality Score or bid issues.",
        "tags": ["metric", "impression-share", "quality-score"],
    },
]


# ── Embedding (lightweight TF-IDF fallback; swap for real embeddings in prod) ──

class TFIDFEmbedder:
    """Lightweight TF-IDF embedder for demo/offline use.
    In production, replace with:
      - AWS Bedrock Titan Embeddings
      - OpenAI text-embedding-3-small
      - SentenceTransformers all-MiniLM-L6-v2
    """

    def __init__(self):
        self.vocab: dict = {}
        self.idf: np.ndarray = None
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, docs: List[str]):
        import math
        N = len(docs)
        df = {}
        tokenized = [self._tokenize(d) for d in docs]
        for tokens in tokenized:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        self.idf = np.array([
            math.log((N + 1) / (df[t] + 1)) + 1 for t in sorted(df)
        ])
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        V = len(self.vocab)
        mat = np.zeros((len(texts), V))
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            for t in tokens:
                if t in self.vocab:
                    mat[i, self.vocab[t]] += 1
            norm = np.linalg.norm(mat[i])
            if norm > 0:
                mat[i] = mat[i] / norm * self.idf
        return mat


class AdVectorStore:
    def __init__(self):
        self.embedder = TFIDFEmbedder()
        self.docs: List[dict] = []
        self.vectors: np.ndarray = None

    def build(self, documents: List[dict] = None):
        self.docs = documents or KNOWLEDGE_BASE
        texts = [d["text"] for d in self.docs]
        self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts)
        logger.info(f"Vector store built: {len(self.docs)} documents, dim={self.vectors.shape[1]}")

    def save(self, path: Path = VECTOR_STORE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embedder": self.embedder, "docs": self.docs, "vectors": self.vectors}, f)
        logger.info(f"Vector store saved to {path}")

    def load(self, path: Path = VECTOR_STORE_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.embedder = data["embedder"]
        self.docs = data["docs"]
        self.vectors = data["vectors"]

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        q_vec = self.embedder.encode([query])[0]
        scores = self.vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.docs[i]) for i in top_idx]


# ── Singleton store ───────────────────────────────────────────────────────────

_store: AdVectorStore = None


def _get_store() -> AdVectorStore:
    global _store
    if _store is None:
        _store = AdVectorStore()
        if VECTOR_STORE_PATH.exists():
            _store.load()
        else:
            _store.build()
            _store.save()
    return _store


def retrieve_ad_context(question: str, top_k: int = 3) -> str:
    """Retrieve top-k relevant knowledge chunks for a question."""
    store = _get_store()
    results = store.search(question, top_k=top_k)
    if not results:
        return "No relevant context found."
    lines = []
    for score, doc in results:
        lines.append(f"[{doc['id']} | relevance={score:.3f}]\n{doc['text']}")
    return "\n\n".join(lines)
