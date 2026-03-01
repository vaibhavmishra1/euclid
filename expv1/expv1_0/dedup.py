from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .utils import normalize_text_for_dedup


@dataclass
class Deduper:
    lexical: bool = True
    embedding: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cosine_threshold: float = 0.92
    max_nn: int = 10

    _seen_lex: set[str] = field(default_factory=set, init=False)
    _texts: List[str] = field(default_factory=list, init=False)
    _embs: Optional[np.ndarray] = field(default=None, init=False)
    _embedder: object = field(default=None, init=False)

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer  # optional dep

        self._embedder = SentenceTransformer(self.embedding_model)

    def _encode(self, texts: List[str]) -> np.ndarray:
        self._ensure_embedder()
        embs = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)

    def check_duplicate(self, text: str) -> Optional[str]:
        """
        Returns a string reason if duplicate, else None.
        """
        if self.lexical:
            norm = normalize_text_for_dedup(text)
            if norm in self._seen_lex:
                return "lexical_duplicate"

        if self.embedding:
            try:
                q = self._encode([text])[0]
                if self._embs is not None and len(self._texts) > 0:
                    # brute force cosine similarity since embeddings are normalized
                    sims = self._embs @ q
                    topk = int(min(self.max_nn, sims.shape[0]))
                    if topk > 0:
                        nn = np.partition(sims, -topk)[-topk:]
                        if float(np.max(nn)) >= self.cosine_threshold:
                            return "embedding_duplicate"
            except Exception:
                # If embedder isn't available, silently fall back to lexical.
                pass

        return None

    def add(self, text: str) -> None:
        if self.lexical:
            self._seen_lex.add(normalize_text_for_dedup(text))
        if self.embedding:
            try:
                e = self._encode([text])
                self._texts.append(text)
                if self._embs is None:
                    self._embs = e
                else:
                    self._embs = np.concatenate([self._embs, e], axis=0)
            except Exception:
                # embedding is optional; don't fail the whole run
                self._texts.append(text)

