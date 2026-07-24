from __future__ import annotations

import numpy as np
from core.models import DocumentChunk, RetrievedChunk


class LocalVectorStore:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._matrix: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Boş chunk listesiyle index oluşturulamaz.")

        self._chunks = chunks
        self._matrix = np.vstack(
            [chunk.embedding for chunk in chunks]
        ).astype(np.float32)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        minimum_score: float = 0.30,
    ) -> list[RetrievedChunk]:
        if self._matrix is None:
            return []

        scores = self._matrix @ query_vector
        best_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedChunk(
                chunk=self._chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(best_indices, start=1)
            if float(scores[index]) >= minimum_score
        ]
