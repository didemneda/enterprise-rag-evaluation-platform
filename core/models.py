from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    content: str
    source: str
    page: int
    chunk_index: int
    embedding: np.ndarray | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    rank: int
