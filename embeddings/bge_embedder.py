from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=8 if self.device == "cuda" else 4,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
