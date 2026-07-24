from __future__ import annotations

import numpy as np
from azure.ai.inference import EmbeddingsClient
from azure.ai.inference.models import EmbeddingInputType
from azure.core.credentials import AzureKeyCredential


class AzureEmbedder:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        batch_size: int = 16,
    ) -> None:
        if not endpoint or not api_key or not deployment:
            raise ValueError(
                "Azure embedding için AZURE_EMBEDDING_DEPLOYMENT gerekli. "
                "İstenirse AZURE_EMBEDDING_ENDPOINT ve "
                "AZURE_EMBEDDING_API_KEY de ayrı olarak tanımlanabilir."
            )

        self.device = "azure"
        self.deployment = deployment
        self.batch_size = batch_size
        self.client = EmbeddingsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            model=deployment,
        )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def _encode(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
    ) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), self.batch_size):
            response = self.client.embed(
                input=texts[start : start + self.batch_size],
                input_type=input_type,
            )
            batch = np.asarray(
                [item.embedding for item in response.data],
                dtype=np.float32,
            )
            batches.append(batch)

        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return self._normalize(np.vstack(batches)).astype(np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, EmbeddingInputType.DOCUMENT)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(
            [text],
            EmbeddingInputType.QUERY,
        )[0]
