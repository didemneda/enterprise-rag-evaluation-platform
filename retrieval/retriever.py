from __future__ import annotations

from retrieval.vector_store import LocalVectorStore


class Retriever:
    def __init__(
        self,
        embedder,
        store: LocalVectorStore,
    ) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        question: str,
        top_k: int = 4,
        minimum_score: float = 0.30,
    ):
        query_vector = self.embedder.encode_query(question)
        return self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            minimum_score=minimum_score,
        )
