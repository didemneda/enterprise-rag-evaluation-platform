import numpy as np

from core.models import DocumentChunk
from retrieval.vector_store import LocalVectorStore


def _chunk(chunk_id, vector):
    return DocumentChunk(
        chunk_id=chunk_id,
        content=chunk_id,
        source="document.pdf",
        page=1,
        chunk_index=0,
        embedding=np.asarray(vector, dtype=np.float32),
    )


def test_vector_store_returns_cosine_order_for_normalized_vectors():
    store = LocalVectorStore()
    store.build(
        [
            _chunk("exact", [1.0, 0.0]),
            _chunk("partial", [0.8, 0.6]),
            _chunk("opposite", [-1.0, 0.0]),
        ]
    )

    results = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=2,
        minimum_score=0.0,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "exact",
        "partial",
    ]
    assert [result.rank for result in results] == [1, 2]


def test_empty_store_returns_no_results():
    store = LocalVectorStore()

    assert store.search(np.asarray([1.0], dtype=np.float32)) == []
