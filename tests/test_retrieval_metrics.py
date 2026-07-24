from evaluation.retrieval_metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


RETRIEVED = [
    {"source": "other.pdf", "page": 1},
    {"source": "document.pdf", "page": 3},
    {"source": "document.pdf", "page": 4},
]
RELEVANT = [
    {"source": "document.pdf", "page": 3},
    {"source": "document.pdf", "page": 4},
]


def test_retrieval_metrics():
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == 2 / 3
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == 1.0
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 3) == 1.0


def test_empty_relevant_sources_are_safe():
    assert recall_at_k(RETRIEVED, [], 3) == 0.0
    assert reciprocal_rank(RETRIEVED, []) == 0.0
    assert hit_rate_at_k(RETRIEVED, [], 3) == 0.0
