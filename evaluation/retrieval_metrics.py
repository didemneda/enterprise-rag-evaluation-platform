def _key(item):
    return item["source"].strip().lower(), int(item["page"])


def precision_at_k(retrieved, relevant_sources, k):
    if k <= 0:
        return 0.0
    relevant = {_key(item) for item in relevant_sources}
    hits = sum(_key(item) in relevant for item in retrieved[:k])
    return hits / k


def recall_at_k(retrieved, relevant_sources, k):
    relevant = {_key(item) for item in relevant_sources}
    if not relevant:
        return 0.0

    found = {
        _key(item)
        for item in retrieved[:k]
        if _key(item) in relevant
    }
    return len(found) / len(relevant)


def reciprocal_rank(retrieved, relevant_sources):
    relevant = {_key(item) for item in relevant_sources}

    for rank, item in enumerate(retrieved, start=1):
        if _key(item) in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved, relevant_sources, k):
    return 1.0 if recall_at_k(retrieved, relevant_sources, k) > 0 else 0.0
