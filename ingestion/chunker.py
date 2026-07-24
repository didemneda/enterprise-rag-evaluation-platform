from __future__ import annotations

import re


def is_navigation_chunk(text: str) -> bool:
    """Detect table-of-contents chunks dominated by dotted leaders."""
    dotted_leaders = re.findall(r"\.{5,}", text)
    return len(dotted_leaders) >= 3


def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 180,
) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size overlap değerinden büyük olmalıdır.")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        tentative_end = min(start + chunk_size, len(text))
        end = tentative_end

        if tentative_end < len(text):
            candidate = text[start:tentative_end]
            boundary = max(
                candidate.rfind(". "),
                candidate.rfind("? "),
                candidate.rfind("! "),
                candidate.rfind("; "),
            )
            if boundary >= int(chunk_size * 0.65):
                end = start + boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks
