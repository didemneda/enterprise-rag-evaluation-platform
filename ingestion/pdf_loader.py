from __future__ import annotations

import io
import re
import fitz


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_pages(
    pdf_bytes: bytes,
    source_name: str,
) -> list[tuple[int, str]]:
    document = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages: list[tuple[int, str]] = []

    for page_index, page in enumerate(document):
        text = clean_text(page.get_text("text"))
        if text:
            pages.append((page_index + 1, text))

    if not pages:
        raise ValueError(
            f"{source_name} dosyasından metin çıkarılamadı. OCR gerekebilir."
        )

    return pages
