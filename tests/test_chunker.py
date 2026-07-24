import pytest

from ingestion.chunker import is_navigation_chunk, split_text


def test_split_text_keeps_overlap_and_content():
    text = "Birinci cümle. " * 100
    chunks = split_text(text, chunk_size=240, overlap=40)

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 240 for chunk in chunks)


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("metin", chunk_size=100, overlap=100)


def test_navigation_chunk_detects_dotted_table_of_contents():
    toc = (
        "1. Giriş................3 "
        "2. Kısıtlar................5 "
        "3. Gereksinimler................8"
    )

    assert is_navigation_chunk(toc)
    assert not is_navigation_chunk(
        "Sosyal ve kültürel kısıtlar kullanıcı tercihlerini etkiler."
    )
