import streamlit as st

st.set_page_config(
    page_title="RAG Architecture",
    page_icon="🏗️",
    layout="wide",
)

st.title("RAG Architecture")

st.code(
    """
INDEXING
PDF documents
  ↓
Page-level text extraction
  ↓
Sentence-aware chunking + TOC filtering
  ↓
Local BGE-M3 OR Azure embedding
  ↓
Normalized in-memory vector index

QUERY
Question
  ↓
Query embedding (same provider)
  ↓
Cosine similarity retrieval
  ↓
Top-k grounded chunks
  ↓
Azure-hosted Grok
  ↓
Answer + page citations

EVALUATION
Golden JSON test set
  ↓
Retrieval-only metrics OR generation + LLM judge
  ↓
CSV export
""".strip(),
    language="text",
)

st.markdown(
    """
### Katmanlar

- **Ingestion:** PDF metnini sayfa bazında çıkarır.
- **Cleaning:** İçindekiler bölümündeki noktalı navigasyon parçalarını eler.
- **Chunking:** Metni cümle sınırlarını gözeterek örtüşmeli parçalara böler.
- **Embedding:** Yerel BGE-M3 veya Azure embedding sağlayıcısını kullanır.
- **Retrieval:** Normalize vektörler üzerinde cosine similarity uygular.
- **Generation:** Azure AI Inference SDK üzerinden kaynaklı cevap üretir.
- **Evaluation:** Retrieval ve generation kalitesini ayrı ayrı ölçer.

> Embedding sağlayıcısı değiştirildiğinde belgeler yeniden indekslenmelidir.
"""
)
