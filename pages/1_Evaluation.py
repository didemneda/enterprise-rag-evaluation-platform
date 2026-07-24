from dataclasses import replace
from pathlib import Path

import streamlit as st

from core.config import Settings
from core.runtime import get_pipeline
from evaluation.evaluator import load_test_cases, run_evaluation

st.set_page_config(
    page_title="RAG Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("RAG Evaluation Dashboard")

settings = Settings.from_env()
provider_options = {
    "Local BGE-M3": "local_bge",
    "Azure Embedding": "azure",
}
default_provider = next(
    (
        label
        for label, value in provider_options.items()
        if value == settings.embedding_provider
    ),
    "Local BGE-M3",
)
provider_label = st.sidebar.selectbox(
    "Embedding sağlayıcısı",
    options=list(provider_options),
    index=list(provider_options).index(default_provider),
    key="embedding_provider_label",
)
settings = replace(
    settings,
    embedding_provider=provider_options[provider_label],
)
if settings.embedding_provider == "azure":
    azure_embedding_deployment = st.sidebar.text_input(
        "Azure embedding deployment",
        value=settings.azure_embedding_deployment,
        key="azure_embedding_deployment",
        help="Azure AI Foundry'deki embedding deployment adı.",
    )
    settings = replace(
        settings,
        azure_embedding_deployment=azure_embedding_deployment.strip(),
    )

try:
    pipeline = get_pipeline(settings)
except Exception as exc:
    st.error(f"Embedding başlatma hatası: {exc}")
    st.stop()

st.info(
    "Evaluation için önce ana sayfada belgeyi indeksle. "
    "Ana sayfa ve Evaluation aynı vektör indeksini kullanır."
)
st.metric("İndekslenen chunk sayısı", pipeline.chunk_count)

if pipeline.chunk_count == 0:
    st.warning(
        "Vektör indeksi boş. Ana sayfaya dönüp PDF'yi yükleyin ve "
        "'Belgeleri indeksle' düğmesine basın."
    )

uploaded_test_set = st.file_uploader(
    "Golden test set JSON yükle",
    type=["json"],
)

top_k = st.slider("Evaluation top-k", 1, 10, 5)

generate_answers = st.checkbox(
    "Her test için model cevabı üret",
    value=False,
    help=(
        "Kapalıyken yalnızca retrieval metrikleri hesaplanır ve Grok kotası "
        "kullanılmaz. Açılması her test vakası için en az bir çağrı oluşturur."
    ),
)

use_llm_judge = st.checkbox(
    "LLM judge metriklerini hesapla",
    help="Ek Grok çağrısı ve maliyet oluşturur.",
    disabled=not generate_answers,
)

if uploaded_test_set and st.button(
    "Değerlendirmeyi çalıştır",
    type="primary",
    disabled=pipeline.chunk_count == 0,
):
    temp_path = Path("evaluation/_uploaded_test_cases.json")
    temp_path.write_bytes(uploaded_test_set.getvalue())

    try:
        test_cases = load_test_cases(temp_path)
        dataframe = run_evaluation(
            pipeline=pipeline,
            test_cases=test_cases,
            top_k=top_k,
            minimum_score=0.0,
            generate_answers=generate_answers,
            use_llm_judge=use_llm_judge,
        )

        st.subheader("Özet")

        summary_metrics = [
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "hit_rate_at_k",
        ]

        columns = st.columns(len(summary_metrics))
        for column, metric in zip(columns, summary_metrics):
            column.metric(metric, f"{dataframe[metric].mean():.3f}")

        st.dataframe(dataframe, use_container_width=True)

        st.download_button(
            "Sonuç CSV'sini indir",
            data=dataframe.to_csv(index=False).encode("utf-8-sig"),
            file_name="rag_evaluation_results.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Evaluation hatası: {exc}")
