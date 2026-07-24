from dataclasses import replace

import streamlit as st

from core.config import Settings
from core.runtime import get_pipeline

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="📄",
    layout="wide",
)

st.title("Enterprise Document Assistant")
st.caption("BGE-M3 + Azure-hosted Grok + evaluation-driven RAG")


try:
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
    pipeline = get_pipeline(settings)
except Exception as exc:
    st.error(f"Başlatma hatası: {exc}")
    st.stop()


with st.sidebar:
    st.header("Belge yönetimi")

    uploaded_files = st.file_uploader(
        "PDF yükle",
        type=["pdf"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", 400, 1600, 1000, 100)
    overlap = st.slider("Overlap", 50, 400, 180, 10)

    if st.button("Belgeleri indeksle", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Önce PDF yüklemelisin.")
        else:
            progress = st.progress(0, text="Hazırlanıyor...")
            try:
                stats = pipeline.ingest(
                    uploaded_files,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    progress_callback=lambda value, message: progress.progress(
                        value,
                        text=message,
                    ),
                )
                progress.progress(100, text="Tamamlandı.")
                st.success(
                    f"{stats['document_count']} belge, "
                    f"{stats['chunk_count']} chunk indekslendi."
                )
            except Exception as exc:
                st.error(str(exc))

    top_k = st.slider("Top-k", 2, 10, 4)
    minimum_score = st.slider(
        "Minimum benzerlik",
        0.0,
        1.0,
        0.30,
        0.05,
    )

    st.metric("Chunk sayısı", pipeline.chunk_count)
    st.caption(f"Embedding: {provider_label}")
    st.caption(f"Embedding cihazı: {pipeline.embedding_device}")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Belgelere soru sor...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            result = pipeline.ask(
                question,
                top_k=top_k,
                minimum_score=minimum_score,
            )

            st.markdown(result["answer"])

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Retrieval",
                f"{result['retrieval_ms']:.0f} ms",
            )
            col2.metric(
                "Generation",
                f"{result['generation_ms']:.0f} ms",
            )
            col3.metric(
                "Toplam",
                f"{result['total_ms']:.0f} ms",
            )

            if result["sources"]:
                with st.expander("Retrieval debugger"):
                    for source in result["sources"]:
                        st.markdown(
                            f"### #{source['rank']} "
                            f"{source['source']} · Sayfa {source['page']}"
                        )
                        st.write(f"Skor: `{source['score']:.4f}`")
                        st.write(source["content"])

                with st.expander("Model promptu"):
                    st.code(result["prompt"])
        except Exception as exc:
            result = {"answer": f"Hata: {exc}"}
            st.error(result["answer"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
