from __future__ import annotations

import time

from core.models import DocumentChunk
from embeddings.azure_embedder import AzureEmbedder
from embeddings.bge_embedder import BGEEmbedder
from ingestion.chunker import is_navigation_chunk, split_text
from ingestion.pdf_loader import extract_pdf_pages
from llm.grok_client import GrokClient
from retrieval.retriever import Retriever
from retrieval.vector_store import LocalVectorStore


class RAGPipeline:
    def __init__(self, settings) -> None:
        self.settings = settings
        if settings.embedding_provider == "azure":
            self.embedder = AzureEmbedder(
                endpoint=settings.azure_embedding_endpoint,
                api_key=settings.azure_embedding_api_key,
                deployment=settings.azure_embedding_deployment,
            )
        else:
            self.embedder = BGEEmbedder(settings.embedding_model)
        self.store = LocalVectorStore()
        self.retriever = Retriever(self.embedder, self.store)
        self.llm = GrokClient(settings)

    @property
    def chunk_count(self) -> int:
        return self.store.size

    @property
    def embedding_device(self) -> str:
        return self.embedder.device

    def ingest(
        self,
        uploaded_files,
        chunk_size=None,
        overlap=None,
        progress_callback=None,
    ) -> dict:
        files = list(uploaded_files)
        chunk_size = chunk_size or self.settings.chunk_size
        overlap = overlap or self.settings.chunk_overlap
        records = []
        page_count = 0

        for file_number, uploaded_file in enumerate(files, start=1):
            if progress_callback:
                progress_callback(
                    int(25 * file_number / max(len(files), 1)),
                    f"{uploaded_file.name} okunuyor...",
                )

            pages = extract_pdf_pages(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )
            page_count += len(pages)

            for page_number, page_text in pages:
                chunks = split_text(
                    page_text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                for chunk_index, content in enumerate(chunks):
                    if is_navigation_chunk(content):
                        continue
                    records.append(
                        (
                            content,
                            uploaded_file.name,
                            page_number,
                            chunk_index,
                        )
                    )

        if not records:
            raise ValueError("İndekslenecek metin bulunamadı.")

        if progress_callback:
            provider_name = (
                "Azure"
                if self.settings.embedding_provider == "azure"
                else "BGE-M3"
            )
            progress_callback(40, f"{provider_name} embedding üretiyor...")

        vectors = self.embedder.encode_documents(
            [record[0] for record in records]
        )

        chunks = [
            DocumentChunk(
                chunk_id=f"{source}-p{page}-c{chunk_index}",
                content=content,
                source=source,
                page=page,
                chunk_index=chunk_index,
                embedding=vector,
            )
            for (content, source, page, chunk_index), vector
            in zip(records, vectors)
        ]

        self.store.build(chunks)

        if progress_callback:
            progress_callback(95, "Vektör indeksi hazırlandı.")

        return {
            "document_count": len(files),
            "page_count": page_count,
            "chunk_count": len(chunks),
        }

    def ask(
        self,
        question: str,
        top_k=None,
        minimum_score=None,
        generate_answer=True,
    ) -> dict:
        top_k = top_k or self.settings.default_top_k
        if minimum_score is None:
            minimum_score = self.settings.default_min_score

        start = time.perf_counter()
        retrieved = self.retriever.retrieve(
            question=question,
            top_k=top_k,
            minimum_score=minimum_score,
        )
        retrieval_ms = (time.perf_counter() - start) * 1000

        if not retrieved:
            return {
                "answer": "Bu bilgi yüklenen belgelerde bulunamadı.",
                "sources": [],
                "retrieval_ms": retrieval_ms,
                "generation_ms": 0.0,
                "total_ms": retrieval_ms,
                "prompt": "",
            }

        sources = [
            {
                "source": item.chunk.source,
                "page": item.chunk.page,
                "score": item.score,
                "rank": item.rank,
                "preview": item.chunk.content[:320],
                "content": item.chunk.content,
            }
            for item in retrieved
        ]

        if not generate_answer:
            return {
                "answer": "",
                "sources": sources,
                "retrieval_ms": retrieval_ms,
                "generation_ms": 0.0,
                "total_ms": retrieval_ms,
                "prompt": "",
            }

        start = time.perf_counter()
        answer, prompt = self.llm.generate_answer(question, retrieved)
        generation_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": retrieval_ms + generation_ms,
            "prompt": prompt,
        }
