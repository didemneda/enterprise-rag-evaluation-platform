from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    azure_model_endpoint: str
    azure_model_api_key: str
    azure_model_deployment: str
    embedding_provider: str
    embedding_model: str
    azure_embedding_endpoint: str
    azure_embedding_api_key: str
    azure_embedding_deployment: str
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    default_min_score: float

    @classmethod
    def from_env(cls) -> "Settings":
        # Streamlit keeps the Python process alive between reruns. Reload the
        # file values so edits to .env do not leave a stale endpoint/model in
        # os.environ for the lifetime of that process.
        load_dotenv(override=True)

        endpoint = os.getenv("AZURE_MODEL_ENDPOINT", "").rstrip("/")
        api_key = os.getenv("AZURE_MODEL_API_KEY", "")
        deployment = os.getenv(
            "AZURE_MODEL_DEPLOYMENT",
            "grok-4-1-fast-non-reasoning",
        )

        if not endpoint or not api_key or not deployment:
            raise ValueError(
                "AZURE_MODEL_ENDPOINT, AZURE_MODEL_API_KEY veya "
                "AZURE_MODEL_DEPLOYMENT eksik."
            )

        if not endpoint.endswith("/models"):
            endpoint += "/models"

        embedding_endpoint = (
            os.getenv("AZURE_EMBEDDING_ENDPOINT") or endpoint
        ).rstrip("/")
        if (
            ".services.ai.azure.com" in embedding_endpoint
            and not embedding_endpoint.endswith("/models")
        ):
            embedding_endpoint += "/models"

        return cls(
            azure_model_endpoint=endpoint,
            azure_model_api_key=api_key,
            azure_model_deployment=deployment,
            embedding_provider=os.getenv(
                "EMBEDDING_PROVIDER",
                "local_bge",
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            azure_embedding_endpoint=embedding_endpoint,
            azure_embedding_api_key=(
                os.getenv("AZURE_EMBEDDING_API_KEY") or api_key
            ),
            azure_embedding_deployment=os.getenv(
                "AZURE_EMBEDDING_DEPLOYMENT",
                "",
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "4")),
            default_min_score=float(os.getenv("DEFAULT_MIN_SCORE", "0.30")),
        )
