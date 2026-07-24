from __future__ import annotations

import json
import re

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from llm.prompts import JUDGE_SYSTEM_PROMPT, SYSTEM_PROMPT


class GrokClient:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.client = ChatCompletionsClient(
            endpoint=settings.azure_model_endpoint,
            credential=AzureKeyCredential(settings.azure_model_api_key),
        )

    def generate_answer(self, question, retrieved):
        context = "\n\n".join(
            f"[KAYNAK {item.rank}]\n"
            f"Belge: {item.chunk.source}\n"
            f"Sayfa: {item.chunk.page}\n"
            f"İçerik: {item.chunk.content}"
            for item in retrieved
        )
        prompt = f"KAYNAKLAR:\n{context}\n\nSORU:\n{question}"

        response = self.client.complete(
            model=self.settings.azure_model_deployment,
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                UserMessage(content=prompt),
            ],
            temperature=0,
            max_tokens=900,
        )
        return response.choices[0].message.content or "Boş cevap", prompt

    def judge(
        self,
        question: str,
        answer: str,
        context: str,
        reference_answer: str,
    ) -> dict[str, float]:
        prompt = f"""
SORU:
{question}

KAYNAKLAR:
{context}

REFERANS CEVAP:
{reference_answer}

MODEL CEVABI:
{answer}

Şu JSON şemasında değerlendir:
{{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "context_precision": 0.0,
  "context_recall": 0.0,
  "answer_correctness": 0.0
}}
""".strip()

        response = self.client.complete(
            model=self.settings.azure_model_deployment,
            messages=[
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                UserMessage(content=prompt),
            ],
            temperature=0,
            max_tokens=400,
        )

        raw = response.choices[0].message.content or "{}"
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

        return {
            key: max(0.0, min(1.0, float(value)))
            for key, value in data.items()
        }
