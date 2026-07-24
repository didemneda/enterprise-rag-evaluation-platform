from azure.ai.inference.models import UserMessage

from core.config import Settings
from llm.grok_client import GrokClient

settings = Settings.from_env()
client = GrokClient(settings)

response = client.client.complete(
    model=settings.azure_model_deployment,
    messages=[
        UserMessage(
            content="Yalnızca 'Bağlantı başarılı.' yaz."
        )
    ],
    temperature=0,
    max_tokens=80,
)

print(response.choices[0].message.content)
