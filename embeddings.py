import json
import os
import urllib.error
import urllib.request
from typing import List, Optional


class EmbeddingError(RuntimeError):
    pass


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EmbeddingError("OPENAI_API_KEY is not set.")

        self.model = model or os.environ.get("OPENAI_EMBEDDINGS_MODEL") or "text-embedding-3-small"
        env_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.base_url = base_url or env_base_url or "https://api.openai.com/v1/embeddings"
        self.timeout = timeout

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": texts,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(
                f"Embedding request failed: {exc.code} {exc.reason} - {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"Embedding request failed: {exc.reason}") from exc

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EmbeddingError("Embedding response was not valid JSON.") from exc

        data = payload.get("data")
        if not isinstance(data, list):
            raise EmbeddingError(f"Embedding response missing data: {payload}")

        embeddings: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise EmbeddingError("Embedding response missing embeddings.")
            embeddings.append(embedding)

        if len(embeddings) != len(texts):
            raise EmbeddingError("Embedding response size mismatch.")

        return embeddings
