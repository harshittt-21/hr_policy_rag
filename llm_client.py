from functools import lru_cache

import requests
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from pydantic import SecretStr
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import config

JINA_BATCH_SIZE = 64


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGroq:
    """
    Create or reuse the Groq chat model.

    Used for query transformation and generating final answers.

    Raises:
        ValueError: If GROQ_API_KEY is missing.
    """
    config.check_llm_config()

    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=SecretStr(config.GROQ_API_KEY),
    )


class JinaEmbeddings(Embeddings):
    """Generate text embeddings using the Jina Embeddings API."""

    def __init__(self):
        if not config.JINA_API_KEY:
            raise ValueError(
                "JINA_API_KEY is missing. Add it to your .env file."
            )

        self.api_key = config.JINA_API_KEY
        self.model = config.JINA_EMBEDDING_MODEL
        self.dimensions = config.EMBEDDING_DIM
        self.api_url = "https://api.jina.ai/v1/embeddings"

        # Reuse connections and retry transient failures (429 / 5xx).
        # POST is not retried by default, so it must be allowed explicitly.
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            respect_retry_after_header=True,
            raise_on_status=False,  # return the last response so we can read its body
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        """Send one batch of texts to Jina and return their embeddings."""
        if not texts:
            return []

        response = self.session.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
                "task": task,
                "dimensions": self.dimensions,
            },
            timeout=60,
        )

        if not response.ok:
            raise RuntimeError(
                f"Jina API error {response.status_code}: {response.text}"
            )

        try:
            result = response.json()["data"]
        except (ValueError, KeyError) as exc:
            raise RuntimeError(
                f"Unexpected Jina response format: {response.text[:500]}"
            ) from exc

        if len(result) != len(texts):
            raise RuntimeError(
                f"Jina returned {len(result)} embeddings for {len(texts)} inputs."
            )

        # Preserve the original input order.
        result.sort(key=lambda item: item["index"])

        return [item["embedding"] for item in result]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks for indexing, in batches."""
        if any(not t or not t.strip() for t in texts):
            raise ValueError(
                "embed_documents received an empty or blank text. "
                "Filter these out in the chunker."
            )

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), JINA_BATCH_SIZE):
            batch = texts[i : i + JINA_BATCH_SIZE]
            embeddings.extend(self._embed(batch, "retrieval.passage"))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a user's search query."""
        if not text or not text.strip():
            raise ValueError("embed_query received an empty query.")
        return self._embed([text], "retrieval.query")[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> JinaEmbeddings:
    """
    Create or reuse the Jina embedding model.

    Raises:
        ValueError: If JINA_API_KEY is missing.
    """
    config.check_embedding_config()
    return JinaEmbeddings()