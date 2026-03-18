"""
OpenAI-compatible embedding encoder.

Supports the official OpenAI endpoint and OpenAI-compatible gateways that
implement the `/embeddings` API, configured via `api_key` and `base_url`.
"""

from typing import List, Union, Optional
import os
import numpy as np


class OpenAIVectorEncoder:
    """Encode text with OpenAI-compatible embedding APIs."""

    _MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        from openai import OpenAI

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or pass api_key explicitly.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.base_url = base_url
        self.dimension = self._MODEL_DIMENSIONS.get(model_name, 1536)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def encode_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append(self.encode(batch))

        if not batches:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack(batches).astype(np.float32)
