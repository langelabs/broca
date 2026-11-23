import asyncio
from collections.abc import Sequence

import numpy as np
from chonkie import BaseEmbeddings
from chonkie.chunker import RecursiveChunker
from chonkie.tokenizer import Tokenizer

from broca.inference.embedding import infer as infer_embeddings, pipeline

from ..types import Chunk


class EmbeddingMapper(BaseEmbeddings, Tokenizer):

    def encode(self, text: str) -> Sequence[int]:
        return pipeline.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return pipeline.decode(tokens)

    def tokenize(self, text: str) -> Sequence[str | int]:
        return pipeline.encode(text)

    def get_tokenizer(self) -> Tokenizer:
        """Return a tokenizer that uses Broca's loaded embedding model.

        :returns: A Chonkie tokenizer that resolves the runtime on each operation.
        """
        return self

    @property
    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> np.ndarray:
        loop = asyncio.get_event_loop()
        embedding = loop.run_until_complete(infer_embeddings([text]))
        return embedding.vectors[0]


chunker = RecursiveChunker.from_recipe(
    "markdown",
    lang="en",
    chunk_size=512,
    tokenizer=EmbeddingMapper(),
)


async def sem_chunk_document(
        content: str
) -> list[Chunk]:

    # chunk the content
    native_chunks = await chunker.achunk(content)
    return [Chunk(index=index, content=chunk.text) for index, chunk in enumerate(native_chunks)]
