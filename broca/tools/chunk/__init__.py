from .types import Document, Chunk
from io import BytesIO

from .pipeline import sem_chunk_document, homogenise, annotate


async def chunk(content: str | BytesIO, name: str = "document") -> Document:
    """Convert a document and return its annotated chunks.

    :param content: HTTPS source URL or in-memory document stream.
    :param name: Document name and filename for in-memory format detection.
    :returns: The named document with its summaries and embeddings.
    """
    md = await homogenise(content, name=name)
    chunks = await sem_chunk_document(md)
    chunks_annotated = await annotate(chunks)
    return Document(name=name, chunks=chunks_annotated)


__all__ = ["Chunk", "Document", "chunk"]
