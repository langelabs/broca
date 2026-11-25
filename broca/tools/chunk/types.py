from pydantic import BaseModel
from typing import Optional


class Chunk(BaseModel):
    index: int
    summary: Optional[str] = None
    embedding: Optional[list[float]] = None
    content: str


class Document(BaseModel):
    name: str
    chunks: list[Chunk]
