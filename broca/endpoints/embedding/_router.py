"""FastAPI router for embedding inference endpoints."""

from fastapi import APIRouter

embedding_router: APIRouter = APIRouter(prefix="/embeddings")
