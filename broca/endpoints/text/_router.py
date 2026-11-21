"""FastAPI router for text inference endpoints."""

from fastapi import APIRouter

text_router: APIRouter = APIRouter(prefix="/chat")
