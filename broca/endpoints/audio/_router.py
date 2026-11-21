"""FastAPI router for audio inference endpoints."""

from fastapi import APIRouter

audio_router: APIRouter = APIRouter(prefix="/audio")
