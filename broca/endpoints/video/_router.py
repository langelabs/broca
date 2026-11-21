"""FastAPI router for video inference endpoints."""

from fastapi import APIRouter

video_router: APIRouter = APIRouter(prefix="/videos")
