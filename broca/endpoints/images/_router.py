"""FastAPI router for image inference endpoints."""

from fastapi import APIRouter

images_router: APIRouter = APIRouter(prefix="/images")
