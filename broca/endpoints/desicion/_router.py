"""FastAPI router for decision inference endpoints."""

from fastapi import APIRouter

decision_router: APIRouter = APIRouter(prefix="/decisions")
