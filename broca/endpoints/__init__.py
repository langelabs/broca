"""Main FastAPI router for all AI inference endpoint subpackages."""

from fastapi import APIRouter

from .audio import audio_router
from .desicion import decision_router
from .embedding import embedding_router
from .images import images_router
from .models import models_router
from .system import system_router
from .text import text_router
from .video import video_router
from .ui import ui_router

router = APIRouter()
router.include_router(system_router)

ai_router: APIRouter = APIRouter(prefix="/v1")

ai_router.include_router(audio_router)
ai_router.include_router(decision_router)
ai_router.include_router(embedding_router)
ai_router.include_router(images_router)
ai_router.include_router(models_router)
ai_router.include_router(text_router)
ai_router.include_router(video_router)

router.include_router(ai_router)
router.include_router(ui_router)
