# API Module
from fastapi import APIRouter
from .routes import recommendation, wardrobe, analysis, chat, scoring, context, pipeline

router = APIRouter()

# Include all route modules
router.include_router(pipeline.router, prefix="/pipeline", tags=["Full Pipeline"])
router.include_router(recommendation.router, prefix="/recommend", tags=["Recommendations"])
router.include_router(wardrobe.router, prefix="/wardrobe", tags=["Wardrobe"])
router.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(scoring.router, prefix="/scoring", tags=["Scoring"])
router.include_router(context.router, prefix="/context", tags=["Context"])
