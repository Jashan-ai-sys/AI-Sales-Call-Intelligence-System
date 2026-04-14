"""
AI Sales Call Intelligence System — FastAPI Application
Full pipeline: Audio → Speech-to-Text → NLP → Magic Moments → LLM → RAG → Dashboard
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.routes import upload, analysis, rag

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── Frontend Path ────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ─── Lifespan ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("🧠 AI Sales Call Intelligence System")
    logger.info("=" * 60)
    logger.info("Pipeline: Audio → STT → NLP → Magic Moments → LLM → RAG")
    logger.info(f"Frontend: {'✅ Serving from ' + str(FRONTEND_DIR) if FRONTEND_DIR.exists() else '❌ Not found'}")
    logger.info("=" * 60)
    yield

# ─── FastAPI App ──────────────────────────────────────────
app = FastAPI(
    title="🧠 AI Sales Call Intelligence",
    description="Automatically analyze sales calls → extract insights → improve conversions",
    version="1.0.0",
    lifespan=lifespan
)

# ─── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(rag.router)

# ─── Static Frontend ─────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ─── Health Check ─────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    from backend.services.rag import rag_service
    return {
        "status": "healthy",
        "service": "AI Sales Call Intelligence System",
        "version": "1.0.0",
        "rag_stats": rag_service.get_stats()
    }


