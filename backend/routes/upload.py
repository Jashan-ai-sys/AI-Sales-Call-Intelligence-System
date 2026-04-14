"""
Upload Route — Handles audio file upload and triggers the full analysis pipeline.
"""
import uuid
import logging
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.config import UPLOADS_DIR
from backend.services.speech import speech_service
from backend.services.nlp import nlp_service
from backend.services.magic_moments import magic_moments_service
from backend.services.llm import llm_service
from backend.services.rag import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Upload"])

# In-memory store for analyzed calls (in production → database)
analyzed_calls: dict = {}


@router.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Upload a sales call audio file and run the full analysis pipeline.
    
    Pipeline:
    1. Save audio file
    2. Transcribe with Whisper
    3. Run NLP pipeline (sentiment, intent, NER, objections)
    4. Detect magic moments
    5. Generate LLM intelligence (summary, score, suggestions)
    6. Store in RAG vector DB
    7. Return full analysis
    """
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique ID
    call_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    # Save file
    file_path = UPLOADS_DIR / f"{call_id}{file_ext}"
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved audio file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    try:
        # ─── Step 1: Transcribe ───────────────────────────
        logger.info(f"[{call_id}] Starting transcription...")
        transcription = speech_service.transcribe(str(file_path))
        
        # ─── Step 2: NLP Pipeline ─────────────────────────
        logger.info(f"[{call_id}] Running NLP pipeline...")
        nlp_results = nlp_service.analyze_full_transcript(transcription["turns"])
        
        # ─── Step 3: Magic Moments ────────────────────────
        logger.info(f"[{call_id}] Detecting magic moments...")
        magic_moments = magic_moments_service.detect_magic_moments(
            transcription["turns"],
            nlp_results["sentiment_trajectory"]
        )
        
        # ─── Step 4: LLM Intelligence ────────────────────
        logger.info(f"[{call_id}] Generating LLM intelligence...")
        llm_output = await llm_service.generate_call_intelligence(
            transcription["full_text"],
            nlp_results,
            magic_moments
        )
        
        # ─── Step 5: Store in RAG ─────────────────────────
        logger.info(f"[{call_id}] Storing in vector DB...")
        rag_service.store_call(call_id, transcription["full_text"], {
            "filename": file.filename,
            "timestamp": timestamp,
            "score": llm_output["call_score"]
        })
        rag_service.store_call_insights(
            call_id,
            llm_output["summary"],
            nlp_results["all_objections"],
            llm_output["call_score"],
            llm_output["conversion_probability"]
        )
        
        # ─── Build Final Result ───────────────────────────
        result = {
            "id": call_id,
            "filename": file.filename,
            "timestamp": timestamp,
            "duration": transcription.get("duration", 0),
            "transcript": transcription["full_text"],
            "turns": nlp_results["turns"],
            "overall_sentiment": nlp_results["overall_sentiment"],
            "sentiment_trajectory": nlp_results["sentiment_trajectory"],
            "all_intents": nlp_results["all_intents"],
            "all_entities": nlp_results["all_entities"],
            "all_objections": nlp_results["all_objections"],
            "magic_moments": magic_moments,
            "summary": llm_output["summary"],
            "call_score": llm_output["call_score"],
            "score_breakdown": llm_output["score_breakdown"],
            "agent_suggestions": llm_output["agent_suggestions"],
            "conversion_probability": llm_output["conversion_probability"]
        }
        
        # Store in memory
        analyzed_calls[call_id] = result
        
        logger.info(f"[{call_id}] ✅ Analysis complete! Score: {llm_output['call_score']}/100")
        return result
        
    except Exception as e:
        logger.error(f"[{call_id}] Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {str(e)}")


@router.post("/demo")
async def run_demo_analysis():
    """
    Run analysis on the built-in demo transcript (no audio file needed).
    Perfect for testing the pipeline without uploading audio.
    """
    call_id = f"demo-{str(uuid.uuid4())[:4]}"
    timestamp = datetime.now().isoformat()
    
    # Get mock transcript
    transcription = speech_service._mock_transcribe("demo")
    
    # Run full pipeline
    nlp_results = nlp_service.analyze_full_transcript(transcription["turns"])
    
    magic_moments = magic_moments_service.detect_magic_moments(
        transcription["turns"],
        nlp_results["sentiment_trajectory"]
    )
    
    llm_output = await llm_service.generate_call_intelligence(
        transcription["full_text"],
        nlp_results,
        magic_moments
    )
    
    rag_service.store_call(call_id, transcription["full_text"], {
        "filename": "demo_call.wav",
        "timestamp": timestamp,
        "score": llm_output["call_score"]
    })
    rag_service.store_call_insights(
        call_id,
        llm_output["summary"],
        nlp_results["all_objections"],
        llm_output["call_score"],
        llm_output["conversion_probability"]
    )
    
    result = {
        "id": call_id,
        "filename": "demo_call.wav",
        "timestamp": timestamp,
        "duration": transcription.get("duration", 0),
        "transcript": transcription["full_text"],
        "turns": nlp_results["turns"],
        "overall_sentiment": nlp_results["overall_sentiment"],
        "sentiment_trajectory": nlp_results["sentiment_trajectory"],
        "all_intents": nlp_results["all_intents"],
        "all_entities": nlp_results["all_entities"],
        "all_objections": nlp_results["all_objections"],
        "magic_moments": magic_moments,
        "summary": llm_output["summary"],
        "call_score": llm_output["call_score"],
        "score_breakdown": llm_output["score_breakdown"],
        "agent_suggestions": llm_output["agent_suggestions"],
        "conversion_probability": llm_output["conversion_probability"]
    }
    
    analyzed_calls[call_id] = result
    return result
