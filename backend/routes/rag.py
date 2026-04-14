"""
RAG Routes — Semantic search over historical call data.
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from backend.services.rag import rag_service

router = APIRouter(prefix="/api/rag", tags=["RAG"])


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/query")
async def query_rag(request: RAGQueryRequest):
    """
    Semantic search over all stored call transcripts and insights.
    
    Example queries:
    - "What are the most common customer objections?"
    - "What strategies work best for closing deals?"
    - "Show me calls where pricing was a concern"
    - "How do top agents handle trust objections?"
    """
    result = rag_service.query(request.query, request.top_k)
    return result


@router.get("/stats")
async def rag_stats():
    """Get vector database statistics."""
    return rag_service.get_stats()
