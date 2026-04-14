"""
Pydantic schemas for request/response models
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─── NLP Results ───────────────────────────────────────────
class SentimentResult(BaseModel):
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float
    
class IntentResult(BaseModel):
    text: str
    intent: str
    confidence: float

class EntityResult(BaseModel):
    text: str
    label: str  # PRICE, PRODUCT, COMPETITOR, ORG, PERSON
    start: int
    end: int

class ObjectionResult(BaseModel):
    text: str
    category: str  # price, trust, urgency, competitor, authority
    confidence: float

class MagicMoment(BaseModel):
    text: str
    moment_type: str  # positive_turning_point, negative_turning_point
    sentiment_score: float
    position_in_call: float  # 0.0 to 1.0

# ─── Turn-level Analysis ──────────────────────────────────
class TurnAnalysis(BaseModel):
    speaker: str  # Agent / Customer
    text: str
    sentiment: SentimentResult
    intents: list[IntentResult] = []
    entities: list[EntityResult] = []
    objections: list[ObjectionResult] = []

# ─── Full Call Analysis ────────────────────────────────────
class CallAnalysis(BaseModel):
    id: str
    filename: str
    timestamp: str
    transcript: str
    turns: list[TurnAnalysis]
    
    # Aggregated NLP
    overall_sentiment: SentimentResult
    sentiment_trajectory: list[dict]  # [{position, score}]
    all_intents: list[IntentResult]
    all_entities: list[EntityResult]
    all_objections: list[ObjectionResult]
    magic_moments: list[MagicMoment]
    
    # LLM outputs
    summary: str
    call_score: int  # 0-100
    score_breakdown: dict  # {sentiment, engagement, objection_handling, closing}
    agent_suggestions: list[str]
    conversion_probability: float  # 0.0 to 1.0

# ─── RAG ───────────────────────────────────────────────────
class RAGQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]  # [{call_id, snippet, score}]

# ─── Dashboard Stats ──────────────────────────────────────
class DashboardStats(BaseModel):
    total_calls: int
    avg_score: float
    avg_sentiment: float
    top_objections: list[dict]  # [{category, count}]
    conversion_rate: float
    recent_calls: list[dict]
