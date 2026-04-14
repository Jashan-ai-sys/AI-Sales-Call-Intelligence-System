"""
AI Sales Call Intelligence System — Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Create directories
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Model configs
WHISPER_MODEL = "openai/whisper-large-v3"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# ─── Intent Labels for Sales Calls ────────────────────────
SALES_INTENT_LABELS = [
    "price concern",
    "product inquiry",
    "competitor comparison", 
    "positive interest",
    "scheduling request",
    "delay or stalling",
    "authority escalation",
    "feature request",
    "complaint",
    "closing agreement",
    "rejection",
    "trust concern",
    "general question"
]

# ─── Objection Categories ─────────────────────────────────
OBJECTION_PATTERNS = {
    "price": [
        r"expensive", r"cost(s|ly)?", r"budget", r"afford",
        r"too much", r"cheaper", r"price", r"pricing",
        r"discount", r"pay\s+that", r"worth\s+it"
    ],
    "trust": [
        r"not sure", r"don't (know|trust)", r"guarantee",
        r"reviews?", r"proof", r"case stud(y|ies)",
        r"how do I know", r"sounds too good"
    ],
    "urgency": [
        r"think about it", r"not (right )?now", r"later",
        r"no rush", r"next (month|quarter|year)",
        r"call me (back|later)", r"get back to you"
    ],
    "competitor": [
        r"already (have|use|using)", r"current (system|solution|provider)",
        r"salesforce", r"hubspot", r"zoho", r"competitor",
        r"alternative", r"other (option|solution)s?"
    ],
    "authority": [
        r"(my |the )?boss", r"manager", r"CTO", r"CEO",
        r"decision maker", r"discuss with", r"check with",
        r"team", r"approval", r"can't decide (alone|on my own)"
    ]
}
