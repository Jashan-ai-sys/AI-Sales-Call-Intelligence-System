# 🧠 AI Sales Call Intelligence System

> Automatically analyze sales calls → extract insights → improve conversions

A full-pipeline AI system that processes sales call recordings through **6 intelligent stages** to deliver actionable business intelligence.

## 🏗️ Architecture

```
Audio Input → Speech-to-Text → NLP Pipeline → Magic Moments → LLM Brain → RAG → Dashboard
```

| Stage | Technology | Purpose |
|-------|-----------|---------|
| 🎧 Speech-to-Text | Groq Whisper Large V3 | Audio → Transcript |
| 🧩 NLP Pipeline | `transformers` + `spaCy` | Intent, Sentiment, NER, Objections |
| ⚡ Magic Moments | Hybrid (Rules + Sentiment) | Detect turning points |
| 🤖 LLM Brain | Groq Llama 3.3 70B | Summary, Score, Coaching |
| 🔍 RAG | ChromaDB + Sentence Transformers | Historical pattern search |
| 📊 Dashboard | FastAPI + Vanilla JS | Real-time visualization |

## 🚀 Quick Start

### 1. Setup
```bash
# Clone & enter project
cd AI-Sales-Call-Intelligence-System

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Configure
```bash
# Copy environment template
copy .env.example .env

# Edit .env and add your Groq API key:
# GROQ_API_KEY=your_key_here
```

Get a free Groq API key: https://console.groq.com/keys

### 3. Run
```bash
# Start the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### 4. Demo
Click **"Run Demo Analysis"** on the dashboard to see the full pipeline in action with a built-in sample call.

## 📁 Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration & environment
│   ├── routes/
│   │   ├── upload.py           # Audio upload + full pipeline
│   │   ├── analysis.py         # Call history + stats
│   │   └── rag.py              # RAG semantic search
│   ├── services/
│   │   ├── speech.py           # Whisper transcription
│   │   ├── nlp.py              # Intent/Sentiment/NER/Objections
│   │   ├── magic_moments.py    # Turning-point detection
│   │   ├── llm.py              # Gemini-powered intelligence
│   │   └── rag.py              # ChromaDB vector store
│   └── models/
│       └── schemas.py          # Pydantic data models
├── frontend/
│   ├── index.html              # Dashboard UI
│   ├── styles.css              # Premium dark theme
│   └── app.js                  # Application logic
├── data/uploads/               # Uploaded audio files
├── chroma_db/                  # Persistent vector store
├── requirements.txt
└── .env.example
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload audio → full analysis |
| `POST` | `/api/demo` | Run demo analysis (no file needed) |
| `GET` | `/api/calls` | List analyzed calls |
| `GET` | `/api/calls/{id}` | Get full call analysis |
| `GET` | `/api/stats` | Dashboard aggregate stats |
| `POST` | `/api/rag/query` | Semantic search over calls |
| `GET` | `/api/health` | System health check |

## 💡 Key Features

- **Intent Detection**: Zero-shot classification of customer intent (price concern, interest, delay, etc.)
- **Objection Detection**: Identifies price, trust, urgency, competitor, and authority objections
- **Magic Moments**: Detects positive/negative turning points in conversations
- **Call Scoring**: 0-100 score with breakdown (sentiment, engagement, objection handling, closing)
- **Agent Coaching**: AI-generated specific suggestions for improvement
- **Conversion Probability**: Predicts likelihood of deal closure
- **RAG Search**: Query patterns across all historical calls

## 🎯 Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **ML/NLP**: Transformers, spaCy
- **STT**: Groq Whisper Large V3
- **LLM**: Groq Llama 3.3 70B Versatile
- **Vector DB**: ChromaDB
- **Frontend**: Vanilla HTML/CSS/JS (premium dark theme)
