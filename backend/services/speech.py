"""
Speech-to-Text Service — Groq Whisper API
Converts uploaded audio files into structured transcripts with speaker labels.
Uses Groq's hosted Whisper Large V3 for fast, accurate transcription.
"""
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeechService:
    """Handles audio → text conversion using Groq Whisper API."""
    
    def __init__(self):
        self._client = None
        self._loaded = False
        
    def _load_client(self):
        """Lazy-load the Groq client."""
        if self._loaded:
            return
            
        try:
            from groq import Groq
            from backend.config import GROQ_API_KEY
            
            if GROQ_API_KEY:
                self._client = Groq(api_key=GROQ_API_KEY)
                logger.info("Groq Whisper client initialized")
            else:
                logger.warning("No GROQ_API_KEY found, using mock transcription")
        except ImportError:
            logger.warning("groq package not installed, using mock transcription")
        except Exception as e:
            logger.error(f"Failed to init Groq client: {e}")
            
        self._loaded = True
                
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file to text using Groq Whisper API.
        
        Returns:
            {
                "full_text": str,
                "segments": [{"start": float, "end": float, "text": str}],
                "turns": [{"speaker": str, "text": str}]
            }
        """
        self._load_client()
        
        if not self._client:
            logger.warning("Groq client not available, using mock transcription")
            return self._mock_transcribe(audio_path)
            
        try:
            file_path = Path(audio_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info(f"Transcribing with Groq Whisper: {file_path.name}")
            
            with open(audio_path, "rb") as audio_file:
                transcription = self._client.audio.transcriptions.create(
                    file=(file_path.name, audio_file.read()),
                    model="whisper-large-v3",
                    language="en",
                    response_format="verbose_json",
                )
            
            # Parse response
            segments = []
            full_text_parts = []
            
            if hasattr(transcription, 'segments') and transcription.segments:
                for seg in transcription.segments:
                    segments.append({
                        "start": round(seg.get("start", seg.start) if isinstance(seg, dict) else seg.start, 2),
                        "end": round(seg.get("end", seg.end) if isinstance(seg, dict) else seg.end, 2),
                        "text": (seg.get("text", "") if isinstance(seg, dict) else seg.text).strip()
                    })
                    full_text_parts.append(segments[-1]["text"])
            else:
                # Fallback: use the full text
                full_text = transcription.text if hasattr(transcription, 'text') else str(transcription)
                full_text_parts.append(full_text)
                segments.append({"start": 0.0, "end": 0.0, "text": full_text})
            
            full_text = " ".join(full_text_parts)
            turns = self._detect_speakers(full_text, segments)
            
            # Get duration
            duration = 0.0
            if hasattr(transcription, 'duration') and transcription.duration:
                duration = float(transcription.duration)
            elif segments:
                duration = segments[-1]["end"]
            
            logger.info(f"Transcription complete: {len(segments)} segments, {duration:.1f}s")
            
            return {
                "full_text": full_text,
                "segments": segments,
                "turns": turns,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Groq transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")
    
    def _detect_speakers(self, full_text: str, segments: list) -> list:
        """
        Simple speaker diarization using heuristics.
        Alternates between Agent and Customer based on conversation patterns.
        """
        turns = []
        current_speaker = "Agent"
        
        for i, seg in enumerate(segments):
            text = seg["text"]
            
            # Heuristic: Agent usually speaks first and asks questions
            if i > 0:
                # Switch speaker on pause gaps > 1.5s
                if segments[i]["start"] - segments[i-1]["end"] > 1.5:
                    current_speaker = "Customer" if current_speaker == "Agent" else "Agent"
                # Switch on question marks (agent asks, customer responds)
                elif segments[i-1]["text"].endswith("?"):
                    current_speaker = "Customer" if current_speaker == "Agent" else "Agent"
            
            turns.append({
                "speaker": current_speaker,
                "text": text,
                "start": seg["start"],
                "end": seg["end"]
            })
        
        return turns
    
    def _mock_transcribe(self, audio_path: str) -> dict:
        """
        Provide a realistic mock transcript for demo/testing purposes.
        """
        mock_turns = [
            {"speaker": "Agent", "text": "Hello! Thank you for taking my call. This is Rahul from TechSolutions. How are you doing today?", "start": 0.0, "end": 4.5},
            {"speaker": "Customer", "text": "Hi Rahul, I'm doing fine. What's this about?", "start": 5.0, "end": 7.2},
            {"speaker": "Agent", "text": "Great! I wanted to tell you about our new CRM platform that's helping businesses increase their sales by 40%. Would you be interested in learning more?", "start": 7.8, "end": 14.0},
            {"speaker": "Customer", "text": "Well, we already have a CRM system in place. We're using Salesforce right now.", "start": 14.5, "end": 18.2},
            {"speaker": "Agent", "text": "That's great that you're already using a CRM. Many of our clients actually switched from Salesforce because our solution offers better AI-powered insights at half the cost. Our premium plan starts at just $49 per user per month.", "start": 18.8, "end": 28.0},
            {"speaker": "Customer", "text": "Hmm, $49 per user? That sounds expensive actually. We have about 50 team members. That would be quite a cost.", "start": 28.5, "end": 34.0},
            {"speaker": "Agent", "text": "I completely understand the concern. For a team of 50, we actually offer volume discounts. We could bring that down to $35 per user. Plus, our clients typically see ROI within the first 3 months.", "start": 34.5, "end": 43.0},
            {"speaker": "Customer", "text": "That's interesting, but I'm not sure I can make this decision on my own. I'd need to discuss with our CTO first.", "start": 43.5, "end": 49.0},
            {"speaker": "Agent", "text": "Absolutely, that makes total sense. Would it help if I set up a demo session where both you and your CTO can see the platform in action?", "start": 49.5, "end": 55.0},
            {"speaker": "Customer", "text": "You know what, that actually sounds good. Let me check his availability and get back to you.", "start": 55.5, "end": 60.0},
            {"speaker": "Agent", "text": "Perfect! I'll send you a calendar link with some available slots. Is your email the best way to reach you?", "start": 60.5, "end": 65.0},
            {"speaker": "Customer", "text": "Yes, please send it to my email. I'll try to get back to you by this week.", "start": 65.5, "end": 69.0},
            {"speaker": "Agent", "text": "Wonderful! Thank you so much for your time today. I'm confident you'll love what you see in the demo. Have a great day!", "start": 69.5, "end": 75.0},
            {"speaker": "Customer", "text": "Thanks, you too. Bye.", "start": 75.5, "end": 77.0},
        ]
        
        full_text = " ".join([t["text"] for t in mock_turns])
        segments = [{"start": t["start"], "end": t["end"], "text": t["text"]} for t in mock_turns]
        
        return {
            "full_text": full_text,
            "segments": segments,
            "turns": mock_turns,
            "duration": 77.0
        }


# Singleton
speech_service = SpeechService()
