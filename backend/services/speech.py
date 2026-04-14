"""
Speech-to-Text Service — Whisper-based transcription
Converts uploaded audio files into structured transcripts with speaker labels.
"""
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeechService:
    """Handles audio → text conversion using faster-whisper."""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        
    def _load_model(self):
        """Lazy-load the Whisper model to avoid memory usage when not needed."""
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading Whisper model: {self.model_size}")
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",  # Use "cuda" if GPU available
                    compute_type="int8"
                )
                logger.info("Whisper model loaded successfully")
            except ImportError:
                logger.warning("faster-whisper not installed, using mock transcription")
                self.model = "mock"
                
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file to text with timestamps.
        
        Returns:
            {
                "full_text": str,
                "segments": [{"start": float, "end": float, "text": str}],
                "turns": [{"speaker": str, "text": str}]
            }
        """
        self._load_model()
        
        if self.model == "mock":
            return self._mock_transcribe(audio_path)
            
        try:
            segments_gen, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                )
            )
            
            segments = []
            full_text_parts = []
            
            for segment in segments_gen:
                segments.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                })
                full_text_parts.append(segment.text.strip())
            
            full_text = " ".join(full_text_parts)
            turns = self._detect_speakers(full_text, segments)
            
            return {
                "full_text": full_text,
                "segments": segments,
                "turns": turns,
                "duration": info.duration if hasattr(info, 'duration') else segments[-1]["end"] if segments else 0
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
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
