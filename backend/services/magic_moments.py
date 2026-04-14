"""
Magic Moments Detection Service
Detects key turning points in sales conversations — both positive and negative.
Uses a hybrid approach: rule-based keyword matching + sentiment shift detection.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Positive Moment Indicators ───────────────────────────
POSITIVE_INDICATORS = [
    r"sounds? good",
    r"(that'?s|this is) (great|interesting|perfect|amazing|wonderful)",
    r"let'?s (do it|proceed|go ahead|move forward|schedule)",
    r"i'?m interested",
    r"tell me more",
    r"sign (me )?up",
    r"(yes|yeah|sure|okay|absolutely),?\s*(let'?s|i'?d like|please)",
    r"how (do|can) (i|we) (get started|sign up|begin)",
    r"(send|give) me (the|a|more) (details|info|link|calendar)",
    r"(that|it) makes? sense",
    r"i (like|love) (that|this|it)",
    r"(great|good) (deal|price|offer)",
    r"we (need|could use) (this|that|something like)",
]

# ─── Negative Moment Indicators ───────────────────────────
NEGATIVE_INDICATORS = [
    r"(i'?ll|let me) think about it",
    r"not (really )?interested",
    r"(too|very|quite) expensive",
    r"(can'?t|don'?t) afford",
    r"no(t| )thank(s| you)",
    r"(maybe|perhaps) (later|next|another)",
    r"i don'?t (think|believe|see)",
    r"(that'?s|this is) (too much|a lot)",
    r"we'?re (happy|satisfied) with (what we have|our current)",
    r"(don'?t|won'?t) (need|want|require)",
    r"(stop|quit) (calling|emailing|contacting)",
    r"not (the right|a good) time",
    r"(i|we) (already|currently) (have|use)",
    r"(sounds )?too good to be true",
]


class MagicMomentsService:
    """
    Detects pivotal moments in sales calls.
    
    These are the turning points where a prospect shifts 
    toward or away from a conversion.
    """
    
    def detect_magic_moments(self, turns: list, sentiment_trajectory: list) -> list:
        """
        Detect magic moments using:
        1. Keyword pattern matching
        2. Sentiment shift detection (big jumps between turns)
        3. Position weighting (moments near the end are more impactful)
        
        Returns list of MagicMoment dicts.
        """
        moments = []
        
        for i, turn in enumerate(turns):
            if turn["speaker"].lower() != "customer":
                continue
                
            text = turn["text"]
            position = i / max(len(turns) - 1, 1)
            
            # ─── 1. Pattern-Based Detection ───────────────
            positive_match = self._match_patterns(text, POSITIVE_INDICATORS)
            negative_match = self._match_patterns(text, NEGATIVE_INDICATORS)
            
            if positive_match:
                sentiment_score = self._get_sentiment_score(i, sentiment_trajectory)
                moments.append({
                    "text": text,
                    "moment_type": "positive_turning_point",
                    "sentiment_score": round(sentiment_score, 4),
                    "position_in_call": round(position, 3),
                    "trigger": positive_match,
                    "impact": self._calculate_impact(position, sentiment_score, "positive")
                })
                
            if negative_match:
                sentiment_score = self._get_sentiment_score(i, sentiment_trajectory)
                moments.append({
                    "text": text,
                    "moment_type": "negative_turning_point",
                    "sentiment_score": round(sentiment_score, 4),
                    "position_in_call": round(position, 3),
                    "trigger": negative_match,
                    "impact": self._calculate_impact(position, sentiment_score, "negative")
                })
            
            # ─── 2. Sentiment Shift Detection ─────────────
            if i > 0 and i < len(sentiment_trajectory):
                shift = self._detect_sentiment_shift(i, sentiment_trajectory)
                if shift and not (positive_match or negative_match):
                    moments.append(shift | {
                        "text": text,
                        "position_in_call": round(position, 3)
                    })
        
        # Sort by impact (most impactful first)
        moments.sort(key=lambda m: m.get("impact", 0), reverse=True)
        
        return moments
    
    def _match_patterns(self, text: str, patterns: list) -> str | None:
        """Check if text matches any pattern, return the matched text."""
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group()
        return None
    
    def _get_sentiment_score(self, turn_index: int, trajectory: list) -> float:
        """Get sentiment score for a turn from the trajectory."""
        if turn_index < len(trajectory):
            return trajectory[turn_index]["score"]
        return 0.0
    
    def _detect_sentiment_shift(self, turn_index: int, trajectory: list) -> dict | None:
        """
        Detect large sentiment shifts between consecutive customer turns.
        Threshold: |shift| > 0.4
        """
        if turn_index >= len(trajectory) or turn_index < 1:
            return None
            
        current = trajectory[turn_index]["score"]
        
        # Find previous customer turn
        prev_score = None
        for j in range(turn_index - 1, -1, -1):
            if trajectory[j].get("speaker", "").lower() == "customer":
                prev_score = trajectory[j]["score"]
                break
        
        if prev_score is None:
            return None
            
        shift = current - prev_score
        
        if abs(shift) > 0.4:
            moment_type = "positive_turning_point" if shift > 0 else "negative_turning_point"
            return {
                "moment_type": moment_type,
                "sentiment_score": round(current, 4),
                "trigger": f"sentiment_shift ({round(shift, 2):+.2f})",
                "impact": round(abs(shift) * 0.8, 3)
            }
        
        return None
    
    def _calculate_impact(self, position: float, sentiment_score: float, moment_type: str) -> float:
        """
        Calculate impact score (0-1) considering:
        - Position in call (later moments are typically more decisive)
        - Strength of sentiment
        """
        position_weight = 0.5 + (position * 0.5)  # 0.5 to 1.0
        sentiment_weight = abs(sentiment_score)
        
        impact = (position_weight * 0.6 + sentiment_weight * 0.4)
        return round(min(impact, 1.0), 3)


# Singleton
magic_moments_service = MagicMomentsService()
