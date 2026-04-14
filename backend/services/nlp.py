"""
NLP Pipeline Service — Core Analysis Engine
Handles: Intent Detection, Sentiment Analysis, Entity Extraction, Objection Detection
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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


class NLPService:
    """Core NLP engine for sales call analysis."""
    
    def __init__(self):
        self._sentiment_pipeline = None
        self._zero_shot_pipeline = None
        self._spacy_nlp = None
        self._loaded = False
        
    def _load_models(self):
        """Lazy-load all NLP models."""
        if self._loaded:
            return
            
        try:
            from transformers import pipeline
            
            logger.info("Loading sentiment analysis model...")
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512
            )
            
            logger.info("Loading zero-shot classification model...")
            self._zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
        except ImportError:
            logger.warning("transformers not installed, using rule-based fallback")
            
        try:
            import spacy
            logger.info("Loading spaCy model...")
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        except ImportError:
            logger.warning("spaCy not installed")
            
        self._loaded = True
    
    # ─── Sentiment Analysis ───────────────────────────────
    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze emotional tone of text.
        Returns: {"label": "POSITIVE/NEGATIVE/NEUTRAL", "score": 0.0-1.0}
        """
        self._load_models()
        
        if self._sentiment_pipeline:
            try:
                result = self._sentiment_pipeline(text[:512])[0]
                return {
                    "label": result["label"],
                    "score": round(result["score"], 4)
                }
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
        
        # Fallback: rule-based sentiment
        return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> dict:
        """Simple keyword-based sentiment fallback."""
        text_lower = text.lower()
        
        positive_words = ["good", "great", "excellent", "love", "perfect", "wonderful", 
                         "amazing", "interested", "sounds good", "proceed", "yes", "sure",
                         "thank", "happy", "glad", "agree"]
        negative_words = ["expensive", "bad", "terrible", "hate", "no", "not interested",
                         "too much", "can't", "won't", "problem", "issue", "complaint",
                         "disappointed", "frustrated", "angry"]
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            return {"label": "POSITIVE", "score": round(min(0.5 + pos_count * 0.1, 0.99), 4)}
        elif neg_count > pos_count:
            return {"label": "NEGATIVE", "score": round(min(0.5 + neg_count * 0.1, 0.99), 4)}
        else:
            return {"label": "NEUTRAL", "score": 0.5}
    
    # ─── Intent Detection ─────────────────────────────────
    def detect_intent(self, text: str) -> dict:
        """
        Classify customer intent using zero-shot classification.
        Returns: {"text": str, "intent": str, "confidence": float}
        """
        self._load_models()
        
        if self._zero_shot_pipeline:
            try:
                result = self._zero_shot_pipeline(
                    text[:512],
                    candidate_labels=SALES_INTENT_LABELS,
                    multi_label=False
                )
                return {
                    "text": text,
                    "intent": result["labels"][0],
                    "confidence": round(result["scores"][0], 4)
                }
            except Exception as e:
                logger.error(f"Intent detection failed: {e}")
        
        # Fallback
        return self._rule_based_intent(text)
    
    def _rule_based_intent(self, text: str) -> dict:
        """Rule-based intent detection fallback."""
        text_lower = text.lower()
        
        intent_keywords = {
            "price concern": ["expensive", "cost", "price", "budget", "afford", "too much"],
            "positive interest": ["sounds good", "interested", "tell me more", "proceed", "let's do it"],
            "delay or stalling": ["think about it", "later", "call me back", "not now", "get back"],
            "competitor comparison": ["already using", "salesforce", "current system", "alternative"],
            "authority escalation": ["boss", "manager", "cto", "discuss with", "decision"],
            "scheduling request": ["schedule", "demo", "meeting", "calendar", "availability"],
            "rejection": ["not interested", "no thanks", "don't need", "pass"],
            "product inquiry": ["features", "how does it", "what can it", "tell me about"],
            "closing agreement": ["let's proceed", "sign up", "okay", "deal", "agreed"],
        }
        
        for intent, keywords in intent_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return {"text": text, "intent": intent, "confidence": 0.75}
        
        return {"text": text, "intent": "general question", "confidence": 0.5}
    
    # ─── Entity Extraction ────────────────────────────────
    def extract_entities(self, text: str) -> list:
        """
        Extract named entities relevant to sales: prices, products, companies, people.
        """
        self._load_models()
        entities = []
        
        # spaCy NER
        if self._spacy_nlp:
            try:
                doc = self._spacy_nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ("MONEY", "ORG", "PERSON", "PRODUCT", "CARDINAL", "PERCENT"):
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char
                        })
            except Exception as e:
                logger.error(f"spaCy NER failed: {e}")
        
        # Custom pattern matching for prices
        price_patterns = re.findall(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:per|/)\s*\w+)?', text)
        for match in price_patterns:
            if not any(e["text"] == match for e in entities):
                start = text.find(match)
                entities.append({
                    "text": match,
                    "label": "PRICE",
                    "start": start,
                    "end": start + len(match)
                })
        
        # Competitor detection
        competitors = ["Salesforce", "HubSpot", "Zoho", "Pipedrive", "Monday", 
                       "Freshsales", "Copper", "Insightly"]
        for comp in competitors:
            if comp.lower() in text.lower():
                start = text.lower().find(comp.lower())
                entities.append({
                    "text": comp,
                    "label": "COMPETITOR",
                    "start": start,
                    "end": start + len(comp)
                })
        
        return entities
    
    # ─── Objection Detection ──────────────────────────────
    def detect_objections(self, text: str) -> list:
        """
        Detect customer objections and classify by category.
        This is GOLD for businesses.
        """
        self._load_models()
        objections = []
        text_lower = text.lower()
        
        for category, patterns in OBJECTION_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Get confidence from zero-shot if available
                    confidence = 0.8
                    if self._zero_shot_pipeline:
                        try:
                            result = self._zero_shot_pipeline(
                                text[:512],
                                candidate_labels=[
                                    f"{category} objection",
                                    "no objection",
                                    "general statement"
                                ]
                            )
                            confidence = round(result["scores"][0], 4)
                        except Exception:
                            pass
                    
                    objections.append({
                        "text": text,
                        "category": category,
                        "confidence": confidence,
                        "matched_pattern": match.group()
                    })
                    break  # One objection per category per text
        
        return objections
    
    # ─── Full Turn Analysis ───────────────────────────────
    def analyze_turn(self, speaker: str, text: str) -> dict:
        """Run full NLP pipeline on a single turn."""
        sentiment = self.analyze_sentiment(text)
        
        # Only detect intents/objections for customer turns
        intents = []
        objections = []
        if speaker.lower() == "customer":
            intents = [self.detect_intent(text)]
            objections = self.detect_objections(text)
        
        entities = self.extract_entities(text)
        
        return {
            "speaker": speaker,
            "text": text,
            "sentiment": sentiment,
            "intents": intents,
            "entities": entities,
            "objections": objections
        }
    
    def analyze_full_transcript(self, turns: list) -> dict:
        """
        Run NLP pipeline on all turns and produce aggregated results.
        """
        analyzed_turns = []
        all_intents = []
        all_entities = []
        all_objections = []
        sentiment_trajectory = []
        
        for i, turn in enumerate(turns):
            analysis = self.analyze_turn(turn["speaker"], turn["text"])
            analyzed_turns.append(analysis)
            
            all_intents.extend(analysis["intents"])
            all_entities.extend(analysis["entities"])
            all_objections.extend(analysis["objections"])
            
            # Track sentiment trajectory
            position = i / max(len(turns) - 1, 1)
            score = analysis["sentiment"]["score"]
            if analysis["sentiment"]["label"] == "NEGATIVE":
                score = -score
            sentiment_trajectory.append({
                "position": round(position, 3),
                "score": round(score, 4),
                "speaker": turn["speaker"]
            })
        
        # Overall sentiment: average
        scores = [s["score"] for s in sentiment_trajectory]
        avg_score = sum(scores) / len(scores) if scores else 0
        overall_label = "POSITIVE" if avg_score > 0.1 else "NEGATIVE" if avg_score < -0.1 else "NEUTRAL"
        
        return {
            "turns": analyzed_turns,
            "overall_sentiment": {"label": overall_label, "score": round(abs(avg_score), 4)},
            "sentiment_trajectory": sentiment_trajectory,
            "all_intents": all_intents,
            "all_entities": all_entities,
            "all_objections": all_objections
        }


# Singleton
nlp_service = NLPService()
