import re
import logging
from huggingface_hub import InferenceClient
from backend.config import HF_API_KEY, SALES_INTENT_LABELS, OBJECTION_PATTERNS, ZERO_SHOT_MODEL

logger = logging.getLogger(__name__)


class NLPService:
    """Production-grade NLP Service with robust API handling + fallback."""

    def __init__(self):
        self._client = InferenceClient(api_key=HF_API_KEY) if HF_API_KEY else None
        self._spacy_nlp = None
        self._load_spacy()

    # ─── spaCy ────────────────────────────────────────────
    def _load_spacy(self):
        try:
            import spacy
            self._spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning("spaCy not found, using limited extraction")

    # ─── HF NORMALIZER (CORE) ─────────────────────────────
    def _normalize_hf_output(self, result):
        try:
            if not result:
                return None, None

            # SDK object with .labels / .scores (ZeroShotClassificationOutput)
            if hasattr(result, "labels") and hasattr(result, "scores"):
                return list(result.labels), list(result.scores)

            # list formats
            if isinstance(result, list):
                if not result:
                    return None, None

                first = result[0]

                # [Element(label, score)]
                if hasattr(first, "label") and hasattr(first, "score"):
                    return [r.label for r in result], [r.score for r in result]

                # [{"label": ..., "score": ...}]
                if isinstance(first, dict) and "label" in first:
                    return [r["label"] for r in result], [r["score"] for r in result]

                if len(result) == 1:
                    return self._normalize_hf_output(result[0])

            # dict formats
            if isinstance(result, dict):
                if "labels" in result:
                    return result["labels"], result["scores"]
                if "label" in result:
                    return [result["label"]], [result["score"]]

            return None, None

        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return None, None

    # ─── ZERO-SHOT CLASSIFICATION (DIRECT API) ──────────────
    def _zero_shot_classify(self, text: str, candidate_labels: list):
        """
        Direct API call for zero-shot classification, bypassing the broken
        huggingface_hub SDK parser (TypeError in v0.36.2).
        Uses a persistent session for connection pooling.
        Returns a dict with 'labels' and 'scores' keys, or None on failure.
        """
        if not hasattr(self, '_http_session'):
            import requests as _requests
            self._http_session = _requests.Session()
            self._http_session.headers.update({
                "Authorization": f"Bearer {HF_API_KEY}"
            })

        try:
            payload = {
                "inputs": text[:512],
                "parameters": {"candidate_labels": candidate_labels}
            }
            resp = self._http_session.post(
                f"https://router.huggingface.co/hf-inference/models/{ZERO_SHOT_MODEL}",
                json=payload,
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "labels" in data:
                    return data
            else:
                logger.warning(f"Zero-shot API returned {resp.status_code}: {resp.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Zero-shot API call failed: {e}")
            return None

    # ─── SENTIMENT ────────────────────────────────────────
    def analyze_sentiment(self, text: str) -> dict:
        if self._client:
            try:
                result = self._client.text_classification(
                    text[:512],
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
                )

                labels, scores = self._normalize_hf_output(result)

                if labels and scores:
                    idx = scores.index(max(scores))
                    return {
                        "label": labels[idx],
                        "score": round(scores[idx], 4)
                    }

            except Exception as e:
                logger.warning(f"Sentiment failed, fallback: {e}")

        return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> dict:
        text = text.lower()
        if any(w in text for w in ["good", "great", "interested", "yes"]):
            return {"label": "POSITIVE", "score": 0.8}
        if any(w in text for w in ["bad", "expensive", "not interested"]):
            return {"label": "NEGATIVE", "score": 0.8}
        return {"label": "NEUTRAL", "score": 0.5}

    # ─── INTENT ───────────────────────────────────────────
    def detect_intent(self, text: str) -> dict:
        if HF_API_KEY:
            try:
                result = self._zero_shot_classify(text, SALES_INTENT_LABELS)

                labels, scores = self._normalize_hf_output(result)

                if labels and scores:
                    idx = scores.index(max(scores))
                    return {
                        "text": text,
                        "intent": labels[idx],
                        "confidence": round(scores[idx], 4)
                    }

            except Exception as e:
                logger.warning(f"Intent failed, fallback: {e}")

        return self._rule_based_intent(text)

    def _rule_based_intent(self, text: str) -> dict:
        text_lower = text.lower()
        if "expensive" in text_lower or "cost" in text_lower or "price" in text_lower:
            return {"text": text, "intent": "price concern", "confidence": 0.7}
        if "interested" in text_lower:
            return {"text": text, "intent": "positive interest", "confidence": 0.7}
        if "later" in text_lower or "think about" in text_lower:
            return {"text": text, "intent": "delay or stalling", "confidence": 0.7}
        if "competitor" in text_lower or "already" in text_lower:
            return {"text": text, "intent": "competitor comparison", "confidence": 0.7}
        if "schedule" in text_lower or "demo" in text_lower or "calendar" in text_lower:
            return {"text": text, "intent": "scheduling request", "confidence": 0.7}
        if "?" in text:
            return {"text": text, "intent": "general question", "confidence": 0.5}
        return {"text": text, "intent": "general question", "confidence": 0.5}

    # ─── ENTITIES ─────────────────────────────────────────
    def extract_entities(self, text: str) -> list:
        entities = []

        if self._spacy_nlp:
            try:
                doc = self._spacy_nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ("MONEY", "ORG", "PERSON"):
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_
                        })
            except Exception:
                pass

        # price regex
        prices = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        for p in prices:
            if not any(e["text"] == p for e in entities):
                entities.append({"text": p, "label": "PRICE"})

        return entities

    # ─── OBJECTIONS ───────────────────────────────────────
    def detect_objections(self, text: str) -> list:
        text_lower = text.lower()
        matched = []

        for category, patterns in OBJECTION_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                matched.append(category)

        if not matched:
            return []

        confidence = 0.8

        if HF_API_KEY:
            try:
                labels = [f"{c} objection" for c in matched]

                result = self._zero_shot_classify(text, labels)
                _, scores = self._normalize_hf_output(result)

                if scores:
                    confidence = round(max(scores), 4)

            except Exception as e:
                logger.warning(f"Objection scoring failed: {e}")

        return [{"text": text, "category": c, "confidence": confidence} for c in matched]

    # ─── TURN ─────────────────────────────────────────────
    def analyze_turn(self, speaker: str, text: str) -> dict:
        sentiment = self.analyze_sentiment(text)

        intents = []
        objections = []

        if speaker.lower() == "customer":
            intents = [self.detect_intent(text)]
            objections = self.detect_objections(text)

        return {
            "speaker": speaker,
            "text": text,
            "sentiment": sentiment,
            "intents": intents,
            "entities": self.extract_entities(text),
            "objections": objections
        }

    # ─── FULL PIPELINE ────────────────────────────────────
    def analyze_full_transcript(self, turns: list) -> dict:
        results = []
        trajectory = []
        all_intents = []
        all_entities = []
        all_objections = []

        for i, turn in enumerate(turns):
            res = self.analyze_turn(turn["speaker"], turn["text"])
            results.append(res)

            # Aggregate cross-turn data
            all_intents.extend(res.get("intents", []))
            all_entities.extend(res.get("entities", []))
            all_objections.extend(res.get("objections", []))

            score = res["sentiment"].get("score", 0.5)
            if res["sentiment"]["label"] == "NEGATIVE":
                score = -score

            trajectory.append({
                "position": i / max(len(turns) - 1, 1),
                "score": score,
                "speaker": turn["speaker"]
            })

        avg = sum(t["score"] for t in trajectory) / len(trajectory) if trajectory else 0

        return {
            "turns": results,
            "overall_sentiment": {
                "label": "POSITIVE" if avg > 0 else "NEGATIVE" if avg < 0 else "NEUTRAL",
                "score": round(abs(avg), 4)
            },
            "sentiment_trajectory": trajectory,
            "all_intents": all_intents,
            "all_entities": all_entities,
            "all_objections": all_objections
        }


# Singleton
nlp_service = NLPService()