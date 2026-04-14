"""
LLM Layer Service — The Brain (Groq-powered)
Uses Groq API with Llama 3.3 70B to generate:
  A. Call summaries
  B. Call scores (0-100)
  C. Agent coaching suggestions
  D. Conversion probability
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMService:
    """LLM-powered intelligence layer for call analysis using Groq."""
    
    def __init__(self):
        self._client = None
        self._model_name = "openai/gpt-oss-120b"
        self._loaded = False
    
    def _load_model(self):
        """Lazy-load Groq client."""
        if self._loaded:
            return
            
        try:
            from groq import Groq
            from backend.config import GROQ_API_KEY
            
            if GROQ_API_KEY:
                self._client = Groq(api_key=GROQ_API_KEY)
                logger.info(f"Groq client initialized (model: {self._model_name})")
            else:
                logger.warning("No GROQ_API_KEY found, using mock LLM")
        except ImportError:
            logger.warning("groq package not installed, using mock LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
            
        self._loaded = True
    
    async def generate_call_intelligence(self, transcript: str, nlp_results: dict, magic_moments: list) -> dict:
        """
        Generate comprehensive call intelligence using LLM.
        
        Returns:
            {
                "summary": str,
                "call_score": int (0-100),
                "score_breakdown": {sentiment, engagement, objection_handling, closing},
                "agent_suggestions": [str],
                "conversion_probability": float (0-1)
            }
        """
        self._load_model()
        
        # Build context for the LLM
        context = self._build_context(transcript, nlp_results, magic_moments)
        
        if self._client:
            return await self._groq_generate(context)
        else:
            return self._mock_generate(nlp_results, magic_moments)
    
    def _build_context(self, transcript: str, nlp_results: dict, magic_moments: list) -> str:
        """Build a structured prompt context for the LLM."""
        
        # Extract key NLP findings
        objections = nlp_results.get("all_objections", [])
        objection_summary = ", ".join(set(o["category"] for o in objections)) if objections else "None detected"
        
        intents = nlp_results.get("all_intents", [])
        intent_summary = ", ".join(set(i["intent"] for i in intents)) if intents else "None detected"
        
        entities = nlp_results.get("all_entities", [])
        entity_summary = ", ".join(set(f'{e["label"]}: {e["text"]}' for e in entities)) if entities else "None detected"
        
        overall_sentiment = nlp_results.get("overall_sentiment", {})
        
        moments_text = ""
        for m in magic_moments[:5]:
            moments_text += f"\n  - [{m['moment_type']}] \"{m['text'][:100]}\" (impact: {m.get('impact', 'N/A')})"
        
        return f"""You are an expert sales call analyst. Analyze this sales call and provide actionable intelligence.

=== TRANSCRIPT ===
{transcript[:3000]}

=== NLP ANALYSIS ===
Overall Sentiment: {overall_sentiment.get('label', 'UNKNOWN')} (score: {overall_sentiment.get('score', 0)})
Customer Intents Detected: {intent_summary}
Objections Found: {objection_summary}
Key Entities: {entity_summary}
Magic Moments: {moments_text if moments_text else "None detected"}

=== YOUR TASK ===
Provide a JSON response with EXACTLY this structure (no markdown, no code blocks):
{{
    "summary": "A 2-3 sentence executive summary of the call",
    "call_score": <integer 0-100>,
    "score_breakdown": {{
        "sentiment": <0-25>,
        "engagement": <0-25>,
        "objection_handling": <0-25>,
        "closing": <0-25>
    }},
    "agent_suggestions": [
        "Specific actionable suggestion 1",
        "Specific actionable suggestion 2",
        "Specific actionable suggestion 3"
    ],
    "conversion_probability": <float 0.0 to 1.0>
}}

Score criteria:
- sentiment: How positive was the overall call mood?
- engagement: How engaged was the customer?
- objection_handling: How well did the agent handle objections?
- closing: How strong was the closing?

Respond with ONLY the JSON object."""

    async def _groq_generate(self, context: str) -> dict:
        """Generate intelligence using Groq API (Llama 3.3 70B)."""
        try:
            # Groq SDK is synchronous — run in default executor
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a sales analytics expert. Always respond with valid JSON only. No markdown, no code blocks, no explanations."
                        },
                        {
                            "role": "user",
                            "content": context
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                    response_format={"type": "json_object"}
                )
            )
            
            text = response.choices[0].message.content.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            result = json.loads(text)
            
            # Validate and clamp values
            result["call_score"] = max(0, min(100, int(result.get("call_score", 50))))
            result["conversion_probability"] = max(0.0, min(1.0, float(result.get("conversion_probability", 0.5))))
            
            # Ensure score_breakdown exists
            if "score_breakdown" not in result:
                result["score_breakdown"] = {"sentiment": 15, "engagement": 15, "objection_handling": 15, "closing": 15}
            
            # Ensure agent_suggestions exists
            if "agent_suggestions" not in result or not result["agent_suggestions"]:
                result["agent_suggestions"] = ["Review call recording for improvement areas."]
            
            logger.info(f"Groq LLM generated score: {result['call_score']}/100")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return self._mock_generate({}, [])
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return self._mock_generate({}, [])
    
    def generate_sync(self, prompt: str) -> str:
        """Synchronous generation for RAG answer synthesis."""
        self._load_model()
        
        if not self._client:
            return ""
        
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful sales analytics assistant. Provide clear, concise answers based on the provided data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq sync generation failed: {e}")
            return ""
    
    def _mock_generate(self, nlp_results: dict, magic_moments: list) -> dict:
        """
        Generate mock intelligence based on NLP results.
        Used when LLM is not available.
        """
        objections = nlp_results.get("all_objections", [])
        overall = nlp_results.get("overall_sentiment", {"label": "NEUTRAL", "score": 0.5})
        turns = nlp_results.get("turns", [])
        
        # Calculate score from NLP data
        sentiment_score = 15
        if overall.get("label") == "POSITIVE":
            sentiment_score = 20
        elif overall.get("label") == "NEGATIVE":
            sentiment_score = 8
        
        positive_moments = sum(1 for m in magic_moments if m.get("moment_type") == "positive_turning_point")
        negative_moments = sum(1 for m in magic_moments if m.get("moment_type") == "negative_turning_point")
        
        # Engagement: based on conversation length, customer participation, and interest signals
        customer_turns = sum(1 for t in turns if t.get("speaker", "").lower() == "customer")
        has_scheduling = any(
            i.get("intent") in ("scheduling request", "positive interest", "closing agreement")
            for t in turns for i in t.get("intents", [])
        )
        engagement_score = min(25, 12 + customer_turns + (5 if has_scheduling else 0) + positive_moments * 2)
        
        # Objection handling: did the agent address objections? (positive moments after negatives = good handling)
        objection_count = len(set(o["category"] for o in objections))  # unique categories only
        if objection_count > 0:
            handled_ratio = positive_moments / max(objection_count, 1)
            objection_score = min(25, int(8 + handled_ratio * 12))
        else:
            objection_score = 20  # No objections = naturally higher
        
        closing_score = min(25, 12 + positive_moments * 4 - negative_moments * 3)
        
        total_score = sentiment_score + engagement_score + objection_score + closing_score
        
        # Build suggestions based on what was detected
        suggestions = []
        objection_categories = set(o["category"] for o in objections)
        
        if "price" in objection_categories:
            suggestions.append("Address pricing concerns early — consider leading with ROI data and case studies before quoting prices.")
        if "trust" in objection_categories:
            suggestions.append("Build trust by sharing specific customer success stories and offering a free trial period.")
        if "authority" in objection_categories:
            suggestions.append("Proactively ask about decision-makers early in the call and offer to include them in a follow-up demo.")
        if "urgency" in objection_categories:
            suggestions.append("Create urgency by highlighting limited-time offers or the cost of delay.")
        if "competitor" in objection_categories:
            suggestions.append("Prepare competitive comparison sheets and focus on unique differentiators rather than attacking competitors.")
        
        if not suggestions:
            suggestions = [
                "Maintain the positive engagement style throughout the call.",
                "Ask more open-ended questions to uncover hidden needs.",
                "Strengthen the closing by proposing clear next steps."
            ]
        
        conversion_prob = min(0.95, max(0.05, total_score / 100 + positive_moments * 0.05 - negative_moments * 0.1))
        
        return {
            "summary": f"The call showed {overall.get('label', 'mixed').lower()} sentiment overall. "
                       f"{'Customer raised ' + str(objection_count) + ' objection(s) including ' + ', '.join(objection_categories) + '.' if objections else 'No significant objections were raised.'} "
                       f"{'The call ended with positive momentum suggesting potential for conversion.' if positive_moments > negative_moments else 'Follow-up is recommended to address remaining concerns.'}",
            "call_score": max(0, min(100, total_score)),
            "score_breakdown": {
                "sentiment": sentiment_score,
                "engagement": engagement_score,
                "objection_handling": objection_score,
                "closing": max(0, closing_score)
            },
            "agent_suggestions": suggestions[:5],
            "conversion_probability": round(conversion_prob, 2)
        }


# Singleton
llm_service = LLMService()
