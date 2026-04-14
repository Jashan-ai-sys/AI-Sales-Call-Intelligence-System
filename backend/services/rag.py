"""
RAG Service — Retrieval Augmented Generation
Stores call transcripts in ChromaDB vector store and enables
semantic search over historical calls for pattern discovery.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RAGService:
    """
    Vector database powered by ChromaDB + Sentence Transformers.
    Enables queries like:
      - "What are common objections?"
      - "What works best for closing deals?"
      - "Show me calls with pricing concerns"
    """
    
    def __init__(self):
        self._client = None
        self._collection = None
        self._embedder = None
        self._loaded = False
    
    def _load(self):
        """Lazy-load ChromaDB and embedding model."""
        if self._loaded:
            return
            
        try:
            import chromadb
            from chromadb.config import Settings
            from backend.config import CHROMA_DIR
            
            logger.info("Initializing ChromaDB...")
            try:
                self._client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
            except Exception as e:
                if "_type" in str(e) or "capture" in str(e):
                    logger.warning(f"ChromaDB telemetry error caught, attempting fallback init: {e}")
                    # Fallback with minimal settings
                    self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                else:
                    raise e
            self._collection = self._client.get_or_create_collection(
                name="sales_calls",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB loaded. Collection has {self._collection.count()} documents.")
            
        except ImportError:
            logger.warning("chromadb not installed, RAG will use mock responses")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            
        self._loaded = True
    
    def store_call(self, call_id: str, transcript: str, metadata: dict):
        """
        Store a call transcript in the vector database.
        Chunks the transcript for better retrieval.
        """
        self._load()
        
        if not self._collection:
            logger.warning("ChromaDB not available, skipping storage")
            return
        
        # Chunk the transcript into segments for better retrieval
        chunks = self._chunk_transcript(transcript, call_id)
        
        # Prepare for ChromaDB
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        try:
            # Upsert (update if exists, insert if new)
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Stored {len(chunks)} chunks for call {call_id}")
        except Exception as e:
            logger.error(f"Failed to store call in ChromaDB: {e}")
    
    def _chunk_transcript(self, transcript: str, call_id: str) -> list:
        """
        Split transcript into meaningful chunks for embedding.
        Strategy: split by speaker turns, then merge small chunks.
        """
        # Split by sentences (roughly)
        sentences = transcript.replace("?", "?|").replace(".", ".|").replace("!", "!|").split("|")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > 500:
                if current_chunk:
                    chunks.append({
                        "id": f"{call_id}_chunk_{chunk_idx}",
                        "text": current_chunk.strip(),
                        "metadata": {
                            "call_id": call_id,
                            "chunk_index": chunk_idx,
                            "type": "transcript_chunk"
                        }
                    })
                    chunk_idx += 1
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        # Add remaining text
        if current_chunk.strip():
            chunks.append({
                "id": f"{call_id}_chunk_{chunk_idx}",
                "text": current_chunk.strip(),
                "metadata": {
                    "call_id": call_id,
                    "chunk_index": chunk_idx,
                    "type": "transcript_chunk"
                }
            })
        
        # Also store the full analysis summary as a separate document
        chunks.append({
            "id": f"{call_id}_full",
            "text": transcript[:2000],  # Store truncated full text
            "metadata": {
                "call_id": call_id,
                "chunk_index": -1,
                "type": "full_transcript"
            }
        })
        
        return chunks
    
    def store_call_insights(self, call_id: str, summary: str, objections: list, 
                            score: int, conversion_prob: float):
        """Store analysis insights as separate searchable documents."""
        self._load()
        
        if not self._collection:
            return
        
        docs = []
        ids = []
        metadatas = []
        
        # Store summary
        if summary:
            docs.append(f"Call Summary: {summary}")
            ids.append(f"{call_id}_summary")
            metadatas.append({
                "call_id": call_id,
                "type": "summary",
                "call_score": score,
                "conversion_probability": conversion_prob
            })
        
        # Store objections
        if objections:
            obj_text = "Objections detected: " + "; ".join(
                f"{o['category']}: {o['text'][:100]}" for o in objections
            )
            docs.append(obj_text)
            ids.append(f"{call_id}_objections")
            metadatas.append({
                "call_id": call_id,
                "type": "objections",
                "objection_categories": ",".join(set(o["category"] for o in objections))
            })
        
        if docs:
            try:
                self._collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
                logger.info(f"Stored insights for call {call_id}")
            except Exception as e:
                logger.error(f"Failed to store insights: {e}")
    
    def query(self, query_text: str, top_k: int = 5) -> dict:
        """
        Semantic search over stored calls.
        
        Returns:
            {
                "results": [{"call_id": str, "text": str, "score": float, "metadata": dict}],
                "answer": str  # Generated answer if LLM available
            }
        """
        self._load()
        
        if not self._collection or self._collection.count() == 0:
            return {
                "results": [],
                "answer": self._generate_mock_answer(query_text)
            }
        
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "call_id": results["metadatas"][0][i].get("call_id", "unknown"),
                    "text": results["documents"][0][i][:300],
                    "score": round(1 - results["distances"][0][i], 4),
                    "metadata": results["metadatas"][0][i]
                })
            
            # Build context for answer generation
            context = "\n\n".join([f"[Source {i+1}] {r['text']}" for i, r in enumerate(formatted_results[:3])])
            answer = self._synthesize_answer(query_text, context)
            
            return {
                "results": formatted_results,
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "results": [],
                "answer": f"Error performing search: {str(e)}"
            }
    
    def _synthesize_answer(self, query: str, context: str) -> str:
        """Synthesize an answer from retrieved context using LLM."""
        try:
            from backend.services.llm import llm_service
            
            prompt = f"""Based on the following sales call data, answer this question concisely:

Question: {query}

Context from past calls:
{context}

Provide a clear, actionable answer based on the data. If the data doesn't contain enough information, say so."""
            
            answer = llm_service.generate_sync(prompt)
            if answer:
                return answer
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
        
        return f"Based on {len(context.split('[Source'))-1} matching call(s), here are relevant excerpts. Review the sources for detailed insights."
    
    def _generate_mock_answer(self, query: str) -> str:
        """Generate a helpful response when no data is available."""
        query_lower = query.lower()
        
        if "objection" in query_lower:
            return "No calls analyzed yet. Upload sales call recordings to build your objection database. Common objections typically include pricing, trust, urgency, and authority concerns."
        elif "closing" in query_lower or "conversion" in query_lower:
            return "No historical data available yet. Upload and analyze sales calls to discover your team's most effective closing strategies."
        elif "best" in query_lower or "top" in query_lower:
            return "Start by uploading your sales calls. The system will identify patterns from your best-performing calls automatically."
        else:
            return "No calls in the database yet. Upload audio recordings to start building your sales intelligence knowledge base."
    
    def get_stats(self) -> dict:
        """Get vector DB statistics."""
        self._load()
        
        if not self._collection:
            return {"total_documents": 0, "status": "not_initialized"}
        
        return {
            "total_documents": self._collection.count(),
            "status": "active"
        }


# Singleton
rag_service = RAGService()
