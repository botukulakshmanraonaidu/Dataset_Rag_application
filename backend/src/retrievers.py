import os
import pickle
import json
import logging
import re
from typing import List, Optional, Any
from rank_bm25 import BM25Okapi
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    A custom hybrid retriever that combines BM25 keyword search and FAISS semantic search.
    Implements weighted scoring: final_score = alpha * bm25_score + beta * vector_score.
    """
    vector_store: Any
    chunks: List[Document]
    bm25: BM25Okapi
    alpha: float = 0.5  # Weight for BM25
    beta: float = 0.5   # Weight for Vector Search
    k: int = 4          # Number of documents to return
    threshold: float = 0.15 # Score threshold for fallback triggering
    use_reranking: bool = True # Whether to use FlashRank reranking

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_chunks(cls, vector_store, chunks: List[Document], alpha: float = 0.5, beta: float = 0.5, k: int = 4, threshold: float = 0.15, use_reranking: bool = True):
        """
        Factory method to initialize from chunks.
        """
        # Prepare corpus for BM25 (tokenized)
        tokenized_corpus = [cls._tokenize(doc.page_content) for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(vector_store=vector_store, chunks=chunks, bm25=bm25, alpha=alpha, beta=beta, k=k, threshold=threshold, use_reranking=use_reranking)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        return re.findall(r"\w+", text.lower())

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieves documents using hybrid scoring.
        """
        tokenized_query = self._tokenize(query)
        
        # 1. Get BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Get Vector scores (FAISS returns distance, smaller is better)
        # We use similarity_search_with_score to get (Document, score)
        vector_results = self.vector_store.similarity_search_with_score(query, k=len(self.chunks))
        
        # Create a mapping of chunk content to vector score safely
        # Note: We assume chunks are unique in content or use their index
        vector_score_map = {res[0].page_content: res[1] for res in vector_results}

        # 3. Normalize scores for weighted combination
        # BM25 normalization (max score to 1.0)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        
        # Vector normalization (FAISS L2 distance: 0 is perfect, higher is worse)
        # We invert it: 1 - (score / max_score)
        max_vector = max(vector_score_map.values()) if vector_score_map else 1.0

        hybrid_results = []
        for i, doc in enumerate(self.chunks):
            # BM25 Normalized
            bm25_norm = bm25_scores[i] / max_bm25
            
            # Vector Normalized (Inverted L2 distance)
            v_score = vector_score_map.get(doc.page_content, max_vector)
            vector_norm = 1.0 - (v_score / max_vector) if max_vector > 0 else 0.0
            
            # Final Weighted Score
            final_score = (self.alpha * bm25_norm) + (self.beta * vector_norm)
            
            hybrid_results.append((doc, final_score))

        # 4. Sort by final score descending
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Filter by threshold (IMPORTANT for Web Search Fallback)
        qualified_results = [res for res in hybrid_results if res[1] >= self.threshold]
        
        if not qualified_results:
            logger.info(f"No documents met the relevance threshold ({self.threshold}). Returning empty list.")
            return []

        # 6. Remove duplicates and collect candidates for reranking
        seen_content = set()
        unique_results = []
        candidate_limit = max(15, self.k * 3) if self.use_reranking else self.k
        for doc, score in qualified_results:
            if doc.page_content not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content)
            if len(unique_results) >= candidate_limit:
                break
                
        # 7. Apply FlashRank Reranking if enabled
        if self.use_reranking and unique_results:
            try:
                from flashrank import Ranker, RerankRequest
                global _global_ranker
                if '_global_ranker' not in globals():
                    logger.info("Initializing FlashRank Ranker...")
                    _global_ranker = Ranker()
                
                passages = [
                    {"id": i, "text": doc.page_content, "meta": doc.metadata}
                    for i, doc in enumerate(unique_results)
                ]
                rerank_request = RerankRequest(query=query, passages=passages)
                reranked_results = _global_ranker.rerank(rerank_request)
                
                # Reconstruct list of Documents in new rank order
                reranked_docs = []
                for item in reranked_results:
                    original_idx = item["id"]
                    reranked_docs.append(unique_results[original_idx])
                
                unique_results = reranked_docs
                logger.info(f"FlashRank reranking complete. Re-ranked {len(unique_results)} candidates.")
            except Exception as e:
                logger.error(f"FlashRank Reranking failed: {e}. Falling back to default hybrid order.")
                
        logger.info(f"Hybrid retrieval complete. Best score: {qualified_results[0][1]:.4f}")
        
        # 8. StackOverflow context enrichment: For each retrieved question, fetch its answers from the corpus.
        final_docs = []
        for doc in unique_results[:self.k]:
            final_docs.append(doc)
            doc_id = doc.metadata.get("id")
            # If it's a question (has 'id', no 'parentid')
            if doc_id and "parentid" not in doc.metadata:
                answers_found = 0
                for chunk in self.chunks:
                    parent_id = chunk.metadata.get("parentid")
                    if parent_id == doc_id:
                        if chunk not in final_docs:
                            final_docs.append(chunk)
                            answers_found += 1
                if answers_found > 0:
                    logger.info(f"Enriched question ID {doc_id} with {answers_found} answer chunk(s) from local corpus.")

        return final_docs

class HybridRetrieverBuilder:
    def __init__(self, vector_store, chunks_path="./bm25_chunks.json"):
        self.vector_store = vector_store
        self.chunks_path = chunks_path

    def save_chunks(self, chunks: List[Document]):
        """Persists chunks for BM25 re-indexing in a human-readable JSON format."""
        try:
            # Convert Document objects to JSON-serializable dictionaries
            serializable_chunks = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in chunks
            ]
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump(serializable_chunks, f, indent=2)
            
            logger.info(f"Saved {len(chunks)} chunks to {self.chunks_path}")
            
            # Legacy cleanup: remove old pickle file if it exists
            legacy_path = self.chunks_path.replace(".json", ".pkl")
            if os.path.exists(legacy_path):
                os.remove(legacy_path)
                logger.info(f"Removed legacy binary file: {legacy_path}")

        except Exception as e:
            logger.error(f"Failed to save chunks to JSON: {str(e)}")

    def load_chunks(self) -> List[Document]:
        """Loads persisted chunks from JSON."""
        # Check for JSON first
        if not os.path.exists(self.chunks_path):
            # Fallback to legacy pickle if JSON doesn't exist yet
            legacy_path = self.chunks_path.replace(".json", ".pkl")
            if os.path.exists(legacy_path):
                try:
                    with open(legacy_path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    return []
            return []
            
        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                serializable_chunks = json.load(f)
                
            # Reconstruct Document objects
            return [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in serializable_chunks
            ]
        except Exception as e:
            logger.error(f"Failed to load chunks from JSON: {str(e)}")
            return []

    def get_hybrid_retriever(self, chunks: List[Document] = None, 
                             alpha: float = 0.5, beta: float = 0.5, k: int = 4,
                             use_reranking: bool = True) -> BaseRetriever:
        """
        Builds the custom HybridRetriever instance.
        """
        if chunks is None:
            chunks = self.load_chunks()
            
        if not chunks:
            logger.warning("No chunks available for BM25. Falling back to pure Vector search.")
            return self.vector_store.as_retriever(search_kwargs={"k": k})

        # Return our custom HybridRetriever
        return HybridRetriever.from_chunks(
            vector_store=self.vector_store,
            chunks=chunks,
            alpha=alpha,
            beta=beta,
            k=k,
            use_reranking=use_reranking
        )

def get_hybrid_retriever_from_store(vector_store, chunks_path="./bm25_chunks.pkl"):
    """Utility to initialize during startup."""
    builder = HybridRetrieverBuilder(vector_store, chunks_path=chunks_path)
    return builder.get_hybrid_retriever()
