import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# ---------- PATH SETUP ----------
# Ensure 'src' can be found regardless of execution directory
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

load_dotenv()

from src.vector_store import load_vector_store
from src.retrievers import HybridRetrieverBuilder
from src.qa_chain import get_qa_chain

try:
    from src.evaluation import run_evaluation
    HAS_RAGAS = True
except Exception as e:
    HAS_RAGAS = False
    RAGAS_ERROR = e


def load_settings() -> dict:
    settings_file = BACKEND_DIR / "settings.json"
    if not settings_file.exists():
        return {
            "model_name": "openai/gpt-4o",
            "temperature": 0.0,
            "max_tokens": 300,
            "hybrid_alpha": 0.5,
            "hybrid_beta": 0.5,
            "use_reranking": True
        }
    try:
        with open(settings_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def main():
    # Load dataset
    dataset_file = CURRENT_DIR / "test_dataset.json"
    if not dataset_file.exists():
        print(f"Error: test_dataset.json not found at {dataset_file}")
        return
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    print(f"Loaded {len(test_data)} test cases from test_dataset.json.")
    
    # Extract questions and ground truths
    test_questions = [item["question"] for item in test_data]
    ground_truths = [item["ground_truth"] for item in test_data]
    
    # Initialize RAG components
    settings = load_settings()
    faiss_dir = BACKEND_DIR / "faiss_index"
    if not faiss_dir.exists():
        print(f"Error: faiss_index does not exist at {faiss_dir}. Please run ingestion first.")
        return
        
    print("Loading vector store and initializing retriever...")
    vector_store = load_vector_store(store_path=str(faiss_dir))
    builder = HybridRetrieverBuilder(vector_store, chunks_path=str(BACKEND_DIR / "bm25_chunks.json"))
    retriever = builder.get_hybrid_retriever(
        alpha=settings.get("hybrid_alpha", 0.5),
        beta=settings.get("hybrid_beta", 0.5),
        use_reranking=settings.get("use_reranking", True)
    )
    
    print("Creating QA Chain...")
    qa_chain = get_qa_chain(retriever, settings)
    
    print("\nRunning evaluation on RAG pipeline...")
    should_run_fallback = not HAS_RAGAS
    
    if HAS_RAGAS:
        try:
            # Run Ragas evaluation
            scores = run_evaluation(qa_chain, test_questions, ground_truths)
            print("\n=== RAGAS Evaluation Results ===")
            for metric, score in scores.items():
                print(f"{metric}: {score:.4f}")
        except Exception as e:
            print(f"\nRAGAS Evaluation encountered a runtime issue: {e}")
            should_run_fallback = True
    else:
        print(f"\nRAGAS library is unavailable due to an import error: {RAGAS_ERROR}")
        
    if should_run_fallback:
        print("Falling back to generating and printing raw test results...\n")
        print("=" * 80)
        for i, item in enumerate(test_data):
            print(f"\nTest Case {i+1}/10")
            print(f"Question: {item['question']}")
            print(f"Source Document: {item['source']}")
            print(f"Expected Ground Truth:\n  {item['ground_truth']}")
            
            try:
                res = qa_chain.invoke({"query": item["question"]})
                answer = res.get("result", "")
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in res.get("source_documents", [])]))
                
                print(f"RAG LLM Output (must be exactly 4 lines):\n---\n{answer}\n---")
                print(f"Retrieved Sources: {sources}")
                
                # Validation checking
                lines = [l for l in answer.split('\n') if l.strip()]
                if len(lines) != 4:
                    print(f"[WARNING] Response is {len(lines)} lines (expected exactly 4 lines).")
                else:
                    print("[OK] Constraint Check: Response has exactly 4 lines.")
            except Exception as inner_e:
                print(f"Failed to query LLM for this test case: {inner_e}")
            print("-" * 80)

if __name__ == "__main__":
    main()
