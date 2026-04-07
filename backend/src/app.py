import os
import sys
import shutil
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import time
from fastapi.middleware.cors import CORSMiddleware

# ---------- PATH ----------
# This ensures that 'src' can be found regardless of whether you run
# from the root folder or the backend folder.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now we can import from src
try:
    from src.ingestion import load_and_chunk_documents
    from src.vector_store import create_vector_store, load_vector_store
    from src.qa_chain import get_qa_chain
    from src.retrievers import HybridRetrieverBuilder, get_hybrid_retriever_from_store
except ImportError as e:
    logger.error(f"Import error: {e}. Attempting relative import...")
    # Fallback for different execution contexts
    try:
        from ingestion import load_and_chunk_documents
        from vector_store import create_vector_store, load_vector_store
        from qa_chain import get_qa_chain
    except ImportError:
        logger.critical("Could not load src modules. Please check your folder structure.")
        raise e

app = FastAPI(title="Enterprise Document QA System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global state
vector_store = None
retriever = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    global vector_store, retriever, qa_chain
    try:
        if os.path.exists("./faiss_index"):
            logger.info("Loading existing vector store from ./faiss_index...")
            vector_store = load_vector_store()
            
            logger.info("Initializing Hybrid Retriever...")
            # Read weights from env or default to 0.5
            alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
            beta = float(os.getenv("HYBRID_BETA", "0.5"))
            
            builder = HybridRetrieverBuilder(vector_store)
            retriever = builder.get_hybrid_retriever(alpha=alpha, beta=beta)
            
            qa_chain = get_qa_chain(retriever)
            logger.info(f"System initialized successfully (Hybrid w/ Alpha={alpha}, Beta={beta}).")
        else:
            logger.info("No existing index found. System ready for document ingestion.")
    except Exception as e:
        logger.error(f"Failed to initialize system on startup: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Enterprise Document QA API (FastAPI)",
        "frontend_guide": "Ensure you have the frontend service running: `npm run dev` from the frontend directory.",
        "docs_url": "/docs",
        "health_url": "/health",
    }

@app.get("/health")
async def health_check():
    data_dir = Path("./data")
    files = list(data_dir.glob("*")) if data_dir.exists() else []
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    return {
        "status": "ok", 
        "initialized": qa_chain is not None,
        "has_files": len(files) > 0,
        "index_exists": os.path.exists("./faiss_index"),
        "total_docs": len([f for f in files if f.is_file()]),
        "total_size_kb": max(0, round(total_size / 1024, 2))
    }

class Query(BaseModel):
    text: str

class Response(BaseModel):
    answer: str
    sources: List[str]
    latency_ms: float
    confidence: float
    # Allow for future expansion without breaking old clients
    class Config:
        extra = "allow"

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    if not qa_chain:
        logger.warning("Query received but QA chain not initialized.")
        raise HTTPException(status_code=400, detail="System not initialized. Please upload and ingest documents first.")
    
    start_time = time.time()
    try:
        logger.info(f"Processing query: {query.text[:50]}...")
        result = qa_chain.invoke({"query": query.text})
        
        # Retrieval Diagnostics
        retrieved_docs = result.get("source_documents", [])
        logger.info(f"Retrieved {len(retrieved_docs)} potential context matches.")
        
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
        logger.info(f"Unique sources: {sources}")
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        confidence = 0.99 if retrieved_docs else 0.0
        
        # Log the full result for debugging validation issues
        logger.debug(f"Raw QA Chain result: {result}")
        
        answer = result.get("result", "")
        if answer is None:
            logger.warning("QA Chain returned None as result. Defaulting to empty string.")
            answer = ""
            
        return {
            "answer": str(answer),
            "sources": sources,
            "latency_ms": float(latency_ms),
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.error(f"Error during QA chain invocation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

@app.post("/ingest")
def ingest_documents():
    global vector_store, qa_chain
    try:
        logger.info("Starting document ingestion process (CPU intensive)...")
        chunks = load_and_chunk_documents("./data")
        
        if not chunks:
            logger.warning("Ingestion failed: No documents found or parsed in ./data")
            return {"message": "No documents found in data directory. Please upload files first."}
        
        # This part is heavy CPU work; using standard 'def' lets FastAPI 
        # run this in a threadpool so it doesn't block other requests.
        vector_store = create_vector_store(chunks)
        
        # Read weights from env or default to 0.5
        alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
        beta = float(os.getenv("HYBRID_BETA", "0.5"))
        
        # New Hybrid Logic: Persist chunks and build hybrid retriever
        builder = HybridRetrieverBuilder(vector_store)
        builder.save_chunks(chunks)
        retriever = builder.get_hybrid_retriever(chunks, alpha=alpha, beta=beta)
        
        qa_chain = get_qa_chain(retriever)
        
        logger.info(f"Ingestion complete. {len(chunks)} chunks indexed (Hybrid w/ Alpha={alpha}, Beta={beta}).")
        return {"message": f"Successfully ingested {len(chunks)} chunks using Hybrid Retrieval (Alpha={alpha}, Beta={beta})."}
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

def sanitize_filename(filename: str) -> str:
    """Basic filename sanitization."""
    if not filename:
        return "unnamed_file"
    # Remove any path components
    filename = os.path.basename(filename)
    # Remove non-alphanumeric/dot/underscore/hyphen
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        
        filename = getattr(file, 'filename', None) or "document"
        safe_name = sanitize_filename(filename)
            
        file_path = data_dir / safe_name
        
        logger.info(f"Uploading file: {filename} -> {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"message": f"File {safe_name} uploaded successfully. Call /ingest to re-index."}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """Returns a list of documents in the data directory."""
    try:
        data_dir = Path("./data")
        if not data_dir.exists():
            return []
        
        files = []
        for path in sorted(data_dir.iterdir()):
            if path.is_file():
                # Correctly format the modification timestamp
                import datetime
                mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                
                files.append({
                    "name": path.name,
                    "size_kb": max(1, round(path.stat().st_size / 1024)),
                    "modified": mtime
                })
        return files
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list documents.")

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Deletes a single document."""
    try:
        data_dir = Path("./data")
        file_path = data_dir / filename
        
        # Security check to prevent path traversal
        if not str(file_path.resolve()).startswith(str(data_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid filename")
            
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return {"message": f"Document {filename} deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail="Document not found.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.delete("/documents")
async def delete_documents():
    """Clears all documents and the vector index."""
    global vector_store, qa_chain
    try:
        # Clear data directory
        data_dir = Path("./data")
        if data_dir.exists():
            for path in data_dir.iterdir():
                if path.is_file():
                    path.unlink()
        
        # Clear FAISS index
        faiss_dir = Path("./faiss_index")
        if faiss_dir.exists():
            shutil.rmtree(str(faiss_dir))
            
        # Clear BM25 indexes (including legacy)
        for idx_file in ["bm25_chunks.json", "bm25_chunks.pkl"]:
            idx_path = Path(idx_file)
            if idx_path.exists():
                idx_path.unlink()
                logger.info(f"Removed index file: {idx_file}")
        
        # Reset state
        vector_store = None
        qa_chain = None
        
        logger.info("Knowledge base cleared successfully.")
        return {"message": "Knowledge base cleared successfully."}
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use the app object directly for consistency with the code block
    uvicorn.run(app, host="0.0.0.0", port=8000)
