import os
import sys
import shutil
import logging
import re
import json
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
    from src.ingestion import load_and_chunk_documents, SUPPORTED_EXTENSIONS
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global state
vector_store = None
retriever = None
qa_chain = None

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"

def load_settings() -> dict:
    default_settings = {
        "model_name": "openai/gpt-4o",
        "temperature": 0.0,
        "max_tokens": 1000,
        "hybrid_alpha": 0.5,
        "hybrid_beta": 0.5,
        "use_reranking": True
    }
    if not SETTINGS_FILE.exists():
        return default_settings
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading settings.json: {e}")
        return default_settings

def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving settings.json: {e}")

def reinitialize_system():
    global vector_store, retriever, qa_chain
    try:
        settings = load_settings()
        if os.path.exists("./faiss_index"):
            if vector_store is None:
                logger.info("Loading existing vector store from ./faiss_index...")
                vector_store = load_vector_store()
            
            logger.info("Initializing Hybrid Retriever with current settings...")
            alpha = settings.get("hybrid_alpha", 0.5)
            beta = settings.get("hybrid_beta", 0.5)
            use_reranking = settings.get("use_reranking", True)
            
            builder = HybridRetrieverBuilder(vector_store)
            retriever = builder.get_hybrid_retriever(
                alpha=alpha,
                beta=beta,
                use_reranking=use_reranking
            )
            
            qa_chain = get_qa_chain(retriever, settings)
            logger.info(f"System initialized successfully (Hybrid w/ Alpha={alpha}, Beta={beta}, Rerank={use_reranking}).")
        else:
            logger.info("No existing index found. System ready for document ingestion.")
            retriever = None
            qa_chain = None
    except Exception as e:
        logger.error(f"Failed to reinitialize system: {str(e)}")

DEFAULT_POLICY_TEXT = """# ACME Corporation - Employee Handbook & Key Policies

## Core Values of the Organization
ACME Corporation is built on four fundamental core values:
1. Integrity: We hold ourselves to the highest ethical standards in all interactions.
2. Innovation: We constantly push boundaries to create forward-thinking solutions.
3. Customer-First: Our customers' success and satisfaction drive everything we do.
4. Collaboration: We believe that diverse, inclusive teams produce the best results.

## Remote Work Policy
At ACME Corporation, we support a hybrid work model to encourage work-life balance:
- Employees are eligible for up to 3 days of remote work per week, subject to approval from their direct manager.
- Core collaboration days are Tuesdays and Thursdays, during which all local employees are expected to work from the office.
- A high-speed internet connection and a quiet, secure workspace are required for remote work days.

## Time Off and Vacation Requests
We believe rest is essential for high performance. Our vacation policy is as follows:
- Full-time employees receive 20 days of paid annual leave per calendar year.
- To request time off or vacation, employees must submit a request through the internal HR Portal (HRIS) at least two weeks (14 days) in advance.
- For emergency leave or sick leave, notify your manager as early as possible on the day of absence.

## General Security & Compliance Policies
- Clean Desk Policy: Employees must lock their workstations when leaving their desks and secure any physical documents containing sensitive data.
- Device Security: All work devices must run the corporate security suite and use multi-factor authentication (MFA) for access.
- Code of Conduct: We maintain a professional, respectful, and harassment-free work environment for all employees.
"""

async def initialize_system_background():
    """Initializes the vector store, retriever, and QA chain in the background to avoid blocking port binding."""
    global vector_store
    import asyncio
    try:
        logger.info("Starting background system initialization...")
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        
        # Check for supported files recursively
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(list(data_dir.rglob(f"*{ext}")))
        
        # If no faiss_index exists, automatically ingest on startup
        if not os.path.exists("./faiss_index"):
            # If no supported files exist recursively, seed default dataset
            if not files:
                logger.info("No documents or index found. Seeding default ACME company policies dataset...")
                try:
                    default_file = data_dir / "company_policies.txt"
                    with open(default_file, "w", encoding="utf-8") as f:
                        f.write(DEFAULT_POLICY_TEXT)
                    # Refresh files list
                    files = []
                    for ext in SUPPORTED_EXTENSIONS:
                        files.extend(list(data_dir.rglob(f"*{ext}")))
                except Exception as e:
                    logger.error(f"Failed to auto-seed default dataset: {str(e)}")
            
            # Ingest whatever files we have recursively
            if files:
                logger.info(f"Automatically ingesting dataset recursively in background ({len(files)} files found)...")
                try:
                    loop = asyncio.get_event_loop()
                    chunks = await loop.run_in_executor(None, load_and_chunk_documents, "./data")
                    if chunks:
                        vector_store = await loop.run_in_executor(None, create_vector_store, chunks)
                        builder = HybridRetrieverBuilder(vector_store)
                        await loop.run_in_executor(None, builder.save_chunks, chunks)
                        logger.info("Dataset ingested successfully in background.")
                except Exception as e:
                    logger.error(f"Failed to auto-ingest dataset in background: {str(e)}")
        
        # Reinitialize system (which loads index and models) in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, reinitialize_system)
        logger.info("Background system initialization complete.")
    except Exception as e:
        logger.critical(f"Unhandled exception in background initialization: {str(e)}")

@app.on_event("startup")
async def startup_event():
    import asyncio
    # Start background initialization task to prevent blocking the port binding
    asyncio.create_task(initialize_system_background())

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "message": "Enterprise Document QA API (FastAPI)",
        "frontend_guide": "Ensure you have the frontend service running: `npm run dev` from the frontend directory.",
        "docs_url": "/docs",
        "health_url": "/health",
    }

@app.api_route("/health", methods=["GET", "HEAD"])
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
        if os.path.exists("./faiss_index"):
            raise HTTPException(status_code=400, detail="System is still initializing. Please wait a few moments and try again.")
        else:
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

class SettingsModel(BaseModel):
    model_name: str
    temperature: float
    max_tokens: int
    hybrid_alpha: float
    hybrid_beta: float
    use_reranking: bool

@app.get("/settings")
def get_settings():
    return load_settings()

@app.post("/settings")
def update_settings(new_settings: SettingsModel):
    save_settings(new_settings.model_dump())
    reinitialize_system()
    return {"message": "Settings updated successfully", "settings": load_settings()}

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
        
        # New Hybrid Logic: Persist chunks
        builder = HybridRetrieverBuilder(vector_store)
        builder.save_chunks(chunks)
        
        # Reinitialize everything with current settings
        reinitialize_system()
        
        settings = load_settings()
        logger.info(f"Ingestion complete. {len(chunks)} chunks indexed (Hybrid w/ Alpha={settings.get('hybrid_alpha')}, Beta={settings.get('hybrid_beta')}).")
        return {"message": f"Successfully ingested {len(chunks)} chunks using Hybrid Retrieval."}
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
    """Returns a list of documents recursively in the data directory."""
    try:
        data_dir = Path("./data")
        if not data_dir.exists():
            return []
        
        files = []
        paths = []
        for ext in SUPPORTED_EXTENSIONS:
            paths.extend(list(data_dir.rglob(f"*{ext}")))
            
        for path in sorted(paths):
            if path.is_file():
                # Correctly format the modification timestamp
                import datetime
                mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                
                # Show relative subpaths (e.g. archive/Questions.csv)
                rel_name = path.relative_to(data_dir).as_posix()
                
                files.append({
                    "name": rel_name,
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
    global vector_store, qa_chain, retriever
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
        retriever = None
        
        logger.info("Knowledge base cleared successfully.")
        return {"message": "Knowledge base cleared successfully."}
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use the app object directly for consistency with the code block
    uvicorn.run(app, host="0.0.0.0", port=8000)
