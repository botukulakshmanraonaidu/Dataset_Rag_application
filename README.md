# 🧠 Enterprise Document QA — RAG Application

Welcome to the **Enterprise Document QA** system! This is a full-stack **Retrieval-Augmented Generation (RAG)** application. 
It allows you to upload enterprise documents (PDFs, Word docs, Text files) and ask natural-language questions about their content. The system uses advanced AI (Hybrid Retrieval combining both keyword and semantic search) to find the right information and generate accurate answers with citations.

---

## 🗂️ Project Structure

The project is divided into two main parts: the **Backend** (Python/FastAPI) and the **Frontend** (React/Vite).

```text
Enterprise_document/
│
├── backend/                        # ⚙️ Python Backend Server
│   ├── src/                        # Source code for the backend logic
│   │   ├── app.py                  # Main API server (FastAPI endpoints: /upload, /ask, etc.)
│   │   ├── ingestion.py            # Code to read and chunk uploaded documents
│   │   ├── vector_store.py         # Code to manage the FAISS vector database
│   │   ├── retrievers.py           # Logic for Hybrid Search (BM25 + FAISS)
│   │   ├── qa_chain.py             # LangChain logic to connect the LLM and retriever
│   │   └── evaluation.py           # Optional tools for evaluating answer quality
│   │
│   ├── data/                       # 📁 Folder where your uploaded documents are saved (Git ignored)
│   ├── faiss_index/                # 🧠 Folder where the vector database is stored (Git ignored)
│   ├── bm25_chunks.json            # 🧠 File storing keyword search indices (Git ignored)
│   ├── requirements.txt            # List of Python dependencies needed
│   └── .env                        # 🔑 Environment variables and API keys (Git ignored)
│
├── frontend/                       # 🎨 React + Vite User Interface
│   ├── src/                        # Source code for the frontend UI
│   │   ├── App.jsx                 # Main React component (the web page you see)
│   │   ├── App.css                 # Specific styles for components
│   │   └── index.css               # Global styles and design themes
│   │
│   ├── public/                     # Static files (like favicon, images)
│   ├── index.html                  # Main HTML file that loads the React app
│   ├── vite.config.js              # Configuration for the Vite build tool
│   └── package.json                # List of Node.js dependencies (React, Vite, etc.)
│
├── .gitignore                      # Tells Git which files to ignore (like passwords/large folders)
└── README.md                       # The instruction manual you are reading right now!
```

---

## 🚀 Getting Started Guide

Follow these steps to run the project on your local machine. You will need to start **both** the backend and the frontend.

### Prerequisites
Make sure you have installed:
- **Python** (version 3.10 or higher)
- **Node.js** (version 18 or higher)

### 1. Backend Setup (Terminal 1)

Open a terminal and navigate to the `backend` folder to set up the server.

```bash
# Move into the backend folder
cd backend

# Create a virtual environment (this keeps dependencies isolated)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install all the required Python libraries
pip install -r requirements.txt
```

**Set up your keys:**
Create a file named `.env` inside the `backend` folder and add your API keys:
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Start the Backend Server:**
```bash
# Run this command while still inside the backend/ folder
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```
*The backend API is now running at **http://localhost:8000***

### 2. Frontend Setup (Terminal 2)

Open a **new** terminal window and navigate to the `frontend` folder.

```bash
# Move into the frontend folder
cd frontend

# Install all the required Node.js libraries
npm install

# Start the frontend user interface
npm run dev
```
*The frontend is now running at **http://localhost:5173***

Open your browser and go to `http://localhost:5173` to use the application!

---

## ⚙️ How the System Works

1. **Upload via UI**: You upload a document (like a PDF) using the web interface.
2. **Backend Storage**: The backend saves this file into the `backend/data/` folder.
3. **Ingestion**: When you ask the system to ingest, it reads the document, breaks it into smaller "chunks", and creates two types of indexes:
   - A **FAISS Vector Index** (for understanding the *meaning* of words).
   - A **BM25 Keyword Index** (for finding exact word matches).
4. **Asking Questions**: You type a question in the UI.
5. **Hybrid Retrieval**: The backend searches both indexes (FAISS and BM25) to find the most relevant chunks of text from your documents.
6. **LLM Answer**: The backend sends your question and those relevant chunks to an AI model (LLM), which reads them and writes a clear answer, alongside verifying which documents it used.

---

## 📝 Important Notes on Git and Committing

To keep the repository clean and secure, certain files are **ignored** by Git (defined in `.gitignore`):
- **Dependencies**: `venv/` and `node_modules/` are huge and can be re-downloaded anytime using `pip install` or `npm install`.
- **Secrets**: `backend/.env` contains your passwords and API keys. NEVER commit this to GitHub.
- **Your Data**: `backend/data/`, `faiss_index/`, and `bm25_chunks.json` are unique to the files you upload locally, so they are not tracked.