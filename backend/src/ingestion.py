import os
import logging
import re
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}

def load_document(file_path: str) -> List[Document]:
    """
    Loads a single document based on file extension.
    Supports PDF, TXT, and DOCX formats.
    """
    ext = Path(file_path).suffix.lower()
    docs = []

    try:
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()

        elif ext == ".txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            docs = loader.load()

        elif ext == ".csv":
            import pandas as pd
            logger.info(f"Loading CSV file (limiting to first 2000 rows for performance): {file_path}")
            df = pd.read_csv(file_path, nrows=2000)
            
            cols = [c.lower() for c in df.columns]
            for _, row in df.iterrows():
                content_parts = []
                metadata = {"source": os.path.basename(file_path)}
                
                # Check for StackOverflow structure
                if "title" in cols and "body" in cols:
                    content_parts.append(f"Title: {row.get('Title', row.get('title'))}")
                    content_parts.append(f"Question: {row.get('Body', row.get('body'))}")
                elif "body" in cols:
                    content_parts.append(f"Content: {row.get('Body', row.get('body'))}")
                else:
                    # Generic CSV row representation
                    for col in df.columns:
                        content_parts.append(f"{col}: {row[col]}")
                
                # Add metadata columns if they exist
                for col in ["id", "score", "parentid", "score", "creationdate"]:
                    for actual_col in df.columns:
                        if actual_col.lower() == col:
                            metadata[col] = str(row[actual_col])
                
                # Clean HTML tags
                doc_text = "\n".join(content_parts)
                doc_text = re.sub(r'<[^>]*>', '', doc_text)
                
                docs.append(Document(page_content=doc_text, metadata=metadata))

        else:
            logger.warning(f"Skipping unsupported file type: {file_path}")
            return []

        logger.info(f"Loaded {len(docs)} pages/sections from: {file_path}")

    except Exception as e:
        logger.error(f"Failed to load '{file_path}': {str(e)}")

    return docs


def load_and_chunk_documents(data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Loads all supported documents recursively from a directory and splits them into chunks.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []

    all_docs = []
    # Search recursively using rglob
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(list(data_path.rglob(f"*{ext}")))
    
    # Sort files to ensure deterministic ingestion order
    files = sorted(files)

    if not files:
        logger.warning(f"No supported documents found in {data_dir}")
        return []

    logger.info(f"Found {len(files)} supported file(s) in {data_dir}")

    for file_path in files:
        docs = load_document(str(file_path))
        all_docs.extend(docs)

    if not all_docs:
        logger.warning("No content extracted from any documents.")
        return []

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Created {len(chunks)} chunks from {len(all_docs)} document sections.")
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_documents("./data")
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Sample chunk:\n{chunks[0].page_content[:200]}")
