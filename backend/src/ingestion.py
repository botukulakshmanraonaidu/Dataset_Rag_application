import os
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

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

        else:
            logger.warning(f"Skipping unsupported file type: {file_path}")
            return []

        logger.info(f"Loaded {len(docs)} pages/sections from: {file_path}")

    except Exception as e:
        logger.error(f"Failed to load '{file_path}': {str(e)}")

    return docs


def load_and_chunk_documents(data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Loads all supported documents from a directory and splits them into chunks.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []

    all_docs = []
    files = [f for f in data_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

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
