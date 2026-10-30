"""
Vector Database Module
Manages FAISS vector store creation, local disk persistence,
loading, and querying with metadata.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import config


def create_vectorstore(documents: List[Document], embeddings) -> FAISS:
    """
    Create a new FAISS vector store in memory from document chunks.
    
    Args:
        documents: List of chunk Document objects.
        embeddings: Embeddings instance.
        
    Returns:
        FAISS vectorstore.
    """
    if not documents:
        raise ValueError("Cannot build vectorstore: No documents provided.")
    
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, folder_path: str = str(config.VECTORSTORE_DIR), index_name: str = "index") -> str:
    """
    Persist FAISS index and docstore to disk.
    
    Args:
        vectorstore: FAISS vectorstore instance.
        folder_path: Directory path to save to.
        index_name: Name of the index files.
        
    Returns:
        Folder path where vectorstore was saved.
    """
    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(folder_path=str(path), index_name=index_name)
    return str(path)


def load_vectorstore(folder_path: str = str(config.VECTORSTORE_DIR), embeddings=None, index_name: str = "index") -> Optional[FAISS]:
    """
    Load an existing FAISS vector store from disk.
    
    Args:
        folder_path: Directory path containing the saved index.
        embeddings: Embeddings instance.
        index_name: Name of the index file.
        
    Returns:
        FAISS vectorstore or None if not found or corrupted.
    """
    path = Path(folder_path)
    index_file = path / f"{index_name}.faiss"
    pkl_file = path / f"{index_name}.pkl"
    
    if not (index_file.exists() and pkl_file.exists()):
        return None
        
    try:
        vectorstore = FAISS.load_local(
            folder_path=str(path),
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        print(f"Warning: Failed to load FAISS index: {e}")
        return None


def clear_vectorstore(folder_path: str = str(config.VECTORSTORE_DIR)) -> bool:
    """
    Remove existing saved vectorstore files from disk.
    """
    path = Path(folder_path)
    if path.exists():
        for file in path.glob("*"):
            try:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            except Exception:
                pass
        return True
    return False

