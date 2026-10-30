"""
Embeddings Module
Initializes and provides the embedding model using HuggingFace / SentenceTransformers.
Caches the model to prevent redundant reloads.
"""

import os
from typing import Optional, List
import config

_cached_embedding_instance = None


def get_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME):
    """
    Get or initialize the embedding model.
    Uses langchain_huggingface or langchain_community HuggingFaceEmbeddings,
    with automatic device detection (cuda / cpu).
    
    Args:
        model_name: HuggingFace model repo ID.
        
    Returns:
        Embeddings model instance compatible with LangChain FAISS.
    """
    global _cached_embedding_instance
    if _cached_embedding_instance is not None:
        return _cached_embedding_instance

    # Determine device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    # Try importing from langchain_huggingface first, fallback to langchain_community
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        except Exception:
            # Fallback wrapper directly using sentence-transformers
            from sentence_transformers import SentenceTransformer
            from langchain_core.embeddings import Embeddings

            class SentenceTransformerWrapper(Embeddings):
                def __init__(self, m_name: str, dev: str):
                    self.model = SentenceTransformer(m_name, device=dev)

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                    return embs.tolist()

                def embed_query(self, text: str) -> List[float]:
                    emb = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
                    return emb[0].tolist()

            embeddings = SentenceTransformerWrapper(model_name, device)

    _cached_embedding_instance = embeddings
    return embeddings

