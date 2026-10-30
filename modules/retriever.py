"""
Retriever Module
Extracts relevant context chunks from the FAISS vector database
using similarity search or MMR, preserving citations and scores.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import config


def retrieve_relevant_chunks(
    vectorstore: FAISS,
    query: str,
    top_k: int = config.TOP_K_RETRIEVAL,
    search_type: str = "similarity"
) -> List[Dict[str, Any]]:
    """
    Search vectorstore for chunks relevant to query.
    
    Args:
        vectorstore: FAISS index.
        query: User question or search query string.
        top_k: Number of chunks to retrieve.
        search_type: 'similarity' or 'mmr'.
        
    Returns:
        List of dictionaries with keys:
          - 'document': Document object
          - 'content': str text
          - 'page': int (1-based page number)
          - 'source': str (file name)
          - 'score': float (similarity score / distance)
          - 'chunk_id': int
    """
    if not vectorstore:
        return []

    results = []

    if search_type == "mmr":
        # Maximal Marginal Relevance to reduce redundancy
        docs = vectorstore.max_marginal_relevance_search(query, k=top_k, fetch_k=top_k * 3)
        for doc in docs:
            results.append({
                "document": doc,
                "content": doc.page_content,
                "page": doc.metadata.get("page", 1),
                "source": doc.metadata.get("source", "Document"),
                "score": None,
                "chunk_id": doc.metadata.get("chunk_id", 0)
            })
    else:
        # Similarity search with score (FAISS L2 distance: lower is closer)
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
        for doc, score in docs_and_scores:
            # Convert L2 distance or cosine to an intuitive similarity percentage
            # For normalized embeddings, distance = 2*(1-cos_sim), so cos_sim = 1 - (dist / 2)
            sim_score = max(0.0, min(1.0, 1.0 - (float(score) / 2.0)))
            results.append({
                "document": doc,
                "content": doc.page_content,
                "page": doc.metadata.get("page", 1),
                "source": doc.metadata.get("source", "Document"),
                "score": round(sim_score * 100, 1),
                "raw_distance": float(score),
                "chunk_id": doc.metadata.get("chunk_id", 0)
            })

    return results


def format_context_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clean, numbered context block with page indicators.
    """
    if not retrieved_chunks:
        return ""

    context_parts = []
    for i, item in enumerate(retrieved_chunks, 1):
        page = item.get("page", "?")
        source = item.get("source", "Document")
        content = item.get("content", "").strip()
        context_parts.append(
            f"[Passage {i} | Page {page} of {source}]\n{content}"
        )

    return "\n\n".join(context_parts)

