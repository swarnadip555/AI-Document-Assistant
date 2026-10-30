"""
Text Splitter Module
Splits document pages into smaller, semantically coherent chunks
while preserving page numbering and source metadata.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def split_documents(
    documents: List[Document],
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Split a list of Document objects into overlapping text chunks.
    
    Args:
        documents: List of LangChain Document objects (one per page).
        chunk_size: Target character size for each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        Tuple of:
          - List[Document]: Chunks with preserved metadata + chunk_id.
          - Dict[str, Any]: Chunking statistics (count, avg length, etc.).
    """
    if not documents:
        return [], {"total_chunks": 0, "avg_chunk_size": 0}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        keep_separator=True
    )

    raw_chunks = text_splitter.split_documents(documents)
    
    # Enrich metadata for each chunk
    enriched_chunks: List[Document] = []
    chunk_lengths = []
    
    for idx, chunk in enumerate(raw_chunks):
        cleaned_content = chunk.page_content.strip()
        if not cleaned_content:
            continue
            
        c_len = len(cleaned_content)
        chunk_lengths.append(c_len)
        
        meta = dict(chunk.metadata)
        meta["chunk_id"] = idx
        meta["chunk_char_length"] = c_len
        # Estimate token count (~4 characters per token heuristic)
        meta["estimated_tokens"] = round(c_len / 4)
        
        enriched_chunks.append(Document(page_content=cleaned_content, metadata=meta))

    total_chunks = len(enriched_chunks)
    avg_len = round(sum(chunk_lengths) / total_chunks, 1) if total_chunks > 0 else 0
    min_len = min(chunk_lengths) if total_chunks > 0 else 0
    max_len = max(chunk_lengths) if total_chunks > 0 else 0

    stats = {
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_len,
        "min_chunk_size": min_len,
        "max_chunk_size": max_len,
        "chunk_size_setting": chunk_size,
        "chunk_overlap_setting": chunk_overlap
    }

    return enriched_chunks, stats

