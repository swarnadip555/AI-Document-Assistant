"""
Document Loader Module
Responsible for loading PDF files, handling corrupt/empty files,
and extracting page-level text with comprehensive metadata.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pypdf
from langchain_core.documents import Document


def load_pdf(file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Load and parse a PDF file, extracting text and metadata per page.
    
    Args:
        file_path: Absolute or relative path to the PDF file.
        
    Returns:
        Tuple of:
          - List[Document]: LangChain Document objects for each valid page.
          - Dict[str, Any]: Metadata summary about the document (page count, size, char count, etc.)
          
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is not a valid PDF or has no extractable text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file format: {path.suffix}. Only PDF files are supported.")
        
    file_size_bytes = path.stat().st_size
    if file_size_bytes == 0:
        raise ValueError("The uploaded PDF file is empty (0 bytes).")

    documents: List[Document] = []
    total_characters = 0
    pages_with_text = 0
    doc_title = path.stem
    doc_metadata: Dict[str, Any] = {}

    try:
        reader = pypdf.PdfReader(str(path))
        num_pages = len(reader.pages)
        
        if num_pages == 0:
            raise ValueError("The PDF document does not contain any pages.")

        # Attempt to read PDF metadata if available
        if reader.metadata:
            if reader.metadata.title:
                doc_title = reader.metadata.title
            doc_metadata = {
                "author": reader.metadata.author or "Unknown",
                "creator": reader.metadata.creator or "Unknown",
                "producer": reader.metadata.producer or "Unknown",
            }

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1  # 1-indexed for user readability
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                page_text = ""
                
            clean_text = page_text.strip()
            char_count = len(clean_text)
            
            if clean_text:
                pages_with_text += 1
                total_characters += char_count
                
                doc = Document(
                    page_content=clean_text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "page": page_num,
                        "total_pages": num_pages,
                        "char_count": char_count
                    }
                )
                documents.append(doc)
            else:
                # Still record empty page placeholder if wanted, but do not chunk empty text
                pass

    except pypdf.errors.PdfReadError as e:
        raise ValueError(f"Corrupt or unreadable PDF file: {str(e)}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error reading PDF file: {str(e)}")

    if not documents:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "It may be a scanned image document that requires OCR."
        )

    summary_stats = {
        "file_name": path.name,
        "file_path": str(path),
        "file_size": file_size_bytes,
        "title": doc_title,
        "total_pages": len(reader.pages),
        "pages_with_text": pages_with_text,
        "total_characters": total_characters,
        "raw_metadata": doc_metadata
    }

    return documents, summary_stats

