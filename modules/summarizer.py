"""
Summarizer Module
Generates comprehensive executive summaries, key highlights,
and topic breakdowns from the document pages.
"""

import os
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import config


def generate_extractive_summary(documents: List[Document], max_points: int = 5) -> Dict[str, Any]:
    """
    Generate structured summary without requiring external API calls.
    Extracts high-importance sentences, headers, and bullet points.
    """
    if not documents:
        return {
            "executive_summary": "No content available to summarize.",
            "key_takeaways": [],
            "sections_covered": []
        }

    total_pages = len(documents)
    all_text = " ".join([d.page_content for d in documents])

    # Sample sentences from beginning, middle, and end
    sampled_chunks = []
    if len(documents) <= 4:
        sampled_chunks = documents
    else:
        # Take first 2 pages, middle 1-2 pages, and last page
        mid = len(documents) // 2
        sampled_chunks = [documents[0], documents[1], documents[mid], documents[-1]]

    # Extract clean sentences
    key_sentences = []
    seen = set()
    for doc in sampled_chunks:
        page_num = doc.metadata.get("page", 1)
        sentences = re.split(r"(?<=[.?!])\s+", doc.page_content)
        for s in sentences:
            s_clean = s.strip()
            # Pick substantive sentences (length between 40 and 220 chars)
            if 40 <= len(s_clean) <= 220 and not s_clean.startswith(("http", "www", "Page")):
                if s_clean.lower() not in seen:
                    seen.add(s_clean.lower())
                    key_sentences.append((page_num, s_clean))
            if len(key_sentences) >= max_points * 2:
                break

    takeaways = [f"{sent} (Page {pg})" for pg, sent in key_sentences[:max_points]]
    
    first_few = [s for _, s in key_sentences[:3]]
    exec_summary = " ".join(first_few) if first_few else all_text[:400] + "..."

    # Detect topics or section titles
    lines = all_text.split("\n")
    potential_headers = []
    for line in lines:
        line_s = line.strip()
        # Common title/header patterns: short, title case or numbered like "1. Introduction"
        if 4 < len(line_s) < 60 and (re.match(r"^\d+[\.\)]\s+", line_s) or line_s.isupper() or line_s.istitle()):
            if line_s not in potential_headers and not line_s.startswith("Page"):
                potential_headers.append(line_s)
        if len(potential_headers) >= 6:
            break

    return {
        "executive_summary": exec_summary,
        "key_takeaways": takeaways,
        "sections_covered": potential_headers,
        "total_pages_analyzed": total_pages,
        "mode": "Extractive Synthesis"
    }


def generate_summary(
    documents: List[Document],
    api_key: Optional[str] = None,
    provider: str = "local"
) -> Dict[str, Any]:
    """
    Generate document summary using Gemini API if key is available,
    or fallback to high-quality extractive synthesis.
    """
    if not documents:
        return {
            "executive_summary": "No document content provided.",
            "key_takeaways": [],
            "sections_covered": []
        }

    gemini_key = api_key or os.getenv("GEMINI_API_KEY")

    if provider == "gemini" and gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)

            # Combine representative text up to 12,000 characters
            combined_text = ""
            for doc in documents[:10]:
                combined_text += f"\n--- Page {doc.metadata.get('page', 1)} ---\n" + doc.page_content[:1500]

            prompt = f"""You are an expert technical editor. Summarize the following document extracted from a PDF.

Provide your response in this exact format:
### EXECUTIVE SUMMARY
[Write a concise 2-3 paragraph executive summary of the document]

### KEY TAKEAWAYS
- [Takeaway 1 with source page]
- [Takeaway 2 with source page]
- [Takeaway 3 with source page]
- [Takeaway 4 with source page]
- [Takeaway 5 with source page]

### TOPICS & SECTIONS
- [Topic 1]
- [Topic 2]
- [Topic 3]

DOCUMENT TEXT:
{combined_text[:12000]}
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw_text = response.text.strip()
            
            return {
                "executive_summary": raw_text,
                "key_takeaways": [],
                "sections_covered": [],
                "raw_markdown": raw_text,
                "total_pages_analyzed": len(documents),
                "mode": "Gemini 2.5 Flash"
            }
        except Exception as e:
            print(f"Summarizer API failed: {e}. Falling back.")

    # Local fallback
    return generate_extractive_summary(documents)

