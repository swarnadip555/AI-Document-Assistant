"""
MCQ Generator Module
Generates 4-option multiple-choice questions with answers,
explanations, and page citations from the processed PDF.
"""

import os
import re
import json
import random
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


def generate_rule_based_mcqs(documents: List[Document], num_questions: int = 4) -> List[Dict[str, Any]]:
    """
    Offline heuristic MCQ generator that extracts factual statements, definitions,
    and key terms from document pages to build valid 4-option multiple choice questions.
    """
    if not documents:
        return []

    mcqs = []
    
    # 1. Collect candidate statements from all pages
    candidates = []
    for doc in documents:
        page_num = doc.metadata.get("page", 1)
        # Check bullet points or sentences
        lines = [line.strip("-•* \t") for line in doc.page_content.split("\n") if len(line.strip()) > 30]
        sentences = re.split(r"(?<=[.?!])\s+", doc.page_content)
        all_units = list(dict.fromkeys(lines + sentences))

        for s_clean in all_units:
            s_clean = s_clean.strip()
            # Pattern A: "Subject: description" or "Subject - description"
            colon_match = re.search(r"^([A-Z][A-Za-z0-9\s\(\)\/\-]{2,35})\s*[:\-]\s*(.+)", s_clean)
            if colon_match and len(s_clean) >= 35:
                subj = colon_match.group(1).strip()
                pred = colon_match.group(2).strip()
                if len(pred) > 20:
                    candidates.append({
                        "page": page_num,
                        "subject": subj,
                        "verb": "specifies",
                        "predicate": pred,
                        "full_sentence": s_clean
                    })
                continue

            # Pattern B: "Subject is/are/provides/enforces predicate"
            verb_match = re.search(r"^([A-Z][A-Za-z0-9\s\-]{2,35})\s+(is|are|refers to|means|provides|allows|combines|enforces|monitors)\s+(.+)", s_clean)
            if verb_match and len(s_clean) >= 35:
                candidates.append({
                    "page": page_num,
                    "subject": verb_match.group(1).strip(),
                    "verb": verb_match.group(2).strip(),
                    "predicate": verb_match.group(3).strip(),
                    "full_sentence": s_clean
                })

    # Distractor pool
    all_subjects = list(dict.fromkeys([c["subject"] for c in candidates]))
    curated_distractors = [
        "Unauthenticated public subnet routing",
        "Legacy single-tier relational data store",
        "Manual nightly backup replication",
        "Non-replicated static log storage",
        "Unencrypted plain text communication"
    ]

    used_questions = set()
    for item in candidates:
        if len(mcqs) >= num_questions:
            break

        subj = item["subject"]
        if subj in used_questions:
            continue
        used_questions.add(subj)

        pred_preview = item["predicate"][:120].rstrip(".")
        question_text = f"According to the document (Page {item['page']}), which component {item['verb']}: \"{pred_preview}\"?"

        # Distractors
        other_subjs = [s for s in all_subjects if s.lower() != subj.lower() and len(s) < 40]
        random.seed(len(mcqs) * 7 + 13)
        random.shuffle(other_subjs)
        distractors = other_subjs[:3]
        while len(distractors) < 3:
            distractors.append(curated_distractors[len(distractors)])

        options_list = [subj] + distractors[:3]
        random.shuffle(options_list)

        correct_idx = options_list.index(subj)
        correct_letter = ["A", "B", "C", "D"][correct_idx]

        mcqs.append({
            "question": question_text,
            "options": {
                "A": options_list[0],
                "B": options_list[1],
                "C": options_list[2],
                "D": options_list[3],
            },
            "correct_answer": correct_letter,
            "explanation": f"Refer to Page {item['page']}: \"{item['full_sentence'][:160]}...\"",
            "page": item["page"]
        })

    # If still need more questions, build page-level topic questions
    while len(mcqs) < num_questions and len(mcqs) < len(documents):
        doc_idx = len(mcqs)
        doc = documents[doc_idx % len(documents)]
        p_num = doc.metadata.get("page", doc_idx + 1)
        
        words = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", doc.page_content) if w not in ("Page", "The", "This", "With", "According")]
        chosen_word = words[0] if words else f"Section {p_num}"

        mcqs.append({
            "question": f"Which core focus area is highlighted on Page {p_num} of the document?",
            "options": {
                "A": chosen_word,
                "B": "Deprecated mainframes",
                "C": "Analog sensor calibration",
                "D": "Offline batch printing"
            },
            "correct_answer": "A",
            "explanation": f"Detailed on Page {p_num} of the processed document.",
            "page": p_num
        })

    return mcqs


def generate_mcqs(
    documents: List[Document],
    num_questions: int = 5,
    api_key: Optional[str] = None,
    provider: str = "local"
) -> List[Dict[str, Any]]:
    """
    Generate interactive MCQs using Gemini API or rule-based fallback.
    """
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")

    if provider == "gemini" and gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)

            sample_text = ""
            for doc in documents[:8]:
                sample_text += f"\n[Page {doc.metadata.get('page', 1)}]\n" + doc.page_content[:1200]

            prompt = f"""You are an educational assessment expert. Generate {num_questions} multiple-choice questions (MCQs) based STRICTLY on the document text provided below.

For each question:
- Provide 4 distinct options: A, B, C, D.
- Design one clearly correct answer supported by the text.
- Provide plausible distractors based on concepts in the text.
- State the correct answer letter ("A", "B", "C", or "D").
- Provide an explanation quoting or referencing the exact text and page number.
- Include the source page number.

OUTPUT FORMAT:
Return a STRICT JSON array of objects with the exact schema:
[
  {{
    "question": "Question text here?",
    "options": {{
      "A": "Option 1",
      "B": "Option 2",
      "C": "Option 3",
      "D": "Option 4"
    }},
    "correct_answer": "A",
    "explanation": "Why this is correct and citation",
    "page": 1
  }}
]

DOCUMENT TEXT:
{sample_text[:10000]}
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw = response.text.strip()
            
            # Clean json fences
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
                
            parsed = json.loads(raw.strip())
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception as e:
            print(f"Gemini MCQ generation failed: {e}. Falling back.")

    return generate_rule_based_mcqs(documents, num_questions=num_questions)
