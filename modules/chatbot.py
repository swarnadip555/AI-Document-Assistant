"""
Chatbot Module
Implements the RAG (Retrieval-Augmented Generation) Question Answering pipeline,
incorporating prompt engineering, source citations, and conversation history.
"""

import os
import re
from typing import List, Dict, Any, Optional
from modules.retriever import retrieve_relevant_chunks, format_context_for_prompt
import config

# Global cache for local Hugging Face pipeline
_local_pipeline = None


def get_local_hf_pipeline(model_name: str = config.DEFAULT_HF_MODEL):
    """
    Load and cache a local Hugging Face model and tokenizer.
    Uses direct AutoModelForSeq2SeqLM generation for universal compatibility with Transformers v5+.
    """
    global _local_pipeline
    if _local_pipeline is not None:
        return _local_pipeline

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device == "cuda":
            model = model.to("cuda")

        def generate_text(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=250, do_sample=False)
            return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        _local_pipeline = generate_text
        return generate_text
    except Exception as e:
        print(f"Notice: Local HF pipeline could not be loaded ({e}). Using extractive QA fallback.")
        return None


def generate_extractive_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    High-fidelity extractive & synthesis fallback engine.
    Finds the most pertinent sentences in the retrieved context answering the query.
    """
    if not retrieved_chunks:
        return (
            "I could not find information regarding this question in the uploaded document. "
            "Please make sure the document has been fully processed or try rephrasing your query."
        )

    # Keywords from query
    query_words = set(re.findall(r"\w+", query.lower())) - {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "is", "are", "was", "were", "do", "does", "did", "the", "a", "an", "and", "or", "in", "of", "to", "for"
    }

    scored_sentences = []
    for chunk in retrieved_chunks:
        page = chunk.get("page", 1)
        content = chunk.get("content", "")
        # Split into sentences
        sentences = re.split(r"(?<=[.?!])\s+", content)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 20:
                continue
            s_words = set(re.findall(r"\w+", s_clean.lower()))
            overlap = len(query_words.intersection(s_words))
            if overlap > 0:
                scored_sentences.append((overlap, page, s_clean))

    if not scored_sentences:
        # Check if the highest chunk has general relevance
        top_chunk = retrieved_chunks[0]
        if top_chunk.get("score") and top_chunk["score"] < 35.0:
            return (
                "The requested information does not appear to be present in the uploaded document. "
                "(The most relevant section had very low relevance to your query)."
            )
        # Return top summary snippet
        first_text = top_chunk["content"][:400].strip()
        return (
            f"Based on the uploaded document (Page {top_chunk.get('page', 1)}):\n\n"
            f"{first_text}..."
        )

    # Sort sentences by keyword relevance
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_sentences[:4]

    # Synthesize answer
    answer_parts = []
    seen_texts = set()
    for _, page, sent in top_matches:
        if sent not in seen_texts:
            seen_texts.add(sent)
            answer_parts.append(f"• {sent} [Page {page}]")

    return (
        "Based on the uploaded document, here are the key findings relevant to your question:\n\n"
        + "\n".join(answer_parts)
    )


def ask_question(
    vectorstore,
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    provider: str = "local",  # "local", "gemini", or "groq"
    top_k: int = config.TOP_K_RETRIEVAL
) -> Dict[str, Any]:
    """
    RAG Query handler. Searches vectorstore, constructs prompt, calls LLM or fallback,
    and returns answer with citations.
    
    Returns:
        Dict with keys:
          - 'answer': str
          - 'citations': list of dicts with page, snippet, score
          - 'raw_context': str
          - 'model': str
    """
    if not vectorstore:
        return {
            "answer": "No document is currently indexed. Please upload a PDF first.",
            "citations": [],
            "raw_context": "",
            "model": "none"
        }

    # 1. Retrieve relevant chunks
    retrieved = retrieve_relevant_chunks(vectorstore, question, top_k=top_k)
    if not retrieved:
        return {
            "answer": "I could not find any relevant sections in the uploaded document to answer this question.",
            "citations": [],
            "raw_context": "",
            "model": "retriever"
        }

    # Format citations
    citations = []
    for chunk in retrieved:
        page = chunk.get("page", 1)
        score = chunk.get("score")
        content = chunk.get("content", "")
        snippet = content[:180] + ("..." if len(content) > 180 else "")
        citations.append({
            "page": page,
            "source": chunk.get("source", "PDF"),
            "score": score,
            "snippet": snippet,
            "chunk_id": chunk.get("chunk_id", 0)
        })

    formatted_context = format_context_for_prompt(retrieved)

    # Strict RAG system prompt
    prompt = f"""You are an Intelligent AI Document Assistant (DocPilot AI).
Answer the user's question accurately and concisely using ONLY the following context extracted from the uploaded PDF document.

CRITICAL INSTRUCTIONS:
1. Base your answer strictly on the facts provided in the Context below.
2. If the answer cannot be found in or directly inferred from the Context, state clearly: "I cannot find this information in the uploaded document." Do NOT fabricate or assume information.
3. When providing key facts, include page references formatted as [Page X] where X is the source page number.
4. Keep the response well-structured, professional, and clear.

---
CONTEXT:
{formatted_context}
---

USER QUESTION: {question}

DETAILED ANSWER:"""

    # 2. Model Dispatch
    # Priority A: Google Gemini if API key is provided
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    if provider == "gemini" and gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer_text = response.text.strip()
            return {
                "answer": answer_text,
                "citations": citations,
                "raw_context": formatted_context,
                "model": "Google Gemini 2.5 Flash"
            }
        except Exception as e:
            print(f"Gemini generation error: {e}. Falling back.")

    # Priority B: Local Hugging Face Pipeline
    if provider == "local":
        pipe = get_local_hf_pipeline()
        if pipe:
            try:
                # Truncate context if needed for flan-t5 input length limit
                input_prompt = prompt[:2000]
                generated = pipe(input_prompt)
                if generated and len(generated) > 10:
                    return {
                        "answer": generated,
                        "citations": citations,
                        "raw_context": formatted_context,
                        "model": config.DEFAULT_HF_MODEL
                    }
            except Exception as e:
                print(f"Local pipeline execution failed: {e}")

    # Priority C: Semantic & Extractive Synthesis Engine (100% reliable local fallback)
    answer_text = generate_extractive_answer(question, retrieved)
    return {
        "answer": answer_text,
        "citations": citations,
        "raw_context": formatted_context,
        "model": "DocPilot Semantic RAG Engine"
    }
