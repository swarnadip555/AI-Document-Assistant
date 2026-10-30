"""
DocPilot AI - Intelligent AI Document Assistant
A minimal, glassmorphic Streamlit web application for PDF Question Answering (RAG),
Summarization, and MCQ Quiz Generation using LangChain, HuggingFace embeddings, and FAISS.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="DocPilot AI - Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from modules.document_loader import load_pdf
from modules.text_splitter import split_documents
from modules.embeddings import get_embedding_model
from modules.vector_database import create_vectorstore, save_vectorstore, load_vectorstore, clear_vectorstore
from modules.chatbot import ask_question
from modules.summarizer import generate_summary
from modules.mcq_generator import generate_mcqs
from utils.helpers import get_glassmorphism_css, format_file_size, create_sample_pdf

# Apply minimal and glass design CSS
st.markdown(get_glassmorphism_css(), unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chunk_stats" not in st.session_state:
    st.session_state.chunk_stats = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "mcqs" not in st.session_state:
    st.session_state.mcqs = None
if "quiz_submissions" not in st.session_state:
    st.session_state.quiz_submissions = {}
if "selected_page_preview" not in st.session_state:
    st.session_state.selected_page_preview = 1
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "local"


# ---------------------------------------------------------
# Helper: Process Uploaded or Sample PDF
# ---------------------------------------------------------
def process_document(file_path: str, chunk_size: int, chunk_overlap: int):
    """Pipeline: Load PDF -> Chunk Text -> Embed & Store in FAISS"""
    with st.spinner("Processing document: reading pages and extracting text..."):
        try:
            docs, stats = load_pdf(file_path)
            st.session_state.documents = docs
            st.session_state.doc_stats = stats
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return False

    with st.spinner("Dividing document into semantic chunks with metadata..."):
        try:
            chunks, c_stats = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.session_state.chunks = chunks
            st.session_state.chunk_stats = c_stats
        except Exception as e:
            st.error(f"Error splitting text: {str(e)}")
            return False

    with st.spinner("Generating embeddings & building FAISS vector database..."):
        try:
            embeddings = get_embedding_model()
            vstore = create_vectorstore(chunks, embeddings)
            save_vectorstore(vstore)
            st.session_state.vectorstore = vstore
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            return False

    # Reset cached inferences for new document
    st.session_state.summary = None
    st.session_state.mcqs = None
    st.session_state.quiz_submissions = {}
    st.session_state.selected_page_preview = 1

    # Preload initial greeting in chat history
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Hello! I have indexed **{stats['title']}** ({stats['total_pages']} pages, {c_stats['total_chunks']} chunks). Ask me any question, request a summary, or generate quizzes from this document!",
            "citations": []
        })

    return True


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    # Brand Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0 16px 0;">
        <div style="background: linear-gradient(135deg, #0284c7, #38bdf8); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 0 15px rgba(56, 189, 248, 0.5);">⚡</div>
        <div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">DocPilot AI</div>
            <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Intelligent Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Document Upload Section
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Select a standard text-based PDF document to analyze."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if uploaded_file is not None:
            if st.button("🚀 Process PDF", use_container_width=True):
                save_path = config.UPLOADED_PDFS_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if process_document(str(save_path), config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP):
                    st.success("Indexed successfully!")
                    st.rerun()

    with col_btn2:
        if st.button("🧪 Sample PDF", help="Load the demo Q3 Technical Report", use_container_width=True):
            sample_path = config.UPLOADED_PDFS_DIR / "Q3_Technical_Report_Cloud_Security.pdf"
            create_sample_pdf(str(sample_path))
            if process_document(str(sample_path), config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP):
                st.success("Demo document loaded!")
                st.rerun()

    # Clear Document Button
    if st.session_state.vectorstore is not None:
        if st.button("🗑️ Reset / Clear All", use_container_width=True):
            clear_vectorstore()
            st.session_state.documents = None
            st.session_state.doc_stats = None
            st.session_state.chunks = None
            st.session_state.chunk_stats = None
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.session_state.mcqs = None
            st.session_state.quiz_submissions = {}
            st.rerun()

    st.markdown("---")

    # Settings & Model Config
    st.markdown("### ⚙️ Engine Settings")
    provider_option = st.selectbox(
        "LLM Generation Engine",
        options=["local", "gemini"],
        format_func=lambda x: "Local / Offline Extractive RAG" if x == "local" else "Google Gemini API (Cloud)",
        index=0 if st.session_state.llm_provider == "local" else 1
    )
    st.session_state.llm_provider = provider_option

    if provider_option == "gemini":
        gemini_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Google Gemini API key or set GEMINI_API_KEY env var."
        )
        st.session_state.api_key = gemini_input

    # Workspace status widget at bottom of sidebar (matching image)
    st.markdown("""
    <div style="margin-top: 2rem; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
        <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Multi-tenant workspace</div>
        <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin: 4px 0;">⚡ Alpha Organization</div>
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            <div style="width: 28px; height: 28px; border-radius: 50%; background: #0284c7; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">SJ</div>
            <div>
                <div style="font-size: 0.8rem; font-weight: 600; color: #f1f5f9;">Sarah J.</div>
                <div style="font-size: 0.7rem; color: #64748b;">Admin & Auditor</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Top Header Bar (Matching Image 2)
# ---------------------------------------------------------
active_title = st.session_state.doc_stats["title"] if st.session_state.doc_stats else "No Document Loaded"
active_pages = st.session_state.doc_stats["total_pages"] if st.session_state.doc_stats else 0
active_status = "Indexed" if st.session_state.vectorstore else "Awaiting PDF"
status_class = "status-badge" if st.session_state.vectorstore else "status-badge indexing"

st.markdown(f"""
<div class="top-header-bar">
    <div>
        <div class="doc-title-badge">
            <span style="color: #38bdf8;">←</span>
            <span>{active_title}</span>
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
            Page {st.session_state.selected_page_preview} of {active_pages} | Doc ID: DP-{abs(hash(active_title)) % 100000} | Status: <span class="{status_class}">● {active_status}</span>
        </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="workspace-pill">Workspace: Alpha</span>
        <span class="workspace-pill">Project: Security Audit Q3</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main App Layout: Two Columns (Doc Preview/Stats + Document Intelligence)
# ---------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="medium")


# =========================================================
# LEFT COLUMN: Document Viewer & Processing Stats
# =========================================================
with left_col:
    # Tab selector for Left Column: Document Reader vs Vector Store Chunks
    viewer_tab1, viewer_tab2 = st.tabs(["📄 Document View", "🧩 Vector Store Chunks"])

    with viewer_tab1:
        if st.session_state.documents is not None:
            total_pages = len(st.session_state.documents)
            selected_page = st.slider(
                "Page Navigation",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.selected_page_preview,
                key="doc_page_slider"
            )
            st.session_state.selected_page_preview = selected_page

            page_doc = next((d for d in st.session_state.documents if d.metadata.get("page") == selected_page), st.session_state.documents[0])

            st.markdown(f"""
            <div class="glass-card" style="min-height: 280px; max-height: 420px; overflow-y: auto;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 8px; margin-bottom: 12px;">
                    <span style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">PAGE {selected_page} OF {total_pages}</span>
                    <span style="font-size: 0.78rem; color: #94a3b8;">{page_doc.metadata.get('char_count', 0)} Characters</span>
                </div>
                <div style="white-space: pre-wrap; font-size: 0.86rem; line-height: 1.6; color: #cbd5e1; font-family: 'JetBrains Mono', monospace;">
{page_doc.page_content}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 1.75rem 1.25rem 1rem 1.25rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📂</div>
                <div style="font-weight: 700; font-size: 1.15rem; color: #f8fafc;">No Document Uploaded</div>
                <div style="font-size: 0.85rem; color: #94a3b8; max-width: 360px; margin: 0.3rem auto 1rem auto;">
                    Upload a PDF below, or click <b>"Load Sample PDF"</b> to instantly test with the Cloud Security Report.
                </div>
            </div>
            """, unsafe_allow_html=True)

            main_file = st.file_uploader(
                "Upload PDF Document",
                type=["pdf"],
                key="main_tab_pdf_uploader",
                help="Select a PDF to process"
            )

            col_up1, col_up2 = st.columns(2)
            with col_up1:
                if main_file is not None:
                    if st.button("🚀 Process Uploaded PDF", key="btn_main_proc", use_container_width=True):
                        save_p = config.UPLOADED_PDFS_DIR / main_file.name
                        with open(save_p, "wb") as f:
                            f.write(main_file.getbuffer())
                        if process_document(str(save_p), config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP):
                            st.success("Indexed successfully!")
                            st.rerun()

            with col_up2:
                if st.button("🧪 Load Sample PDF", key="btn_main_sample", use_container_width=True):
                    sample_p = config.UPLOADED_PDFS_DIR / "Q3_Technical_Report_Cloud_Security.pdf"
                    create_sample_pdf(str(sample_p))
                    if process_document(str(sample_p), config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP):
                        st.success("Sample report loaded!")
                        st.rerun()

    with viewer_tab2:
        if st.session_state.chunks is not None:
            st.markdown(f"<span style='font-size: 0.85rem; color: #94a3b8;'>Displaying {len(st.session_state.chunks)} chunks stored in FAISS</span>", unsafe_allow_html=True)
            chunk_idx = st.selectbox(
                "Select Chunk to Inspect",
                options=range(len(st.session_state.chunks)),
                format_func=lambda i: f"Chunk #{i+1} (Page {st.session_state.chunks[i].metadata.get('page', '?')}, {st.session_state.chunks[i].metadata.get('chunk_char_length', 0)} chars)"
            )
            ch = st.session_state.chunks[chunk_idx]
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 600; margin-bottom: 6px;">
                    Source: {ch.metadata.get('source')} | Page: {ch.metadata.get('page')} | Tokens: ~{ch.metadata.get('estimated_tokens')}
                </div>
                <div style="white-space: pre-wrap; font-size: 0.82rem; color: #e2e8f0; font-family: monospace;">
{ch.page_content}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No chunks available. Upload a document to view vector store chunks.")

    # Glass Card 1: Document Metadata
    size_str = format_file_size(st.session_state.doc_stats["file_size"]) if st.session_state.doc_stats else "—"
    pages_str = str(st.session_state.doc_stats["total_pages"]) if st.session_state.doc_stats else "—"
    status_str = "Indexed" if st.session_state.vectorstore else "Not Loaded"

    st.markdown(f"""
    <div class="glass-card">
        <div style="font-size: 0.85rem; font-weight: 700; color: #f1f5f9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
            Document Metadata
        </div>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Size</div>
                <div class="metric-value">{size_str}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Type</div>
                <div class="metric-value">PDF</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Pages</div>
                <div class="metric-value">{pages_str}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Status</div>
                <div class="metric-value" style="color: #4ade80; font-size: 1rem;">{status_str}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Glass Card 2: Document Processing Stats (with mini sparkline visuals matching Image 2)
    chunks_str = f"{st.session_state.chunk_stats['total_chunks']} Chunks" if st.session_state.chunk_stats else "—"
    pages_processed_str = f"{st.session_state.doc_stats['pages_with_text']}/{st.session_state.doc_stats['total_pages']}" if st.session_state.doc_stats else "—"
    vector_status_str = "Complete (FAISS)" if st.session_state.vectorstore else "Pending"

    st.markdown(f"""
    <div class="glass-card">
        <div style="font-size: 0.85rem; font-weight: 700; color: #f1f5f9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
            Document Processing Stats
        </div>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Pages Processed</div>
                <div class="metric-value" style="font-size: 1rem;">{pages_processed_str}</div>
                <div class="sparkline-container">
                    <div class="sparkline-bar" style="height: 45%;"></div>
                    <div class="sparkline-bar" style="height: 70%;"></div>
                    <div class="sparkline-bar" style="height: 60%;"></div>
                    <div class="sparkline-bar" style="height: 95%;"></div>
                    <div class="sparkline-bar" style="height: 100%;"></div>
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Chunk Extraction</div>
                <div class="metric-value" style="font-size: 1rem;">{chunks_str}</div>
                <div class="sparkline-container">
                    <div class="sparkline-bar" style="height: 50%;"></div>
                    <div class="sparkline-bar" style="height: 85%;"></div>
                    <div class="sparkline-bar" style="height: 75%;"></div>
                    <div class="sparkline-bar" style="height: 90%;"></div>
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Vector Status</div>
                <div class="metric-value" style="font-size: 0.95rem; color: #38bdf8;">{vector_status_str}</div>
                <div class="sparkline-container">
                    <div class="sparkline-bar" style="height: 100%; background: #38bdf8;"></div>
                    <div class="sparkline-bar" style="height: 100%; background: #38bdf8;"></div>
                    <div class="sparkline-bar" style="height: 100%; background: #38bdf8;"></div>
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Semantic Index</div>
                <div class="metric-value" style="font-size: 0.95rem; color: #4ade80;">Ready</div>
                <div class="sparkline-container">
                    <div class="sparkline-bar" style="height: 80%; background: #4ade80;"></div>
                    <div class="sparkline-bar" style="height: 100%; background: #4ade80;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# RIGHT COLUMN: Document Intelligence (Chat, Summary, MCQs)
# =========================================================
with right_col:
    right_tab1, right_tab2, right_tab3 = st.tabs([
        "💬 Document Intelligence",
        "📑 PDF Summary",
        "🎯 MCQ & Quiz Generator"
    ])

    # -----------------------------------------------------
    # TAB 1: Document Intelligence (RAG Chat Q&A)
    # -----------------------------------------------------
    with right_tab1:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">Document Intelligence Q&A</span>
            <span style="font-size: 0.75rem; color: #38bdf8; font-family: monospace;">FAISS Dense Retrieval</span>
        </div>
        """, unsafe_allow_html=True)

        # Quick question prompt suggestions
        quick_cols = st.columns(3)
        with quick_cols[0]:
            if st.button("💡 Recommendations?", use_container_width=True):
                st.session_state.quick_q = "What are the primary recommendations for optimizing cloud infrastructure security according to this report?"
        with quick_cols[1]:
            if st.button("🔐 MFA & Identity?", use_container_width=True):
                st.session_state.quick_q = "What multi-factor authentication and identity controls are enforced?"
        with quick_cols[2]:
            if st.button("🛡️ Threat Audits?", use_container_width=True):
                st.session_state.quick_q = "What quarterly security audits and threat detection protocols are described?"

        # Chat message display area
        chat_container = st.container(height=420)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align: center; color: #64748b; padding: 2rem;">
                    Ask questions about the uploaded PDF. Answers are generated using RAG with exact page citations.
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-user-bubble">
                        <div style="font-size: 0.72rem; color: #94a3b8; margin-bottom: 4px; font-weight: 600;">YOU</div>
                        <div>{msg['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    citations_html = ""
                    if msg.get("citations"):
                        pills = []
                        for c in msg["citations"]:
                            pills.append(f"<span class='citation-card'>[Source: Page {c.get('page')}]</span>")
                        citations_html = f"""
                        <div style="margin-top: 10px; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Citations</div>
                            {' '.join(pills)}
                        </div>
                        """
                    
                    st.markdown(f"""
                    <div class="chat-assistant-bubble">
                        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
                            <span style="font-size: 14px;">⚡</span>
                            <span style="font-size: 0.78rem; font-weight: 700; color: #38bdf8;">DocPilot Assistant</span>
                        </div>
                        <div style="line-height: 1.55;">{msg['content']}</div>
                        {citations_html}
                    </div>
                    """, unsafe_allow_html=True)

        # Question Input Bar
        default_val = st.session_state.pop("quick_q", "")
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question about the document:",
                value=default_val,
                placeholder="Type your question (e.g. 'What are the main security recommendations?')...",
                label_visibility="collapsed"
            )
            col_submit, col_clear = st.columns([4, 1])
            with col_submit:
                submit_button = st.form_submit_button("Ask Question 🚀", use_container_width=True)
            with col_clear:
                clear_chat = st.form_submit_button("Clear Chat", use_container_width=True)

        if clear_chat:
            st.session_state.chat_history = []
            st.rerun()

        if submit_button and user_input.strip():
            if st.session_state.vectorstore is None:
                st.warning("Please upload and process a PDF document first!")
            else:
                # Add user question
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                with st.spinner("Searching document & synthesizing answer..."):
                    response = ask_question(
                        vectorstore=st.session_state.vectorstore,
                        question=user_input,
                        api_key=st.session_state.api_key,
                        provider=st.session_state.llm_provider,
                        top_k=config.TOP_K_RETRIEVAL
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "citations": response["citations"]
                    })
                st.rerun()

    # -----------------------------------------------------
    # TAB 2: PDF Summary
    # -----------------------------------------------------
    with right_tab2:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">Document Summary & Key Takeaways</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.documents is None:
            st.info("Upload and index a PDF document to generate an executive summary.")
        else:
            if st.button("✨ Generate Document Summary", use_container_width=True):
                with st.spinner("Analyzing document structure & summarizing..."):
                    summary_data = generate_summary(
                        st.session_state.documents,
                        api_key=st.session_state.api_key,
                        provider=st.session_state.llm_provider
                    )
                    st.session_state.summary = summary_data

            if st.session_state.summary is not None:
                s_data = st.session_state.summary
                
                # If raw markdown returned from Gemini API
                if "raw_markdown" in s_data:
                    st.markdown(f"""
                    <div class="glass-card">
                        {s_data['raw_markdown']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 8px;">
                            Executive Summary
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.6; color: #e2e8f0; margin-bottom: 1rem;">
                            {s_data.get('executive_summary', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if s_data.get("key_takeaways"):
                        st.markdown("""
                        <div class="glass-card">
                            <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 8px;">
                                Key Takeaways & Evidence
                            </div>
                        """, unsafe_allow_html=True)
                        for point in s_data["key_takeaways"]:
                            st.markdown(f"- {point}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    if s_data.get("sections_covered"):
                        st.markdown("""
                        <div class="glass-card">
                            <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 8px;">
                                Sections & Topics Covered
                            </div>
                        """, unsafe_allow_html=True)
                        for sec in s_data["sections_covered"]:
                            st.markdown(f"- 📌 **{sec}**")
                        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # TAB 3: MCQ & Quiz Generator
    # -----------------------------------------------------
    with right_tab3:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">Interactive MCQ Quiz Generator</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.documents is None:
            st.info("Upload and index a PDF document to generate multiple choice quiz questions.")
        else:
            col_q1, col_q2 = st.columns([3, 1])
            with col_q1:
                num_mcqs = st.slider("Number of Questions", min_value=2, max_value=8, value=4)
            with col_q2:
                gen_mcqs_btn = st.button("🎲 Generate MCQs", use_container_width=True)

            if gen_mcqs_btn:
                with st.spinner("Formulating multiple-choice questions with citations..."):
                    mcq_list = generate_mcqs(
                        st.session_state.documents,
                        num_questions=num_mcqs,
                        api_key=st.session_state.api_key,
                        provider=st.session_state.llm_provider
                    )
                    st.session_state.mcqs = mcq_list
                    st.session_state.quiz_submissions = {}

            if st.session_state.mcqs is not None:
                st.markdown(f"**Quiz: {len(st.session_state.mcqs)} Questions Generated**")
                
                total_answered = len(st.session_state.quiz_submissions)
                correct_count = sum(1 for q_idx, ans in st.session_state.quiz_submissions.items() if ans == st.session_state.mcqs[q_idx]["correct_answer"])

                if total_answered > 0:
                    st.markdown(f"""
                    <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem;">
                        📊 <b>Score:</b> {correct_count} / {total_answered} answered ({round((correct_count/total_answered)*100)}%)
                    </div>
                    """, unsafe_allow_html=True)

                for idx, mcq in enumerate(st.session_state.mcqs):
                    st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 12px;">
                        <div style="font-weight: 700; color: #f1f5f9; font-size: 0.92rem; margin-bottom: 10px;">
                            Q{idx+1}. {mcq['question']}
                        </div>
                    """, unsafe_allow_html=True)

                    options_dict = mcq.get("options", {})
                    radio_choices = [f"{k}: {v}" for k, v in options_dict.items()]
                    
                    user_selection = st.radio(
                        label=f"Options for Q{idx+1}",
                        options=radio_choices,
                        key=f"mcq_{idx}",
                        label_visibility="collapsed"
                    )

                    selected_letter = user_selection.split(":")[0].strip() if user_selection else None

                    col_ans, col_exp = st.columns([1, 3])
                    with col_ans:
                        if st.button(f"Submit Q{idx+1}", key=f"btn_q_{idx}"):
                            st.session_state.quiz_submissions[idx] = selected_letter

                    if idx in st.session_state.quiz_submissions:
                        user_ans = st.session_state.quiz_submissions[idx]
                        is_correct = (user_ans == mcq["correct_answer"])

                        if is_correct:
                            st.markdown(f"""
                            <div style="color: #4ade80; font-weight: 600; font-size: 0.85rem; margin-top: 8px;">
                                ✅ Correct! Option {mcq['correct_answer']} is the right answer.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="color: #f87171; font-weight: 600; font-size: 0.85rem; margin-top: 8px;">
                                ❌ Incorrect. You selected {user_ans}. Correct is {mcq['correct_answer']}.
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="background: rgba(255, 255, 255, 0.04); border-left: 3px solid #38bdf8; padding: 6px 10px; margin-top: 6px; font-size: 0.8rem; color: #94a3b8;">
                            <b>Explanation & Source:</b> {mcq.get('explanation', '')} [Page {mcq.get('page', 1)}]
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

