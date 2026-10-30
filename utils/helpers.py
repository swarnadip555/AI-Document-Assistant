"""
Helper Utilities Module
Provides UI CSS injection, file size formatting, sample PDF creation,
and layout styling for the minimal glass design interface.
"""

import os
from pathlib import Path
from typing import Dict, Any
import pypdf


def format_file_size(size_bytes: int) -> str:
    """Format bytes into human-readable size string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def get_glassmorphism_css() -> str:
    """
    Returns custom CSS for the sleek, minimal, glassmorphic UI matching DocPilot AI.
    Features:
      - Deep slate / cyber dark background (#080c14, #0b1120)
      - Frosted glass cards with backdrop-filter blur and cyan neon borders
      - Glowing pill badges, modern tabs, and responsive layout
    """
    return """
    <style>
    /* Global Background and Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Glass Gradient */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 75%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
                    #080c14;
        color: #e2e8f0;
    }

    /* Hide default Streamlit footer and menu, preserve header for sidebar expand button */
    #MainMenu, footer {
        visibility: hidden;
        display: none !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.8rem !important;
        z-index: 99999 !important;
    }

    /* Make sidebar expand/collapse button prominent and always visible */
    [data-testid="collapsedControl"], button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #38bdf8 !important;
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 8px !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="collapsedControl"]:hover, button[data-testid="stSidebarCollapseButton"]:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.5) !important;
        transform: scale(1.05);
    }

    /* Sidebar Glass Design */
    section[data-testid="stSidebar"] {
        background-color: rgba(11, 17, 32, 0.85) !important;
        backdrop-filter: blur(18px) saturate(180%);
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }
    
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.85rem;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(56, 189, 248, 0.14);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.28);
    }

    /* Header Bar */
    .top-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1.25rem;
    }

    .doc-title-badge {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-weight: 600;
        font-size: 1.05rem;
        color: #f8fafc;
    }

    .workspace-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38bdf8;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-badge.indexing {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.3);
        color: #fbbf24;
    }

    /* Metric Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin-top: 0.5rem;
    }

    .metric-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.75rem;
        text-align: left;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    /* Chat Messages Glass */
    .chat-user-bubble {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px 14px 2px 14px;
        padding: 0.85rem 1.1rem;
        margin-left: 2rem;
        margin-bottom: 0.85rem;
        color: #f8fafc;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .chat-assistant-bubble {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px 14px 14px 2px;
        padding: 1rem 1.25rem;
        margin-right: 2rem;
        margin-bottom: 1rem;
        color: #e2e8f0;
        box-shadow: 0 8px 25px rgba(2, 132, 199, 0.08);
    }

    /* Citation Badges */
    .citation-card {
        display: inline-block;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 8px;
        padding: 0.35rem 0.65rem;
        margin-top: 0.5rem;
        margin-right: 0.5rem;
        font-size: 0.78rem;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
    }

    .citation-card:hover {
        background: rgba(56, 189, 248, 0.16);
    }

    /* Sparkline Stat Bars */
    .sparkline-container {
        display: flex;
        align-items: flex-end;
        gap: 4px;
        height: 28px;
        margin-top: 6px;
    }

    .sparkline-bar {
        flex: 1;
        background: linear-gradient(180deg, #38bdf8 0%, #0284c7 100%);
        border-radius: 3px 3px 0 0;
        opacity: 0.85;
    }

    /* Buttons Styling */
    .stButton>button {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.8) 0%, rgba(2, 132, 199, 0.9) 100%);
        color: white;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.25);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.9) 0%, rgba(14, 165, 233, 1) 100%);
        border-color: rgba(56, 189, 248, 0.6);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
        transform: translateY(-1px);
    }

    /* Radio & Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        font-weight: 600;
    }
    </style>
    """


def create_sample_pdf(output_path: str) -> str:
    """
    Generate a realistic, text-rich multi-page sample PDF matching the DocPilot AI screenshot:
    'Q3 Technical Report: Cloud Security and Compliance'.
    This enables immediate instant testing right out of the box!
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use pypdf to generate a valid PDF with text streams
    writer = pypdf.PdfWriter()

    pages_content = [
        # Page 1
        (
            "Q3 Technical Report: Cloud Security and Compliance\n"
            "Document ID: DP-98124 | Classification: Confidential | Date: 2026-10-15\n\n"
            "1. EXECUTIVE SUMMARY & INFRASTRUCTURE OVERVIEW\n"
            "DocPilot AI cloud infrastructure is deployed across multi-region Kubernetes clusters on hybrid cloud. "
            "Our primary objective in Q3 was hardening cloud assets, transitioning to Zero Trust Architecture (ZTA), "
            "and ensuring compliance across all production workloads.\n\n"
            "Key Architectural Principles:\n"
            "- Principle of Least Privilege (PoLP): All services authenticate via short-lived OIDC tokens.\n"
            "- Microsegmentation: Virtual Private Clouds (VPCs) are partitioned using strict network security groups.\n"
            "- End-to-End Encryption: TLS 1.3 is enforced in-transit, and AES-256 GCM is applied at-rest with customer-managed keys.\n\n"
            "Cloud Security Posture Management (CSPM) monitors 1,420 cloud instances continuously, achieving a 99.98% audit pass rate."
        ),
        # Page 2
        (
            "Q3 Technical Report: Cloud Security and Compliance (Page 2)\n"
            "Document ID: DP-98124 | Access Control & Authentication\n\n"
            "2. MULTI-FACTOR AUTHENTICATION & IDENTITY GOVERNANCE\n"
            "Identity is the primary perimeter in modern cloud native architectures. "
            "During Q3, we completed mandatory rollout of hardware-backed FIDO2 / WebAuthn Multi-Factor Authentication (MFA).\n\n"
            "Key Protocols and Controls:\n"
            "- MFA Enforcement: 100% of administrative and privileged accounts enforce hardware token MFA.\n"
            "- Role-Based Access Control (RBAC): Privileged access requires Just-In-Time (JIT) approval with automatic session expiry after 2 hours.\n"
            "- Single Sign-On (SSO): Okta and Azure AD identity providers are integrated with automated SCIM provisioning.\n"
            "- Service-to-Service Authentication: Mutual TLS (mTLS) with SPIFFE/SPIRE certificates verifies every RPC transaction.\n\n"
            "Any access anomaly triggers immediate token revocation and sends real-time PagerDuty alerts to SecOps."
        ),
        # Page 3
        (
            "Q3 Technical Report: Cloud Security and Compliance (Page 3)\n"
            "Document ID: DP-98124 | Vulnerability Management & Threat Detection\n\n"
            "3. THREAT DETECTION & CONTINUOUS VULNERABILITY AUDITING\n"
            "Our automated scanning pipeline executes daily static code analysis (SAST) and dynamic scanning (DAST) on all CI/CD pipelines.\n\n"
            "Key Audit Metrics:\n"
            "- Quarterly Security Audits: Conducted quarterly vulnerability assessments across container registries and host kernels.\n"
            "- Real-Time Threat Detection: eBPF-based runtime sensors detect unexpected process spawning and unauthorized socket binding within 45 milliseconds.\n"
            "- Automated Patch Remediation: Critical CVEs are automatically patched and redeployed through blue-green pipelines within 24 hours.\n"
            "- Incident Response Drills: Simulated red-team phishing and ransomware scenarios yielded an average response time of 6.2 minutes.\n\n"
            "Zero critical severity CVEs remained unaddressed past the 48-hour SLA during Q3."
        ),
        # Page 4
        (
            "Q3 Technical Report: Cloud Security and Compliance (Page 4)\n"
            "Document ID: DP-98124 | Compliance & Strategic Recommendations\n\n"
            "4. COMPLIANCE CERTIFICATIONS & RECOMMENDATIONS\n"
            "Independent third-party audits concluded with zero material findings, successfully renewing key compliance frameworks:\n"
            "- SOC 2 Type II: Clean report covering Security, Availability, and Confidentiality trust criteria.\n"
            "- ISO/IEC 27001:2022: Full certification across engineering, IT operations, and product development.\n"
            "- HIPAA & GDPR: Full alignment with data privacy, sovereignty, and cryptographic retention regulations.\n\n"
            "Primary Recommendations for Q4:\n"
            "1. Enhance multi-factor authentication (MFA) protocols for all administrative and database access.\n"
            "2. Conduct quarterly security audits and vulnerability assessments across external API endpoints.\n"
            "3. Implement real-time threat detection and automated response orchestration using behavioral ML models.\n"
            "4. Expand automated container image signing with Cosign and Sigstore in production registries."
        )
    ]

    for page_text in pages_content:
        # Create a page with standard PDF text stream
        page = pypdf.PageObject.create_blank_page(width=612, height=792)
        
        # Build text stream
        lines = page_text.split("\n")
        # PDF operations: BT (Begin Text), Tf (Font), Td (Position), Tj (Show Text), ET (End Text)
        content_stream_parts = ["BT", "/F1 11 Tf", "50 740 Td", "14 TL"]
        
        for idx, line in enumerate(lines):
            safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if idx == 0:
                content_stream_parts.append(f"({safe_line}) Tj")
            else:
                content_stream_parts.append(f"T* ({safe_line}) Tj")
                
        content_stream_parts.append("ET")
        raw_stream = "\n".join(content_stream_parts).encode("latin-1", "replace")
        
        # Add default font resource
        font_dict = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/Type"): pypdf.generic.NameObject("/Font"),
            pypdf.generic.NameObject("/Subtype"): pypdf.generic.NameObject("/Type1"),
            pypdf.generic.NameObject("/BaseFont"): pypdf.generic.NameObject("/Helvetica"),
        })
        resources_dict = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/Font"): pypdf.generic.DictionaryObject({
                pypdf.generic.NameObject("/F1"): font_dict
            })
        })
        page[pypdf.generic.NameObject("/Resources")] = resources_dict

        # Attach content stream
        stream_obj = pypdf.generic.DecodedStreamObject()
        stream_obj.set_data(raw_stream)
        page[pypdf.generic.NameObject("/Contents")] = stream_obj

        writer.add_page(page)

    writer.add_metadata({
        "/Title": "Q3 Technical Report: Cloud Security and Compliance",
        "/Author": "DocPilot AI Security Operations",
        "/Subject": "Cloud Security, Compliance, and Audit Architecture",
        "/Creator": "DocPilot AI Generator"
    })

    with open(str(path), "wb") as f:
        writer.write(f)

    return str(path)

