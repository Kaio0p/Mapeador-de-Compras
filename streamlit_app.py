# -*- coding: utf-8 -*-
"""
streamlit_app.py — Gerador de Mapa de Compras (Arquitetura Multi-Agente)
=========================================================================
Design: Apple HIG · Liquid Glass · SF Pro system stack

Fluxo de agentes:
  PDF nativo    → Gemini (extract_items_from_text    — texto nativo)
  PDF escaneado → Gemini (extract_items_from_images  — OCR multi-chave)
  Imagem direta → Gemini (extract_items_from_jpeg/png_images)
  Normalização  → Cohere (normalize_and_match        — command-r-plus)
  Auditoria     → Gemini (audit_purchase_map          — cross-reference)
  Revisão       → Humano (st.data_editor              — Human-in-the-Loop)
  Download      → Excel  (generate_excel)
"""
import sys
import base64
import streamlit as st
import pandas as pd
from datetime import date

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Importações dos módulos ───────────────────────────────────────────────────
from modules.pdf_extractor import (
    extract_text_from_pdf, extract_images_from_pdf, get_pdf_page_count,
    detect_image_mime,
)
from modules.gemini_processor import (
    configure as configure_gemini,
    extract_items_from_images,
    extract_items_from_jpeg_images,
    audit_purchase_map,
    ALLOWED_UNITS,
)
from modules.cohere_processor import (
    normalize_and_match,
)
from modules.llm_manager import (
    get_system_status,
    configure_gemini_with_key,
    get_random_gemini_key,
    get_cohere_client,
)
from modules.excel_generator import generate_excel
from modules.preferences_manager import (
    detect_corrections, merge_corrections, build_prompt_context,
    load_from_supabase, save_to_supabase,
    load_catalog_from_supabase,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mapa de Compras · EBD",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Apple Liquid Glass + Geist + DeviceDesk Design System ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500&display=swap');

/* ══ DESIGN TOKENS ══ */
:root {
  --bg: #f2f2f7; --surf: #ffffff; --surf2: #f2f2f7; --surf3: #e5e5ea;
  --ln: rgba(0,0,0,0.08); --ln2: rgba(0,0,0,0.14);
  --tx: #1c1c1e; --tx2: #6c6c70; --tx3: #aeaeb2;
  --ac: #007aff; --ac-h: #0062cc; --ac-bg: rgba(0,122,255,0.09); --ac-bd: rgba(0,122,255,0.20);
  --gr: #34c759; --gr-bg: rgba(52,199,89,0.10);
  --am: #ff9500; --am-bg: rgba(255,149,0,0.10);
  --rd: #ff3b30; --rd-bg: rgba(255,59,48,0.10);
  --r: 18px; --r-sm: 12px; --r-pill: 999px;
  --sh: 0 2px 5px rgba(0,0,0,0.02), 0 8px 16px rgba(0,0,0,0.03), 0 16px 32px rgba(0,0,0,0.04);
  --sh-lg: 0 4px 10px rgba(0,0,0,0.03), 0 12px 28px rgba(0,0,0,0.05), 0 24px 48px rgba(0,0,0,0.06);
  --ease: cubic-bezier(0.32, 0.72, 0, 1);
  --spring: cubic-bezier(0.175, 0.885, 0.32, 1.15);
  --fast: cubic-bezier(0.25, 0.1, 0.25, 1);
  --f: 'Geist', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
  --fm: 'Geist Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: var(--f) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ══ SCROLLBAR FINA (estilo DeviceDesk) ══ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surf3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--tx3); }

/* ══ ESCONDE CHROME NATIVO DO STREAMLIT ══ */
/* Só esconde elementos específicos — NÃO usa .stApp > header para evitar esconder sidebar */
[data-testid="stToolbar"]  { visibility: hidden !important; height: 0 !important; overflow: hidden !important; }
#MainMenu                  { visibility: hidden !important; }
footer                     { visibility: hidden !important; height: 0 !important; overflow: hidden !important; }
/* Reduz o padding do header nativo para zero sem esconder o container */
[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    padding: 0 !important;
}

/* ══ BASE ══ */
.stApp { background: var(--bg) !important; min-height: 100vh; }

/* ══ TOPBAR PERSONALIZADA — Sticky glass ══ */
.topbar {
    position: sticky; top: 0; z-index: 200;
    height: 52px;
    display: flex; align-items: center;
    padding: 0 32px; gap: 14px;
    background: rgba(242,242,247,0.0);
    border-bottom: 0.5px solid transparent;
    transition: background 0.35s var(--ease), border-color 0.35s var(--ease),
                backdrop-filter 0.35s var(--ease);
    margin: 0 -2.5rem;
    width: calc(100% + 5rem);
}
.topbar.scrolled {
    background: rgba(242,242,247,0.75);
    backdrop-filter: saturate(200%) blur(24px);
    -webkit-backdrop-filter: saturate(200%) blur(24px);
    border-bottom-color: rgba(0,0,0,0.07);
}
.topbar-title {
    font-size: 0.9rem; font-weight: 600; letter-spacing: -0.02em; color: var(--tx);
    opacity: 0; transform: translateY(-5px);
    transition: opacity 0.28s var(--ease), transform 0.28s var(--ease);
    flex: 1;
}
.topbar.scrolled .topbar-title { opacity: 1; transform: translateY(0); }
.topbar-step-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.8); border: 0.5px solid var(--ln2);
    border-radius: var(--r-pill); padding: 4px 14px;
    font-size: 0.75rem; font-weight: 600; color: var(--tx2);
    opacity: 0; transform: translateY(-4px);
    transition: opacity 0.28s var(--ease), transform 0.28s var(--ease);
}
.topbar.scrolled .topbar-step-pill { opacity: 1; transform: translateY(0); }

/* ══ PAGE TITLE — sempre visível ══ */
.page-title-wrap {
    padding: 8px 0 4px;
    animation: fadeSlideIn 0.45s var(--spring) both;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ══ SIDEBAR — hover slide-in ══ */
/* A sidebar começa colapsada (estado Streamlit) e aparece suavemente ao hover */
[data-testid="stSidebar"] {
    /* Posição e transição suave */
    transform: translateX(0) !important;
    transition: transform 0.32s var(--ease),
                box-shadow 0.32s var(--ease),
                opacity 0.32s var(--ease) !important;
    will-change: transform, box-shadow;

    /* Visual liquid glass */
    background: linear-gradient(160deg,
        rgba(255,255,255,0.90) 0%,
        rgba(255,255,255,0.75) 60%,
        rgba(242,242,247,0.82) 100%) !important;
    backdrop-filter: blur(32px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
    border-right: 0.5px solid rgba(255,255,255,0.72) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.08), 1px 0 0 rgba(0,0,0,0.05) !important;
}

/* Estado colapsado: sidebar recuada e quase invisível */
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(-88%) !important;
    opacity: 0.0 !important;
    box-shadow: none !important;
    pointer-events: none !important;
}

/* Trigger zone — faixa invisível na borda esquerda que ativa o hover */
[data-testid="stSidebar"][aria-expanded="false"]::before {
    content: '' !important;
    position: fixed !important;
    left: 0 !important; top: 0 !important; bottom: 0 !important;
    width: 18px !important;
    z-index: 9999 !important;
    cursor: pointer !important;
    pointer-events: auto !important;
}

/* Hover na trigger zone OU na própria sidebar mostra a sidebar */
[data-testid="stSidebar"][aria-expanded="false"]:hover {
    transform: translateX(0) !important;
    opacity: 1 !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.10), 1px 0 0 rgba(0,0,0,0.05) !important;
    pointer-events: auto !important;
}

[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ══ SIDEBAR — brand header ══ */
.sb-brand {
    display: flex; flex-direction: column; align-items: center;
    padding: 28px 20px 20px;
    border-bottom: 0.5px solid rgba(0,0,0,0.07);
    margin-bottom: 4px;
    gap: 14px;
}
.sb-logo-wrap {
    width: 110px; height: 66px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.sb-logo-wrap img {
    width: 100%; height: 100%; object-fit: contain;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.08));
    transition: opacity 0.3s var(--ease);
}
.sb-brand-text { text-align: center; display: flex; flex-direction: column; gap: 3px; }
.sb-app-name {
    font-size: 0.95rem; font-weight: 600; letter-spacing: -0.025em;
    color: var(--tx); line-height: 1.2;
}
.sb-app-sub {
    font-size: 0.68rem; font-weight: 500; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--tx3);
}

/* ══ SIDEBAR — section labels ══ */
.sb-section-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--tx3);
    padding: 12px 4px 4px; display: block;
}

/* ══ SIDEBAR — status indicator rows ══ */
.sb-status-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 10px; border-radius: 10px;
    margin-bottom: 3px;
    transition: background 0.15s var(--fast);
}
.sb-status-row:hover { background: rgba(0,0,0,0.03); }
.sb-status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    flex-shrink: 0; margin-top: 1px;
}
.sb-status-dot.ok   { background: var(--gr); box-shadow: 0 0 0 3px rgba(52,199,89,0.18); }
.sb-status-dot.err  { background: var(--rd); box-shadow: 0 0 0 3px rgba(255,59,48,0.18); }
.sb-status-dot.warn { background: var(--am); box-shadow: 0 0 0 3px rgba(255,149,0,0.18); }
.sb-status-text { display: flex; flex-direction: column; gap: 1px; }
.sb-status-label { font-size: 0.8rem; font-weight: 500; color: var(--tx); line-height: 1.2; }
.sb-status-sub   { font-size: 0.7rem; color: var(--tx3); }

/* ══ SIDEBAR — preferences pill ══ */
.sb-pref-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,122,255,0.08); border: 0.5px solid rgba(0,122,255,0.18);
    border-radius: var(--r-pill); padding: 4px 12px;
    font-size: 0.75rem; font-weight: 500; color: #0062cc;
    margin: 4px 0 2px;
}

/* ══ SIDEBAR — footer ══ */
.sb-footer {
    padding: 12px 16px 16px;
    border-top: 0.5px solid rgba(0,0,0,0.06);
    margin-top: 8px;
}
.sb-version {
    font-size: 0.68rem; color: var(--tx3);
    font-family: var(--fm);
    letter-spacing: 0.03em;
    text-align: center;
    padding: 6px 0;
}

/* ══ SIDEBAR — Streamlit nativo override ══ */
[data-testid="stSidebar"] h3 {
    font-size: 0.62rem !important; font-weight: 700 !important;
    letter-spacing: 0.09em !important; text-transform: uppercase !important;
    color: var(--tx3) !important; margin: 14px 0 4px !important;
    padding: 0 4px !important;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 0.5px solid rgba(0,0,0,0.07) !important;
    margin: 10px 0 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--tx2) !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size: 0.83rem !important;
}

/* ══ LAYOUT ══ */
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px !important; }

/* ══ TYPOGRAPHY ══ */
h1 {
    font-size: 2rem !important; font-weight: 700 !important;
    letter-spacing: -0.035em !important; color: var(--tx) !important;
    margin-bottom: 0.2rem !important; line-height: 1.12 !important;
}
h2 {
    font-size: 1.35rem !important; font-weight: 600 !important;
    letter-spacing: -0.025em !important; color: var(--tx) !important;
}
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #3A3A3C !important; }
p, label, .stMarkdown, .stCaption {
    color: #3A3A3C !important; font-size: 0.9rem !important; line-height: 1.55 !important;
}
.stCaption, small { color: var(--tx3) !important; font-size: 0.78rem !important; }

/* ══ INPUTS ══ */
.stTextInput input, .stDateInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.92) !important;
    border: 0.5px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important; padding: 10px 14px !important;
    font-size: 0.88rem !important; font-family: var(--f) !important;
    color: var(--tx) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--ac) !important;
    box-shadow: 0 0 0 3px var(--ac-bg), 0 1px 3px rgba(0,0,0,0.04) !important;
    outline: none !important;
}

/* ══ BUTTONS ══ */
.stButton > button {
    background: rgba(255,255,255,0.92) !important; color: var(--ac) !important;
    border: 0.5px solid var(--ac-bd) !important; border-radius: var(--r-pill) !important;
    padding: 9px 22px !important; font-size: 0.88rem !important; font-weight: 500 !important;
    font-family: var(--f) !important; letter-spacing: -0.01em !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important;
    transition: all 0.18s var(--fast) !important; cursor: pointer !important;
}
.stButton > button:hover {
    background: var(--ac-bg) !important; border-color: rgba(0,122,255,0.4) !important;
    transform: translateY(-1px) !important; box-shadow: 0 4px 14px rgba(0,122,255,0.18) !important;
}
.stButton > button:active { transform: scale(0.97) !important; }
.stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, #1A8DFF 0%, var(--ac-h) 100%) !important;
    color: #fff !important; border: none !important;
    box-shadow: 0 2px 12px rgba(0,122,255,0.36) !important;
}
.stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(180deg, #1E96FF 0%, #0068D6 100%) !important;
    box-shadow: 0 4px 20px rgba(0,122,255,0.48) !important; transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: linear-gradient(180deg, #34C759 0%, #28A745 100%) !important;
    color: #fff !important; border: none !important; border-radius: var(--r-pill) !important;
    padding: 10px 26px !important; font-weight: 600 !important; font-size: 0.9rem !important;
    box-shadow: 0 2px 14px rgba(40,167,69,0.36) !important; transition: all 0.18s var(--ease) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important; box-shadow: 0 6px 22px rgba(40,167,69,0.44) !important;
}

/* ══ FILE UPLOADER (default — passo 1 tem override próprio) ══ */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.62) !important; backdrop-filter: blur(16px) !important;
    border: 1.5px dashed rgba(0,122,255,0.22) !important; border-radius: 14px !important;
    padding: 1.2rem !important; transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,122,255,0.50) !important; background: rgba(0,122,255,0.02) !important;
}

/* ══ ALERTS ══ */
.stAlert { border-radius: 12px !important; border: none !important; }
div[data-testid="stAlert"] {
    background: var(--ac-bg) !important; border-left: 3px solid var(--ac) !important;
    border-radius: 12px !important;
}

/* ══ EXPANDERS ══ */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.78) !important; border-radius: 10px !important;
    border: 0.5px solid rgba(0,0,0,0.08) !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 12px 16px !important;
    transition: background 0.18s var(--fast) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.streamlit-expanderHeader:hover { background: rgba(255,255,255,0.94) !important; }
.streamlit-expanderContent {
    background: rgba(255,255,255,0.52) !important;
    border: 0.5px solid rgba(0,0,0,0.06) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 16px !important;
}

/* ══ PROGRESS ══ */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--ac), var(--gr)) !important;
    border-radius: 999px !important;
}
.stProgress > div { background: rgba(0,0,0,0.06) !important; border-radius: 999px !important; height: 4px !important; }

/* ══ DATA FRAME / IFRAME ══ */
[data-testid="stDataFrame"], iframe {
    border-radius: 12px !important; overflow: hidden !important;
    box-shadow: var(--sh) !important; border: 0.5px solid rgba(0,0,0,0.07) !important;
}
.stSpinner > div { color: var(--ac) !important; }
[data-baseweb="select"] > div {
    border-radius: 10px !important; border-color: rgba(0,0,0,0.1) !important;
    background: rgba(255,255,255,0.88) !important;
}

/* ══ COMPONENTES CUSTOMIZADOS ══ */
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.88) 0%, rgba(255,255,255,0.54) 100%);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 0.5px solid rgba(255,255,255,0.92);
    border-radius: var(--r); padding: 20px 24px; margin-bottom: 12px;
    box-shadow: var(--sh-lg), 0 1px 0 rgba(255,255,255,0.9) inset;
    transition: box-shadow 0.2s var(--ease);
}
.glass-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.09), 0 1px 0 rgba(255,255,255,0.9) inset;
}
.glass-card .supplier-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--tx3); margin-bottom: 4px;
}
.glass-card .supplier-name { font-size: 1.05rem; font-weight: 600; color: var(--tx); letter-spacing: -0.02em; }

.step-track {
    display: flex; align-items: center;
    background: linear-gradient(135deg, rgba(255,255,255,0.82) 0%, rgba(255,255,255,0.58) 100%);
    backdrop-filter: blur(20px) saturate(180%);
    border: 0.5px solid rgba(255,255,255,0.9); border-radius: 14px;
    padding: 12px 20px; margin-bottom: 1.8rem;
    box-shadow: var(--sh);
    gap: 0;
}
.step-item { display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0; }
.step-sep {
    flex: 0 0 auto; display: flex; align-items: center; padding: 0 8px;
}
.step-sep::after {
    content: ''; display: block; width: 22px; height: 1px; background: var(--ln2);
    border-radius: 1px;
}
.step-dot {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; flex-shrink: 0; transition: all 0.3s var(--spring);
}
.step-dot.done   { background: var(--gr); color: #fff; }
.step-dot.active {
    background: var(--ac); color: #fff;
    box-shadow: 0 0 0 4px rgba(0,122,255,0.18);
}
.step-dot.idle   { background: rgba(0,0,0,0.07); color: var(--tx3); }
.step-text { font-size: 0.82rem; line-height: 1.2; min-width: 0; }
.step-text .num { font-weight: 600; color: var(--tx); letter-spacing: -0.01em; white-space: nowrap; }
.step-text .sub { font-size: 0.72rem; color: var(--tx3); white-space: nowrap; }

.page-header { margin-bottom: 0.8rem; }
.page-header h1 { margin-bottom: 4px !important; }
.page-header .subtitle { font-size: 0.88rem; color: #636366; letter-spacing: -0.01em; }

/* ── Page title (acima do step tracker) ── */
.page-title {
    font-size: 1.75rem; font-weight: 700; letter-spacing: -0.035em;
    color: var(--tx); line-height: 1.15; margin-bottom: 1rem;
}
.page-title .page-tag {
    display: inline-block; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--ac); background: var(--ac-bg);
    border: 0.5px solid var(--ac-bd); border-radius: var(--r-pill);
    padding: 2px 8px; vertical-align: middle; margin-right: 8px; margin-bottom: 2px;
}

.section-eyebrow {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--ac); margin-bottom: 6px;
}
.section-title {
    font-size: 1.25rem; font-weight: 650; letter-spacing: -0.025em;
    color: var(--tx); margin-bottom: 1rem;
}
.apple-divider {
    height: 0.5px;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.09), transparent);
    margin: 1.5rem 0; border: none;
}
.ref-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--tx3); margin-bottom: 4px;
}

.success-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--gr-bg); border: 0.5px solid rgba(52,199,89,0.22);
    color: #1A7F37; border-radius: var(--r-pill); padding: 5px 14px;
    font-size: 0.82rem; font-weight: 500; margin-top: 8px;
}
.img-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(88,86,214,0.08); border: 0.5px solid rgba(88,86,214,0.22);
    color: #5856D6; border-radius: var(--r-pill); padding: 4px 12px;
    font-size: 0.78rem; font-weight: 500; margin-top: 4px;
}
.suspect-card {
    background: var(--am-bg); border: 0.5px solid rgba(255,149,0,0.22);
    border-left: 3px solid var(--am); border-radius: 10px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.82rem;
}
.suspect-card .suspect-title  { font-weight: 600; color: #B25000; font-size: 0.84rem; margin-bottom: 4px; }
.suspect-card .suspect-reason { color: #7A4500; line-height: 1.5; }

.agent-badge {
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: var(--r-pill); padding: 3px 10px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
    margin-right: 6px; vertical-align: middle;
}
.agent-gemini { background: rgba(59,130,246,0.08); border: 0.5px solid rgba(59,130,246,0.22); color: #1D4ED8; }
.agent-cohere { background: rgba(249,115,22,0.08); border: 0.5px solid rgba(249,115,22,0.22); color: #C2410C; }
.agent-audit  { background: rgba(239,68,68,0.08);  border: 0.5px solid rgba(239,68,68,0.22);  color: #B91C1C; }

/* ── Upload count badge ── */
.upload-count {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--ac-bg); border: 0.5px solid var(--ac-bd);
    border-radius: var(--r-pill); padding: 5px 14px;
    font-size: 0.82rem; font-weight: 600; color: var(--ac);
    margin-bottom: 12px;
}
.upload-count.complete {
    background: var(--gr-bg); border-color: rgba(52,199,89,0.22);
    color: #1A7F37;
}
.upload-count.none {
    background: rgba(0,0,0,0.04); border-color: var(--ln);
    color: var(--tx3);
}

/* ── Helper text under disabled button ── */
.btn-helper {
    font-size: 0.75rem; color: var(--tx3); margin-top: 6px;
    display: flex; align-items: center; gap: 4px;
}

/* ── Back button — visually secondary ── */
.stButton > button.back-btn,
button[data-testid="baseButton-secondary"].back-btn {
    background: transparent !important; color: var(--tx2) !important;
    border: 0.5px solid var(--ln2) !important;
    box-shadow: none !important;
}
.stButton > button.back-btn:hover {
    background: rgba(0,0,0,0.04) !important; color: var(--tx) !important;
    transform: none !important; box-shadow: none !important;
}

/* ── Step action bar (bottom nav) ── */
.step-action-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 0 4px; margin-top: 8px;
    border-top: 0.5px solid var(--ln);
}

/* ── Section header row ── */
.section-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 1rem;
}
.section-header-left { display: flex; flex-direction: column; gap: 4px; }

/* ══ STATUS pill com ponto pulsante ══ */
@keyframes statusPulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(1.4); }
}
.status-dot-pulse { animation: statusPulse 2.2s ease-in-out infinite; }

/* ══ OVERRIDES MODO ESCURO ══ */
.stApp[data-theme="dark"] { background: #000000 !important; }
[data-theme="dark"] [data-testid="stSidebar"] {
    background: linear-gradient(160deg,
        rgba(28,28,30,0.92) 0%,
        rgba(20,20,22,0.80) 60%,
        rgba(28,28,30,0.88) 100%) !important;
    border-right: 0.5px solid rgba(255,255,255,0.08) !important;
    box-shadow: 2px 0 24px rgba(0,0,0,0.3) !important;
}
[data-theme="dark"] .sb-brand { border-bottom-color: rgba(255,255,255,0.07) !important; }
[data-theme="dark"] .sb-app-name { color: #F5F5F7 !important; }
[data-theme="dark"] .sb-footer { border-top-color: rgba(255,255,255,0.07) !important; }
[data-theme="dark"] h1, [data-theme="dark"] h2 { color: #F5F5F7 !important; }
[data-theme="dark"] h3 { color: #E5E5EA !important; }
[data-theme="dark"] p, [data-theme="dark"] label, [data-theme="dark"] .stMarkdown { color: #AEAEB2 !important; }
[data-theme="dark"] .stCaption, [data-theme="dark"] small { color: #636366 !important; }
[data-theme="dark"] .stTextInput input, [data-theme="dark"] .stDateInput input,
[data-theme="dark"] .stNumberInput input, [data-theme="dark"] .stTextArea textarea {
    background: rgba(44,44,46,0.92) !important;
    border: 0.5px solid rgba(255,255,255,0.10) !important; color: #F5F5F7 !important;
}
[data-theme="dark"] .stButton > button {
    background: rgba(44,44,46,0.92) !important; color: #0A84FF !important;
    border: 0.5px solid rgba(10,132,255,0.28) !important;
}
[data-theme="dark"] .stButton > button[kind="primary"],
[data-theme="dark"] button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, #0A84FF 0%, #0071E3 100%) !important;
    color: #fff !important; border: none !important;
}
[data-theme="dark"] .stDownloadButton > button { background: linear-gradient(180deg, #30D158 0%, #25A244 100%) !important; }
[data-theme="dark"] [data-testid="stFileUploader"] {
    background: rgba(44,44,46,0.62) !important;
    border: 1.5px dashed rgba(10,132,255,0.28) !important;
}
[data-theme="dark"] div[data-testid="stAlert"] {
    background: rgba(10,132,255,0.10) !important; border-left: 3px solid #0A84FF !important;
}
[data-theme="dark"] .streamlit-expanderHeader {
    background: rgba(44,44,46,0.78) !important; border: 0.5px solid rgba(255,255,255,0.07) !important;
}
[data-theme="dark"] .streamlit-expanderContent {
    background: rgba(44,44,46,0.52) !important; border: 0.5px solid rgba(255,255,255,0.06) !important;
}
[data-theme="dark"] [data-baseweb="select"] > div {
    background: rgba(44,44,46,0.88) !important; border-color: rgba(255,255,255,0.10) !important;
}
[data-theme="dark"] [data-testid="stDataFrame"], [data-theme="dark"] iframe {
    border: 0.5px solid rgba(255,255,255,0.07) !important;
}
[data-theme="dark"] .glass-card {
    background: linear-gradient(135deg, rgba(44,44,46,0.82) 0%, rgba(28,28,30,0.54) 100%) !important;
    border: 0.5px solid rgba(255,255,255,0.09) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.32) !important;
}
[data-theme="dark"] .glass-card .supplier-name  { color: #F5F5F7 !important; }
[data-theme="dark"] .glass-card .supplier-label { color: #636366 !important; }
[data-theme="dark"] .step-sep::after { background: rgba(255,255,255,0.14) !important; }
[data-theme="dark"] .step-track {
    background: linear-gradient(135deg, rgba(44,44,46,0.82) 0%, rgba(28,28,30,0.58) 100%) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
}
[data-theme="dark"] .step-dot.idle { background: rgba(255,255,255,0.09) !important; }
[data-theme="dark"] .step-text .num { color: #E5E5EA !important; }
[data-theme="dark"] .section-title  { color: #F5F5F7 !important; }
[data-theme="dark"] .page-header .subtitle { color: #8E8E93 !important; }
[data-theme="dark"] .apple-divider {
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.09), transparent) !important;
}
[data-theme="dark"] .ref-label { color: #636366 !important; }
[data-theme="dark"] .suspect-card {
    background: rgba(255,149,0,0.06) !important; border-color: rgba(255,149,0,0.18) !important;
}
[data-theme="dark"] .suspect-card .suspect-title  { color: #FF9F0A !important; }
[data-theme="dark"] .suspect-card .suspect-reason { color: #FFCC80 !important; }
[data-theme="dark"] .agent-gemini { background: rgba(96,165,250,0.07) !important; color: #60A5FA !important; }
[data-theme="dark"] .agent-cohere { background: rgba(249,115,22,0.07) !important; color: #FB923C !important; }
[data-theme="dark"] .agent-audit  { background: rgba(248,113,113,0.07) !important; color: #F87171 !important; }
[data-theme="dark"] .sb-status-row:hover { background: rgba(255,255,255,0.04) !important; }
[data-theme="dark"] .sb-pref-pill {
    background: rgba(10,132,255,0.12) !important;
    border-color: rgba(10,132,255,0.25) !important; color: #60A5FA !important;
}
[data-theme="dark"] .sb-version { color: #48484A !important; }
</style>
""", unsafe_allow_html=True)


# ── State init ────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 1,
        "supplier_data": {},
        "normalized_items": [],
        "edited_items": [],
        "api_key_ok": False,
        "cohere_ok": False,
        "preferences": {"corrections": [], "version": 1},
        "preferences_context": "",
        "prefs_loaded": False,
        "catalog": [],
        "approved_supplier": None,
        "original_texts": {},
        "original_images": {},
        "auto_run_extraction": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Inicialização das APIs ─────────────────────────────────────────────────────
def _get_sb_creds():
    try:
        return st.secrets.get("SUPABASE_URL", ""), st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        return "", ""


def _init_apis():
    status = get_system_status()

    # Gemini — pool de chaves
    if not st.session_state.api_key_ok and status["gemini_configured"]:
        key = get_random_gemini_key()
        if key:
            try:
                configure_gemini_with_key(key)
                st.session_state.api_key_ok = True
            except Exception:
                pass

    # Cohere
    if not st.session_state.cohere_ok and status["cohere_configured"]:
        try:
            get_cohere_client()
            st.session_state.cohere_ok = True
        except Exception:
            pass

    return status


_system_status = _init_apis()

# ── Auto-carrega preferências e catálogo do Supabase ─────────────────────────
if not st.session_state.prefs_loaded:
    _sb_url, _sb_key = _get_sb_creds()
    if _sb_url and _sb_key:
        try:
            from modules.preferences_manager import load_from_supabase
            loaded = load_from_supabase(_sb_url, _sb_key)
            st.session_state.preferences = loaded
            st.session_state.preferences_context = build_prompt_context(loaded)
        except Exception:
            pass
        try:
            catalog = load_catalog_from_supabase(_sb_url, _sb_key)
            st.session_state.catalog = catalog or []
        except Exception:
            st.session_state.catalog = []
    st.session_state.prefs_loaded = True


# ── Sidebar ───────────────────────────────────────────────────────────────────
_FILIAIS = [
    "Taquara",
    "Duque de Caxias",
    "São Gonçalo",
    "Piraí",
    "São Pedro",
    "Petrópolis",
]

with st.sidebar:
    # ── Brand header com logo Grupo EBD ──────────────────────────────────────
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo-wrap">
            <img
                src="https://www.ebdgrupo.com.br/wp-content/uploads/2019/08/AF-GrupoEBD-1024x703.png"
                alt="Grupo EBD"
            />
        </div>
        <div class="sb-brand-text">
            <div class="sb-app-name">Mapa de Compras</div>
            <div class="sb-app-sub">Grupo EBD · Depto. Compras</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Agentes IA ────────────────────────────────────────────────────────────
    n_keys = _system_status.get("gemini_key_count", 0)

    gem_ok    = st.session_state.api_key_ok
    gem_dot   = "ok" if gem_ok else "err"
    gem_label = (
        "Gemini  ·  {} chave{}".format(n_keys, "s" if n_keys != 1 else "")
        if gem_ok else "Gemini não configurado"
    )
    gem_sub   = "Extração + Auditoria" if gem_ok else "Adicione GEMINI_API_KEYS nos secrets"

    coh_ok    = st.session_state.cohere_ok
    coh_dot   = "ok" if coh_ok else "err"
    coh_label = "Cohere command-a" if coh_ok else "Cohere não configurado"
    coh_sub   = "Normalização + Cruzamento" if coh_ok else "Adicione COHERE_API_KEY nos secrets"

    st.markdown(
        '<span class="sb-section-label">Agentes IA</span>'
        '<div class="sb-status-row">'
        '  <div class="sb-status-dot {gd}"></div>'
        '  <div class="sb-status-text">'
        '    <span class="sb-status-label">{gl}</span>'
        '    <span class="sb-status-sub">{gs}</span>'
        '  </div>'
        '</div>'
        '<div class="sb-status-row">'
        '  <div class="sb-status-dot {cd}"></div>'
        '  <div class="sb-status-text">'
        '    <span class="sb-status-label">{cl}</span>'
        '    <span class="sb-status-sub">{cs}</span>'
        '  </div>'
        '</div>'.format(
            gd=gem_dot, gl=gem_label, gs=gem_sub,
            cd=coh_dot, cl=coh_label, cs=coh_sub,
        ),
        unsafe_allow_html=True,
    )

    # Preferências ativas
    n_corr = len(st.session_state.get("preferences", {}).get("corrections", []))
    if n_corr > 0:
        st.markdown(
            '<div class="sb-pref-pill">🧠 {} preferência{} ativa{}</div>'.format(
                n_corr,
                "s" if n_corr != 1 else "",
                "s" if n_corr != 1 else "",
            ),
            unsafe_allow_html=True,
        )

    # ── Cabeçalho ─────────────────────────────────────────────────────────────
    st.markdown("### Cabeçalho")
    numero_seq  = st.text_input("Nº Sequencial", value="2026001001")
    filial      = st.selectbox("Filial", options=_FILIAIS, index=2)
    responsavel = st.text_input("Responsável", value="")
    data_compra = st.date_input("Data", value=date.today())

    # ── Fornecedores ──────────────────────────────────────────────────────────
    st.markdown("### Fornecedores")
    n_suppliers = st.radio("Quantidade", [3, 4], horizontal=True)

    supplier_names = []
    for i in range(n_suppliers):
        name = st.text_input(
            "Fornecedor {}".format(i + 1),
            key="sup_name_{}".format(i),
            placeholder="Nome do fornecedor {}".format(i + 1),
        )
        supplier_names.append(name)

    # ── Orçamento Aprovado ────────────────────────────────────────────────────
    st.markdown("### Orçamento Aprovado")
    active_suppliers_sidebar = [s for s in supplier_names if s]
    approved_options  = ["Nenhum (não preencher)"] + active_suppliers_sidebar
    approved_selection = st.selectbox(
        "Fornecedor aprovado", options=approved_options, index=0,
        help="Se selecionado, preenche 'Preço Autorizado' no Excel automaticamente.",
    )
    st.session_state.approved_supplier = (
        None if approved_selection == "Nenhum (não preencher)" else approved_selection
    )

    # ── Footer com versão ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-footer">'
        '  <div class="sb-version">v3.1 · Multi-Agente · 2026</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Helpers UI ────────────────────────────────────────────────────────────────
step = st.session_state.step


def step_tracker():
    steps = [("Upload", "PDFs/Imgs"), ("Extração", "IA"), ("Revisão", "Itens"), ("Download", "Excel")]
    parts = []
    for i, (label, sub) in enumerate(steps, 1):
        if i < step:    cls = "done";   icon = "&#10003;"
        elif i == step: cls = "active"; icon = str(i)
        else:           cls = "idle";   icon = str(i)
        parts.append(
            '<div class="step-item">'
            '<div class="step-dot {cls}">{icon}</div>'
            '<div class="step-text"><div class="num">{label}</div><div class="sub">{sub}</div></div>'
            '</div>'.format(cls=cls, icon=icon, label=label, sub=sub)
        )
        if i < len(steps):
            parts.append('<div class="step-sep"></div>')
    st.markdown('<div class="step-track">{}</div>'.format("".join(parts)), unsafe_allow_html=True)


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def _agent_badge(agent: str) -> str:
    badges = {
        "gemini": '<span class="agent-badge agent-gemini">👁 Gemini</span>',
        "cohere": '<span class="agent-badge agent-cohere">⚙ Cohere</span>',
        "audit":  '<span class="agent-badge agent-audit">🔍 Auditor</span>',
    }
    return badges.get(agent, "")


# ── Topbar + Page header ──────────────────────────────────────────────────────
_STEP_TITLES = {
    1: "Upload dos Orçamentos",
    2: "Extração via IA",
    3: "Revisão e Aprovação",
    4: "Download do Mapa",
}
_step_title_now = _STEP_TITLES.get(step, "Mapa de Compras")

# Monta a topbar sem .format() para evitar conflito com {} do JavaScript
_topbar_html = (
    '<div class="topbar" id="app-topbar">'
    '  <div class="topbar-title">Mapa de Compras &mdash; ' + _step_title_now + '</div>'
    '  <div class="topbar-step-pill">Passo ' + str(step) + ' de 4</div>'
    '</div>'
    '<script>'
    '(function(){'
    '  var tb = document.getElementById("app-topbar");'
    '  if(!tb) return;'
    '  function onScroll(){'
    '    var y = window.scrollY || document.documentElement.scrollTop'
    '           || document.body.scrollTop || 0;'
    '    tb.classList.toggle("scrolled", y > 48);'
    '  }'
    '  window.addEventListener("scroll", onScroll, {passive:true});'
    '  var main = document.querySelector("[data-testid=\'stAppViewContainer\']")'
    '           || document.querySelector(".main");'
    '  if(main) main.addEventListener("scroll", onScroll, {passive:true});'
    '  onScroll();'
    '})();'
    '</script>'
    '<div class="page-title" style="padding-top:0.6rem;">'
    '  <span class="page-tag">EBD Compras</span>'
    '  Mapa de Compras'
    '</div>'
)
st.markdown(_topbar_html, unsafe_allow_html=True)

step_tracker()


# ═════════════════════════════════════════════════════════════════════════════
# PASSO 1 · Upload
# ═════════════════════════════════════════════════════════════════════════════
if step == 1:
    st.markdown(
        '<div class="section-eyebrow">Passo 1 de 4</div>'
        '<div class="section-title">Upload dos orçamentos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:var(--tx2);font-size:0.88rem;margin-bottom:1.5rem;">'
        'Carregue um PDF ou imagem por fornecedor. '
        'PDFs escaneados são detectados automaticamente e enviados para OCR via Gemini Vision.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── CSS: file_uploader inteiro vira o painel único ───────────────────────
    st.markdown("""
    <style>
    /* ── O widget inteiro do uploader é o painel ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.78) !important;
        backdrop-filter: blur(22px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(22px) saturate(180%) !important;
        border: 1px solid rgba(255,255,255,0.92) !important;
        border-radius: 20px !important;
        padding: 22px 22px 18px 22px !important;
        box-shadow: 0 4px 28px rgba(0,0,0,0.07),
                    0 1px 0 rgba(255,255,255,0.9) inset !important;
        transition: box-shadow 0.25s ease, transform 0.2s ease !important;
        min-height: 220px !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 10px !important;
    }
    [data-testid="stFileUploader"]:hover {
        box-shadow: 0 8px 36px rgba(0,0,0,0.11),
                    0 1px 0 rgba(255,255,255,0.9) inset !important;
        transform: translateY(-2px) !important;
    }

    /* ── Label do uploader = eyebrow + nome do fornecedor ── */
    [data-testid="stFileUploader"] label {
        display: block !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
        /* linha 1: eyebrow em caps — feito via ::before no wrapper */
        color: #1C1C1E !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        padding: 0 !important;
        margin-bottom: 2px !important;
        line-height: 1.3 !important;
        cursor: default !important;
        pointer-events: none !important;
    }

    /* ── Dropzone: preenche o espaço restante do painel ── */
    [data-testid="stFileUploaderDropzone"] {
        flex: 1 !important;
        min-height: 130px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(0,122,255,0.03) !important;
        border: 1.5px dashed rgba(0,122,255,0.18) !important;
        border-radius: 12px !important;
        transition: background 0.2s, border-color 0.2s !important;
        cursor: pointer !important;
        gap: 6px !important;
        position: relative !important;
    }
    /* Overlay invisível que expande a área de drop para cobrir o card inteiro */
    [data-testid="stFileUploaderDropzone"]::before {
        content: '' !important;
        position: absolute !important;
        inset: -22px -22px -18px -22px !important;
        z-index: 0 !important;
        cursor: pointer !important;
        border-radius: 20px !important;
    }
    [data-testid="stFileUploaderDropzone"] > * { position: relative !important; z-index: 1 !important; }
    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploader"]:hover [data-testid="stFileUploaderDropzone"] {
        background: rgba(0,122,255,0.07) !important;
        border-color: rgba(0,122,255,0.45) !important;
    }

    /* Ícone e texto da dropzone */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 4px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        width: 28px !important;
        height: 28px !important;
        opacity: 0.35 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 0.78rem !important;
        color: #8E8E93 !important;
        text-align: center !important;
    }
    /* Botão Browse */
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0.8rem !important;
        padding: 6px 16px !important;
        border-radius: 980px !important;
        border: 1px solid rgba(0,113,227,0.3) !important;
        background: rgba(255,255,255,0.9) !important;
        color: #0071E3 !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        transition: all 0.15s !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: rgba(0,113,227,0.06) !important;
        border-color: rgba(0,113,227,0.5) !important;
    }

    /* Status pill após upload */
    .upanel-status {
        font-size: 0.78rem;
        font-weight: 500;
        color: #1A7F37;
        background: rgba(52,199,89,0.1);
        border: 1px solid rgba(52,199,89,0.2);
        border-radius: 980px;
        padding: 3px 12px;
        display: inline-block;
        margin-top: 2px;
    }
    .upanel-status.img {
        color: #0071E3;
        background: rgba(0,113,227,0.08);
        border-color: rgba(0,113,227,0.18);
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_files = {}
    any_uploaded   = False

    # ── Grid de colunas ──────────────────────────────────────────────────────
    if n_suppliers <= 3:
        all_cols = st.columns(n_suppliers, gap="medium")
    else:
        row1 = st.columns(2, gap="medium")
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        row2 = st.columns(2, gap="medium")
        all_cols = list(row1) + list(row2)

    for i in range(n_suppliers):
        sname    = supplier_names[i] or "Fornecedor {}".format(i + 1)
        # Label do uploader = eyebrow em caps + nome do fornecedor
        # O CSS estiliza o label como header do painel
        label = "ORÇAMENTO {}  ·  {}".format(i + 1, sname.upper())

        with all_cols[i]:
            f = st.file_uploader(
                label,
                type=["pdf", "png", "jpg", "jpeg"],
                key="file_{}".format(i),
            )
            if f:
                raw = f.read()
                if isinstance(raw, str):
                    raw = raw.encode("latin-1")
                is_img = _is_image_file(f.name)
                uploaded_files[sname] = {"bytes": raw, "name": f.name, "is_image": is_img}
                if is_img:
                    st.markdown(
                        '<div class="upanel-status img">🖼 {}</div>'.format(f.name[:32]),
                        unsafe_allow_html=True,
                    )
                else:
                    pages = get_pdf_page_count(raw)
                    st.markdown(
                        '<div class="upanel-status">✓ {}  ·  {} pág.</div>'.format(f.name[:30], pages),
                        unsafe_allow_html=True,
                    )
                any_uploaded = True

    # ── Lista de referência ──────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="ref-label">Lista de referência <span style="font-weight:400;color:#AEAEB2;">(opcional)</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Cole os itens que você quer comprar. Ex: HIPOCLORITO 5% 5L, 2 BB")
    ref_text = st.text_area(
        "", height=110,
        placeholder="DETERGENTE GOLD 5L, 1 BB\nSACO DE LIXO 100L, 6 PCT\n...",
        label_visibility="collapsed",
    )

    # ── Resumo de uploads + botão avançar ───────────────────────────────────
    n_uploaded = sum(1 for i in range(n_suppliers) if st.session_state.get("file_{}".format(i)))
    n_total    = n_suppliers

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if n_uploaded == 0:
        badge_cls  = "none"
        badge_text = "Nenhum arquivo enviado"
        badge_icon = "○"
    elif n_uploaded < n_total:
        badge_cls  = ""
        badge_text = "{} de {} arquivo{} enviado{}".format(
            n_uploaded, n_total,
            "s" if n_total != 1 else "",
            "s" if n_uploaded != 1 else "",
        )
        badge_icon = "◑"
    else:
        badge_cls  = "complete"
        badge_text = "Todos os {} arquivos enviados".format(n_total)
        badge_icon = "✓"

    st.markdown(
        '<div class="upload-count {cls}">{icon}&nbsp;&nbsp;{text}</div>'.format(
            cls=badge_cls, icon=badge_icon, text=badge_text,
        ),
        unsafe_allow_html=True,
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button(
            "Avançar para extração →",
            type="primary", disabled=not any_uploaded, use_container_width=True,
        ):
            st.session_state.uploaded_files    = uploaded_files
            st.session_state.ref_text          = ref_text
            st.session_state.step              = 2
            st.session_state.auto_run_extraction = True
            st.rerun()
        if not any_uploaded:
            st.markdown(
                '<div class="btn-helper">↑ Faça upload de pelo menos 1 orçamento</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PASSO 2 · Extração IA (Fluxo Maestro Multi-Agente)
# ═════════════════════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown(
        '<div class="section-eyebrow">Passo 2 de 4</div>'
        '<div class="section-title">Extração, Normalização e Auditoria via IA</div>',
        unsafe_allow_html=True,
    )

    # Verifica disponibilidade dos agentes
    if not st.session_state.api_key_ok:
        st.error(
            "Gemini não configurado. "
            "Adicione GEMINI_API_KEYS ao arquivo `.streamlit/secrets.toml`."
        )
        st.stop()

    if not st.session_state.cohere_ok:
        st.error(
            "Cohere não configurado. "
            "Adicione COHERE_API_KEY ao arquivo `.streamlit/secrets.toml`."
        )
        st.stop()

    uploaded_files = st.session_state.get("uploaded_files", {})
    ref_text       = st.session_state.get("ref_text", "")

    # Parse da lista de referência
    reference_list = []
    if ref_text.strip():
        for line in ref_text.strip().split("\n"):
            parts_line = line.split(",")
            if len(parts_line) >= 2:
                item_name = parts_line[0].strip().upper()
                rest = parts_line[1].strip().split()
                try:
                    qty  = float(rest[0])
                    unit = rest[1].upper() if len(rest) > 1 else "UN"
                except Exception:
                    qty, unit = 1, "UN"
                reference_list.append({"item": item_name, "quantidade": qty, "unidade": unit})

    # Consome flag de auto-run (setada pelo botão "Avançar" do passo 1)
    auto_run = st.session_state.pop("auto_run_extraction", False)

    if not auto_run:
        # Mostra botão manual apenas se não veio do auto-run
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            manual_run = st.button(
                "Iniciar extração com IA →", type="primary", use_container_width=True
            )
    else:
        manual_run = False

    run_extraction = auto_run or manual_run

    if run_extraction:
        supplier_items  = {}
        original_texts  = {}   # textos originais para a auditoria do Gemini
        original_images = {}   # imagens originais para auditoria visual
        prefs_ctx       = st.session_state.get("preferences_context", "")
        catalog         = st.session_state.get("catalog") or None

        with st.status("Analisando orçamentos…", expanded=True) as status_box:

            # ─────────────────────────────────────────────────────────────────
            # ETAPA 1 — Extração por fornecedor via Gemini
            # ─────────────────────────────────────────────────────────────────
            for supplier_name, finfo in uploaded_files.items():
                pdf_bytes = finfo["bytes"]
                fname     = finfo["name"]
                is_image  = finfo["is_image"]

                try:
                    if is_image:
                        # Imagem direta → Gemini Vision
                        mime = detect_image_mime(pdf_bytes)
                        st.write(
                            "🖼 {} Processando imagem **{}** de **{}**…".format(
                                _agent_badge("gemini"), fname, supplier_name
                            )
                        )
                        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                        if mime == "image/jpeg":
                            items = extract_items_from_jpeg_images(
                                [b64],
                                preferences_context=prefs_ctx,
                                catalog=catalog,
                            )
                        else:
                            items = extract_items_from_images(
                                [b64],
                                preferences_context=prefs_ctx,
                                catalog=catalog,
                            )
                        original_texts[supplier_name] = "[Imagem — OCR via Gemini Vision]"
                        original_images[supplier_name] = [b64]

                    else:
                        # PDF — detecta se é nativo ou escaneado
                        st.write("📄 Lendo PDF de **{}**…".format(supplier_name))
                        text, is_img_pdf = extract_text_from_pdf(pdf_bytes)

                        if is_img_pdf:
                            # PDF escaneado → Gemini Vision (OCR)
                            st.write(
                                "🔍 {} PDF escaneado detectado — enviando para OCR…".format(
                                    _agent_badge("gemini")
                                )
                            )
                            images = extract_images_from_pdf(pdf_bytes)
                            items  = extract_items_from_images(
                                images,
                                preferences_context=prefs_ctx,
                                catalog=catalog,
                            )
                            original_texts[supplier_name] = "[PDF escaneado — OCR via Gemini Vision]"
                            original_images[supplier_name] = images[:3]  # guarda para auditoria
                        else:
                            # PDF nativo com texto → Gemini extrai direto do texto
                            st.write(
                                "👁 {} Extraindo itens de **{}** via Gemini…".format(
                                    _agent_badge("gemini"), supplier_name
                                )
                            )
                            # Usa o prompt de visão com texto inline (sem imagem)
                            items = extract_items_from_images(
                                [],
                                preferences_context=prefs_ctx,
                                text_fallback=text,
                                catalog=catalog,
                            )
                            original_texts[supplier_name] = text

                    supplier_items[supplier_name] = items

                    n_suspect = sum(1 for it in items if it.get("is_suspect"))
                    msg = "✅ **{}** — {} item(s) extraído(s)".format(supplier_name, len(items))
                    if n_suspect:
                        msg += " · ⚠️ {} suspeito(s)".format(n_suspect)
                    st.write(msg)

                    with st.expander("Ver itens brutos de {}".format(supplier_name), expanded=False):
                        st.json(items)

                except RuntimeError as e:
                    status_box.update(label="Limite de requisições atingido", state="error")
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error("Erro ao processar {}: {}".format(supplier_name, e))

            if not supplier_items:
                status_box.update(label="Nenhum item extraído", state="error")
                st.warning("Nenhum item foi extraído. Verifique os arquivos e tente novamente.")
                st.stop()

            # ─────────────────────────────────────────────────────────────────
            # ETAPA 2 — Normalização e Fuzzy Match via Cohere
            # ─────────────────────────────────────────────────────────────────
            st.write(
                "⚙ {} Cruzando e normalizando itens entre {} fornecedor(es)… "
                "_(pode levar até 30s)_".format(
                    _agent_badge("cohere"), len(supplier_items)
                )
            )
            normalized = []
            normalization_error = None
            try:
                normalized = normalize_and_match(
                    supplier_items,
                    reference_list or None,
                    preferences_context=prefs_ctx,
                    catalog=st.session_state.get("catalog") or None,
                )
            except Exception as e:
                normalization_error = e

            if normalization_error is not None:
                status_box.update(label="Erro na normalização (Cohere)", state="error")
                st.error(
                    "**Erro na normalização (Cohere):** {}\n\n"
                    "Verifique se `COHERE_API_KEY` está correta no `.streamlit/secrets.toml` "
                    "e se o modelo `{}` ainda está disponível.".format(
                        normalization_error, "command-a-reasoning-08-2025"
                    )
                )
                st.stop()

            if not normalized:
                status_box.update(label="Normalização retornou lista vazia", state="error")
                st.error(
                    "**A normalização (Cohere) retornou 0 itens.** Isso pode indicar:\n"
                    "- Resposta do modelo foi truncada (`max_tokens` insuficiente)\n"
                    "- JSON malformado que não pôde ser parseado\n"
                    "- Modelo retornou resposta vazia\n\n"
                    "Itens brutos extraídos pelo Gemini (para diagnóstico):"
                )
                st.json(supplier_items)
                st.stop()

            # ─────────────────────────────────────────────────────────────────
            # ETAPA 3 — Auditoria Final via Gemini (cross-reference visual)
            # ─────────────────────────────────────────────────────────────────
            st.write(
                "🔍 {} Auditando mapa — cruzando com imagens originais dos orçamentos…".format(
                    _agent_badge("audit")
                )
            )
            try:
                normalized = audit_purchase_map(
                    normalized,
                    original_texts=original_texts,
                    original_images=original_images,
                )
            except Exception as e_audit:
                # Auditoria é não-bloqueante: avisa mas continua com os itens normalizados
                st.warning(
                    "⚠️ Auditoria Gemini falhou (não-bloqueante): {}. "
                    "Os itens foram normalizados mas não auditados.".format(e_audit)
                )

            # ─────────────────────────────────────────────────────────────────
            # Finaliza
            # ─────────────────────────────────────────────────────────────────
            # ── Guarda obrigatória: não avança se normalização retornou vazio ──
            if not normalized:
                status_box.update(
                    label="⚠️ Normalização não retornou itens — verifique os arquivos",
                    state="error",
                    expanded=True,
                )
                st.error(
                    "A etapa de normalização (Cohere) não retornou nenhum item. "
                    "Possíveis causas:\n"
                    "- O modelo Cohere falhou silenciosamente (verifique a chave `COHERE_API_KEY`)\n"
                    "- Os PDFs não contêm texto reconhecível (tente re-upload)\n"
                    "- Limite de tokens excedido (tente com menos fornecedores por vez)\n\n"
                    "**Itens brutos extraídos pelo Gemini:**"
                )
                st.json(supplier_items)
                st.stop()

            st.session_state.supplier_data    = supplier_items
            st.session_state.normalized_items = normalized
            st.session_state.edited_items     = [dict(x) for x in normalized]
            st.session_state.original_texts   = original_texts

            n_suspect_total = sum(1 for it in normalized if it.get("is_suspect"))
            label_done = "✅ {} itens prontos para revisão".format(len(normalized))
            if n_suspect_total:
                label_done += " · 🚨 {} item(s) para revisar".format(n_suspect_total)
            status_box.update(label=label_done, state="complete", expanded=False)
            st.session_state.step = 3
            st.rerun()

    st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Voltar para upload", use_container_width=True):
            st.session_state.step = 1
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PASSO 3 · Revisão — Human-in-the-Loop
# ═════════════════════════════════════════════════════════════════════════════
elif step == 3:
    st.markdown(
        '<div class="section-eyebrow">Passo 3 de 4</div>'
        '<div class="section-title">Revisão e Aprovação — Human-in-the-Loop</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "📋 Revise os itens abaixo. Linhas marcadas com 🚨 foram sinalizadas pelo "
        "**Auditor Gemini** (cross-reference com os documentos originais). "
        "Edite diretamente na tabela e clique em **Aprovar e Gerar Excel** quando estiver satisfeito."
    )

    # Usa edited_items se disponível (preserva edições anteriores do usuário);
    # cai para normalized_items como fallback.
    items = st.session_state.get("edited_items") or st.session_state.get("normalized_items") or []

    if not items:
        st.error(
            "**Nenhum item encontrado para revisão.**\n\n"
            "O estado interno da sessão foi perdido ou a normalização falhou silenciosamente. "
            "Clique em **← Voltar para extração** e execute novamente.\n\n"
            "Se o problema persistir após nova extração, verifique:\n"
            "- Se a chave `COHERE_API_KEY` está válida\n"
            "- Se o modelo `command-a-reasoning-08-2025` está disponível em sua conta Cohere\n"
            "- Os logs do terminal Streamlit para mensagens `[Cohere]`"
        )
        if st.button("← Voltar para extração"):
            st.session_state.step = 2
            st.rerun()
    else:
        # Painel de alertas do Auditor
        suspect_items = [it for it in items if it.get("is_suspect")]
        if suspect_items:
            with st.expander(
                "🚨 {} alerta(s) do Auditor — clique para revisar".format(len(suspect_items)),
                expanded=True,
            ):
                for it in suspect_items:
                    reasons = it.get("alert_reason") or []
                    if isinstance(reasons, str):
                        reasons = [reasons]
                    reasons_html = "".join("<div>• {}</div>".format(r) for r in reasons)
                    st.markdown(
                        '<div class="suspect-card">'
                        '<div class="suspect-title">🔍 {}</div>'
                        '<div class="suspect-reason">{}</div>'
                        '</div>'.format(it.get("item", "?"), reasons_html or "(sem detalhe)"),
                        unsafe_allow_html=True,
                    )

        st.caption("⚠️ Unidades permitidas: {}".format(", ".join(ALLOWED_UNITS)))

        active_suppliers = [s for s in supplier_names if s]

        # Monta tabela para st.data_editor
        rows = []
        for item in items:
            is_suspect = item.get("is_suspect", False)
            reasons    = item.get("alert_reason") or []
            if isinstance(reasons, str):
                reasons = [reasons]
            if is_suspect and reasons:
                r0        = reasons[0]
                alert_col = "🚨 " + (r0[:60] + "…" if len(r0) > 60 else r0)
            elif is_suspect:
                alert_col = "🚨 Verificar"
            else:
                alert_col = ""

            row = {
                "ID":         item.get("id", ""),
                "Item":       item.get("item", ""),
                "Marca":      item.get("marca") or "",
                "Qtd":        item.get("quantidade", 1),
                "UND":        item.get("unidade", "UN"),
                "🚨 Alerta":  alert_col,
            }
            for sname in active_suppliers:
                fdata = item.get("fornecedores", {}).get(sname, {})
                row["R$ {}".format(sname)] = fdata.get("preco_unit") if fdata else None
            row["Observação"] = item.get("observacao") or ""
            rows.append(row)

        col_cfg = {
            "ID":   st.column_config.NumberColumn("ID", min_value=1, step=1, width="small"),
            "Item": st.column_config.TextColumn("Item", width="large"),
            "Marca":st.column_config.TextColumn("Marca", width="medium"),
            "Qtd":  st.column_config.NumberColumn("Qtd", min_value=0, step=0.5, width="small"),
            "UND":  st.column_config.SelectboxColumn("UND", options=ALLOWED_UNITS, width="small"),
            "🚨 Alerta": st.column_config.TextColumn("🚨 Alerta do Auditor", width="large", disabled=True),
            "Observação": st.column_config.TextColumn("Obs", width="medium"),
        }
        for sname in active_suppliers:
            col_cfg["R$ {}".format(sname)] = st.column_config.NumberColumn(
                sname, min_value=0, format="R$ %.2f", width="medium"
            )

        edited_df = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            num_rows="dynamic",
            column_config=col_cfg,
            hide_index=True,
        )

        if st.session_state.approved_supplier:
            st.markdown(
                '<div style="font-size:0.85rem;color:#0071E3;padding:8px 0;">'
                '✓ Orçamento aprovado: <b>{}</b> — '
                'A coluna "Preço Autorizado" será preenchida automaticamente no Excel.</div>'.format(
                    st.session_state.approved_supplier
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

        # Adicionar item manualmente
        with st.expander("Adicionar item manualmente (e-commerce / 4º fornecedor)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            new_name  = c1.text_input("Nome do item")
            new_marca = c2.text_input("Marca")
            new_qtd   = c3.number_input("Qtd", min_value=0.0, step=1.0)
            new_und   = c4.selectbox("UND", options=ALLOWED_UNITS, index=0)
            price_cols = st.columns(len(active_suppliers))
            new_prices = {}
            for i, sname in enumerate(active_suppliers):
                new_prices[sname] = price_cols[i].number_input(
                    "Preço {}".format(sname), min_value=0.0, format="%.2f", key="np_{}".format(i)
                )
            new_obs = st.text_input("Observação", key="new_obs")
            if st.button("Adicionar item") and new_name:
                new_row = {
                    "ID": len(edited_df) + 1, "Item": new_name.upper(),
                    "Marca": new_marca, "Qtd": new_qtd, "UND": new_und,
                    "🚨 Alerta": "", "Observação": new_obs,
                }
                for sname in active_suppliers:
                    new_row["R$ {}".format(sname)] = (
                        new_prices[sname] if new_prices[sname] > 0 else None
                    )
                edited_df = pd.concat([edited_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success("'{}' adicionado.".format(new_name))

        # ── Converter DataFrame → lista de itens ─────────────────────────────
        def _is_nan(val) -> bool:
            """Retorna True para None e float NaN — robusto para valores do data_editor."""
            if val is None:
                return True
            try:
                return pd.isna(val)
            except (TypeError, ValueError):
                return False

        def df_to_items(df, sup_names):
            result = []
            for _, row in df.iterrows():
                item_val = row.get("Item")
                # Linhas deletadas no data_editor deixam Item como NaN ou vazio — pular
                if _is_nan(item_val) or not str(item_val).strip():
                    continue
                forn_dict = {}
                for sname in sup_names:
                    price = row.get("R$ {}".format(sname))
                    forn_dict[sname] = {
                        "preco_unit": float(price) if not _is_nan(price) and price else None,
                        "obs": None,
                    }
                und = str(row.get("UND") or "UN").upper()
                if und not in ALLOWED_UNITS:
                    und = "UN"
                id_val = row.get("ID")
                # ID pode ser NaN quando a linha foi adicionada manualmente sem ID
                if _is_nan(id_val):
                    id_val = len(result) + 1
                result.append({
                    "id":           int(float(id_val)),
                    "item":         str(item_val).strip().upper(),
                    "marca":        row.get("Marca") or None,
                    "quantidade":   float(row.get("Qtd") or 1) if not _is_nan(row.get("Qtd")) else 1.0,
                    "unidade":      und,
                    "fornecedores": forn_dict,
                    "observacao":   row.get("Observação") or None,
                })
            return result

        # ── Detecção de itens ausentes no fornecedor aprovado ─────────────────
        approved = st.session_state.get("approved_supplier")
        if approved and approved in active_suppliers:
            current_items = df_to_items(edited_df, active_suppliers)
            absent_from_approved = [
                it for it in current_items
                if not (it.get("fornecedores") or {}).get(approved, {}).get("preco_unit")
                and any(
                    (it.get("fornecedores") or {}).get(s, {}).get("preco_unit")
                    for s in active_suppliers if s != approved
                )
            ]
            if absent_from_approved:
                with st.expander(
                    "⚠️ {} item(s) sem preço em **{}** (orçamento aprovado) — clique para revisar".format(
                        len(absent_from_approved), approved
                    ),
                    expanded=True,
                ):
                    st.markdown(
                        '<div style="font-size:0.85rem;color:#B25000;margin-bottom:10px;">'
                        'Os itens abaixo existem apenas em orçamentos <b>recusados</b>. '
                        'Geralmente não precisam ser incluídos no mapa — você pode removê-los '
                        'ou mantê-los caso sejam compras complementares.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    absent_names = []
                    for it in absent_from_approved:
                        outros = ", ".join(
                            "{}=R${:.2f}".format(s, it["fornecedores"][s]["preco_unit"])
                            for s in active_suppliers
                            if s != approved and it["fornecedores"].get(s, {}).get("preco_unit")
                        )
                        st.markdown(
                            '<div class="suspect-card">'
                            '<div class="suspect-title">📦 {}</div>'
                            '<div class="suspect-reason">Sem preço em <b>{}</b> · '
                            'Outros fornecedores: {}</div>'
                            '</div>'.format(it["item"], approved, outros or "—"),
                            unsafe_allow_html=True,
                        )
                        absent_names.append(it["item"])

                    if st.button(
                        "🗑 Remover {} item(s) ausente(s) do aprovado".format(len(absent_from_approved)),
                        key="btn_remove_absent",
                    ):
                        # Filtra o DataFrame removendo as linhas dos itens ausentes
                        absent_set = set(n.upper() for n in absent_names)
                        mask = edited_df["Item"].apply(
                            lambda v: (str(v).strip().upper() not in absent_set)
                            if not _is_nan(v) else True
                        )
                        edited_df = edited_df[mask].reset_index(drop=True)
                        st.success("{} item(s) removido(s).".format(len(absent_names)))
                        st.rerun()

        # ── Painel de correções capturadas ─────────────────────────────────────
        with st.expander("🧠 Correções capturadas nesta sessão", expanded=False):
            ai_items = st.session_state.normalized_items
            corrections_so_far = detect_corrections(
                ai_items, df_to_items(edited_df, active_suppliers), active_suppliers
            )
            if corrections_so_far:
                for c in corrections_so_far:
                    st.markdown(
                        '<div style="font-size:0.82rem;padding:4px 0;border-bottom:1px solid rgba(128,128,128,0.1);">'
                        '<b>{}</b> · {}</div>'.format(c["type"], c["note"]),
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Nenhuma diferença detectada em relação ao output da IA.")

        st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
        col_back, col_fwd = st.columns([1, 3])
        with col_back:
            if st.button("← Voltar para extração", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with col_fwd:
            if st.button("✅ Aprovar e Gerar Excel →", type="primary", use_container_width=True):
                final = df_to_items(edited_df, active_suppliers)

                # Salva correções para aprendizado futuro
                new_corr = detect_corrections(
                    st.session_state.normalized_items, final, active_suppliers
                )
                if new_corr:
                    updated_prefs, n_added = merge_corrections(
                        st.session_state.get("preferences", {"corrections": []}), new_corr
                    )
                    st.session_state.preferences = updated_prefs
                    st.session_state.preferences_context = build_prompt_context(updated_prefs)
                    _sb_url, _sb_key = _get_sb_creds()
                    if _sb_url and _sb_key and n_added > 0:
                        save_to_supabase(_sb_url, _sb_key, updated_prefs)

                st.session_state.final_items = final
                st.session_state.step = 4
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PASSO 4 · Download
# ═════════════════════════════════════════════════════════════════════════════
elif step == 4:
    st.markdown(
        '<div class="section-eyebrow">Passo 4 de 4</div>'
        '<div class="section-title">Mapa de Compras pronto</div>',
        unsafe_allow_html=True,
    )

    final_items       = st.session_state.get("final_items", [])
    active_suppliers  = [s for s in supplier_names if s]
    approved_supplier = st.session_state.get("approved_supplier")

    if not final_items:
        st.warning("Nenhum item para gerar. Volte ao passo 3.")
    else:
        with st.spinner("Gerando planilha…"):
            try:
                result = generate_excel(
                    items=final_items,
                    supplier_names=active_suppliers,
                    numero_sequencial=numero_seq,
                    filial=filial,
                    responsavel=responsavel,
                    data_compra=data_compra,
                    approved_supplier=approved_supplier,
                )
                if isinstance(result, tuple):
                    xlsx_bytes, overflow_warning = result
                else:
                    xlsx_bytes, overflow_warning = result, None

                if overflow_warning:
                    st.warning(overflow_warning)

                filename = "MapaCompras_{}_{}.xlsx".format(
                    filial.replace(" ", "_"), data_compra.strftime("%d%m%Y")
                )

                total_menor = sum(
                    min(
                        item.get("fornecedores", {}).get(s, {}).get("preco_unit") or 999999
                        for s in active_suppliers
                    ) * item["quantidade"]
                    for item in final_items
                    if any(
                        item.get("fornecedores", {}).get(s, {}).get("preco_unit")
                        for s in active_suppliers
                    )
                )

                total_autorizado = None
                if approved_supplier:
                    total_autorizado = sum(
                        (item.get("fornecedores", {}).get(approved_supplier, {}).get("preco_unit") or 0)
                        * item["quantidade"]
                        for item in final_items
                    )

                autorizados_html = ""
                if total_autorizado is not None:
                    autorizados_html = (
                        '<div>'
                        '<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--tx3);margin-bottom:4px;">Total Autorizado ({})</div>'
                        '<div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:var(--ac);">R$ {:,.2f}</div>'
                        '</div>'
                    ).format(approved_supplier, total_autorizado)

                st.markdown(
                    '<div class="glass-card" style="margin-bottom:1.5rem;">'
                    '<div style="display:flex;gap:40px;align-items:center;flex-wrap:wrap;">'
                    '<div>'
                    '<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--tx3);margin-bottom:4px;">Itens comparados</div>'
                    '<div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:var(--tx);">{}</div>'
                    '</div>'
                    '<div>'
                    '<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--tx3);margin-bottom:4px;">Fornecedores</div>'
                    '<div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:var(--tx);">{}</div>'
                    '</div>'
                    '<div>'
                    '<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--tx3);margin-bottom:4px;">Total (menor preço)</div>'
                    '<div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#1A7F37;">R$ {:,.2f}</div>'
                    '</div>'
                    '{}'
                    '</div></div>'.format(
                        len(final_items), len(active_suppliers), total_menor, autorizados_html
                    ),
                    unsafe_allow_html=True,
                )

                col_dl, col_new = st.columns([2, 1])
                with col_dl:
                    st.download_button(
                        label="⬇  Baixar Mapa de Compras.xlsx",
                        data=xlsx_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True,
                    )
                with col_new:
                    if st.button("Novo mapa", use_container_width=True):
                        for k in ["step", "supplier_data", "normalized_items",
                                  "edited_items", "final_items", "uploaded_files", "original_texts"]:
                            st.session_state[k] = (
                                1 if k == "step"
                                else ({} if k in ["supplier_data", "uploaded_files", "original_texts"] else [])
                            )
                        st.rerun()

                prefs_now    = st.session_state.get("preferences", {})
                n_corr_total = len(prefs_now.get("corrections", []))
                if n_corr_total > 0:
                    st.markdown(
                        '<div style="font-size:0.85rem;color:#1A7F37;padding:8px 0;">'
                        '🧠 {} preferência(s) salva(s) automaticamente — '
                        'serão aplicadas na próxima extração.</div>'.format(n_corr_total),
                        unsafe_allow_html=True,
                    )

                st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-eyebrow" style="margin-top:1rem;">Preview</div>',
                    unsafe_allow_html=True,
                )

                preview_rows = []
                for item in final_items:
                    row = {
                        "#":    item["id"],
                        "Item": item["item"],
                        "Qtd":  item["quantidade"],
                        "UND":  item["unidade"],
                    }
                    prices = []
                    for s in active_suppliers:
                        p = item.get("fornecedores", {}).get(s, {}).get("preco_unit")
                        row[s] = "R$ {:,.2f}".format(p) if p else "—"
                        if p:
                            prices.append(p)
                    row["✦ Menor"] = "R$ {:,.2f}".format(min(prices)) if prices else "—"
                    row["Total"]   = "R$ {:,.2f}".format(item["quantidade"] * min(prices)) if prices else "—"
                    if approved_supplier:
                        ap = item.get("fornecedores", {}).get(approved_supplier, {}).get("preco_unit")
                        row["P. Autorizado"] = "R$ {:,.2f}".format(ap) if ap else "—"
                    preview_rows.append(row)

                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error("Erro ao gerar Excel: {}".format(e))
                st.exception(e)

    st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Voltar para revisão", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
