# -*- coding: utf-8 -*-
"""
streamlit_app.py — Gerador de Mapa de Compras
Design: Apple HIG · Liquid Glass · SF Pro system stack
"""
import sys, base64
import streamlit as st
import pandas as pd
from datetime import date

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from modules.pdf_extractor   import (
    extract_text_from_pdf, extract_images_from_pdf, get_pdf_page_count,
    image_to_base64_png, image_to_base64_jpeg, detect_image_mime,
)
from modules.gemini_processor import (
    configure as configure_gemini,
    extract_items_from_text,
    extract_items_from_images,
    extract_items_from_jpeg_images,
    normalize_and_match,
    ALLOWED_UNITS,
)
from modules.excel_generator  import generate_excel
from modules.preferences_manager import (
    detect_corrections, merge_corrections, build_prompt_context,
    load_from_supabase, save_to_supabase,
    load_preferences, preferences_to_json_bytes,
    load_catalog_from_supabase
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mapa de Compras · EBD",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Apple Liquid Glass — modo claro e escuro ─────────────────────────────
# IMPORTANTE: a ordem correta é:
#   1) estilos base (modo claro)
#   2) componentes customizados (modo claro)
#   3) overrides de modo escuro (media query + [data-theme])
# Colocar os overrides escuros ANTES dos estilos base faz o base sobrescrever o dark.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display',
                 'DM Sans', 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ══════════════════════════════════════════════════════════
   1. ESTILOS BASE — MODO CLARO
   ══════════════════════════════════════════════════════════ */

.stApp { background: #F2F2F7 !important; min-height: 100vh; }

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.82) !important;
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border-right: 1px solid rgba(0,0,0,0.07) !important;
    box-shadow: 2px 0 32px rgba(0,0,0,0.05) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }

.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px !important; }

h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important;
     color: #1C1C1E !important; margin-bottom: 0.2rem !important; line-height: 1.15 !important; }
h2 { font-size: 1.35rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; color: #1C1C1E !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #3A3A3C !important; }
p, label, .stMarkdown, .stCaption { color: #3A3A3C !important; font-size: 0.9rem !important; line-height: 1.55 !important; }
.stCaption, small { color: #8E8E93 !important; font-size: 0.78rem !important; }

[data-testid="stSidebar"] h3 {
    font-size: 0.65rem !important; font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: #8E8E93 !important; margin: 1.4rem 0 0.5rem !important;
}
[data-testid="stSidebar"] hr { border: none !important; border-top: 1px solid rgba(0,0,0,0.07) !important; margin: 1rem 0 !important; }

.stTextInput input, .stDateInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.9) !important; border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 10px !important; padding: 10px 14px !important; font-size: 0.88rem !important;
    color: #1C1C1E !important; box-shadow: 0 1px 3px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #0071E3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15), 0 1px 3px rgba(0,0,0,0.04) !important;
    outline: none !important;
}

.stButton > button {
    background: rgba(255,255,255,0.9) !important; color: #0071E3 !important;
    border: 1px solid rgba(0,113,227,0.25) !important; border-radius: 980px !important;
    padding: 9px 22px !important; font-size: 0.88rem !important; font-weight: 500 !important;
    letter-spacing: -0.01em !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
    transition: all 0.18s cubic-bezier(0.25,0.46,0.45,0.94) !important; cursor: pointer !important;
}
.stButton > button:hover {
    background: rgba(0,113,227,0.06) !important; border-color: rgba(0,113,227,0.5) !important;
    transform: translateY(-1px) !important; box-shadow: 0 4px 16px rgba(0,113,227,0.18) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.98) !important; }
.stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, #1A8DFF 0%, #0071E3 100%) !important;
    color: #fff !important; border: none !important; box-shadow: 0 2px 12px rgba(0,113,227,0.38) !important;
}
.stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(180deg, #1E96FF 0%, #0077ED 100%) !important;
    box-shadow: 0 4px 20px rgba(0,113,227,0.5) !important; transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: linear-gradient(180deg, #34C759 0%, #28A745 100%) !important;
    color: #fff !important; border: none !important; border-radius: 980px !important;
    padding: 10px 26px !important; font-weight: 600 !important; font-size: 0.9rem !important;
    box-shadow: 0 2px 14px rgba(40,167,69,0.38) !important; transition: all 0.18s ease !important;
}
.stDownloadButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 22px rgba(40,167,69,0.45) !important; }

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.6) !important; backdrop-filter: blur(16px) !important;
    border: 1.5px dashed rgba(0,113,227,0.25) !important; border-radius: 14px !important;
    padding: 1.2rem !important; transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(0,113,227,0.55) !important; background: rgba(0,113,227,0.03) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { font-size: 0.85rem !important; color: #636366 !important; }

.stAlert { border-radius: 12px !important; border: none !important; backdrop-filter: blur(12px) !important; }
div[data-testid="stAlert"] { background: rgba(0,113,227,0.07) !important; border-left: 3px solid #0071E3 !important; border-radius: 12px !important; }
div[data-baseweb="notification"] { border-radius: 12px !important; }

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.75) !important; border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.07) !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 12px 16px !important; transition: background 0.2s !important;
}
.streamlit-expanderHeader:hover { background: rgba(255,255,255,0.95) !important; }
.streamlit-expanderContent {
    background: rgba(255,255,255,0.5) !important; border: 1px solid rgba(0,0,0,0.06) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 16px !important;
}

.stRadio > div { gap: 8px !important; }
.stRadio label {
    background: rgba(255,255,255,0.8) !important; border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 8px !important; padding: 6px 14px !important; font-size: 0.85rem !important; transition: all 0.15s !important;
}

.stProgress > div > div { background: linear-gradient(90deg, #0071E3, #34C759) !important; border-radius: 999px !important; }
.stProgress > div { background: rgba(0,0,0,0.06) !important; border-radius: 999px !important; height: 4px !important; }

[data-testid="stDataFrame"], iframe {
    border-radius: 12px !important; overflow: hidden !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important; border: 1px solid rgba(0,0,0,0.06) !important;
}
.stSpinner > div { color: #0071E3 !important; }
[data-baseweb="select"] > div { border-radius: 10px !important; border-color: rgba(0,0,0,0.1) !important; background: rgba(255,255,255,0.85) !important; }

/* ══════════════════════════════════════════════════════════
   2. COMPONENTES CUSTOMIZADOS — MODO CLARO
   ══════════════════════════════════════════════════════════ */

.glass-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 18px; padding: 20px 24px; margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07), 0 1px 0 rgba(255,255,255,0.9) inset;
    transition: box-shadow 0.25s ease, transform 0.2s ease;
}
.glass-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.1), 0 1px 0 rgba(255,255,255,0.9) inset;
    transform: translateY(-1px);
}
.glass-card .supplier-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #8E8E93; margin-bottom: 4px; }
.glass-card .supplier-name  { font-size: 1.05rem; font-weight: 600; color: #1C1C1E; letter-spacing: -0.02em; }

.step-track {
    display: flex; align-items: center;
    background: rgba(255,255,255,0.6); backdrop-filter: blur(16px);
    border: 1px solid rgba(0,0,0,0.07); border-radius: 14px;
    padding: 10px 20px; margin-bottom: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.step-item { display: flex; align-items: center; gap: 8px; flex: 1; position: relative; }
.step-item:not(:last-child)::after {
    content: ''; position: absolute; right: 0; top: 50%;
    transform: translateY(-50%); width: 1px; height: 20px; background: rgba(0,0,0,0.1);
}
.step-dot {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; flex-shrink: 0; transition: all 0.3s ease;
}
.step-dot.done   { background: #34C759; color: #fff; }
.step-dot.active { background: #0071E3; color: #fff; box-shadow: 0 0 0 4px rgba(0,113,227,0.2); }
.step-dot.idle   { background: rgba(0,0,0,0.08); color: #8E8E93; }
.step-text { font-size: 0.82rem; line-height: 1.2; }
.step-text .num { font-weight: 600; color: #1C1C1E; letter-spacing: -0.01em; }
.step-text .sub { font-size: 0.72rem; color: #8E8E93; }

.page-header { margin-bottom: 1.6rem; }
.page-header h1 { margin-bottom: 4px !important; }
.page-header .subtitle { font-size: 0.88rem; color: #636366; letter-spacing: -0.01em; }

.section-eyebrow { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #0071E3; margin-bottom: 6px; }
.section-title   { font-size: 1.25rem; font-weight: 650; letter-spacing: -0.025em; color: #1C1C1E; margin-bottom: 1rem; }

.apple-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,0,0,0.08), transparent); margin: 1.5rem 0; border: none; }
.ref-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #8E8E93; margin-bottom: 4px; }

.success-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(52,199,89,0.12); border: 1px solid rgba(52,199,89,0.25);
    color: #1A7F37; border-radius: 980px; padding: 5px 14px;
    font-size: 0.82rem; font-weight: 500; margin-top: 8px;
}
.img-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(88,86,214,0.1); border: 1px solid rgba(88,86,214,0.25);
    color: #5856D6; border-radius: 980px; padding: 4px 12px;
    font-size: 0.78rem; font-weight: 500; margin-top: 4px;
}
.suspect-card {
    background: rgba(255,149,0,0.07); border: 1px solid rgba(255,149,0,0.25);
    border-left: 3px solid #FF9500; border-radius: 10px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.82rem;
}
.suspect-card .suspect-title  { font-weight: 600; color: #B25000; font-size: 0.84rem; margin-bottom: 4px; }
.suspect-card .suspect-reason { color: #7A4500; line-height: 1.5; }

/* ══════════════════════════════════════════════════════════
   3. OVERRIDES MODO ESCURO — [data-theme="dark"]
   Streamlit injeta data-theme="dark" / "light" no <body>
   baseado na escolha do usuário nas configurações do app.
   NÃO usamos @media(prefers-color-scheme) pois ele leria
   o tema do SO e ignoraria a escolha dentro do Streamlit.
   ══════════════════════════════════════════════════════════ */

.stApp[data-theme="dark"] { background: #1C1C1E !important; }
.stApp[data-theme="dark"] ~ * [data-testid="stSidebar"],
[data-theme="dark"] [data-testid="stSidebar"] { background: rgba(28,28,30,0.92) !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }

[data-theme="dark"] h1, [data-theme="dark"] h2 { color: #F5F5F7 !important; }
[data-theme="dark"] h3 { color: #E5E5EA !important; }
[data-theme="dark"] p, [data-theme="dark"] label, [data-theme="dark"] .stMarkdown { color: #AEAEB2 !important; }
[data-theme="dark"] .stCaption, [data-theme="dark"] small { color: #636366 !important; }

[data-theme="dark"] .stTextInput input,
[data-theme="dark"] .stDateInput input,
[data-theme="dark"] .stNumberInput input,
[data-theme="dark"] .stTextArea textarea {
    background: rgba(44,44,46,0.9) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #F5F5F7 !important;
}
[data-theme="dark"] .stButton > button { background: rgba(44,44,46,0.9) !important; color: #0A84FF !important; border: 1px solid rgba(10,132,255,0.3) !important; }
[data-theme="dark"] .stButton > button[kind="primary"],
[data-theme="dark"] button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, #0A84FF 0%, #0071E3 100%) !important; color: #fff !important; border: none !important;
}
[data-theme="dark"] .stDownloadButton > button { background: linear-gradient(180deg, #30D158 0%, #25A244 100%) !important; }

[data-theme="dark"] [data-testid="stFileUploader"] { background: rgba(44,44,46,0.6) !important; border: 1.5px dashed rgba(10,132,255,0.3) !important; }
[data-theme="dark"] div[data-testid="stAlert"] { background: rgba(10,132,255,0.1) !important; border-left: 3px solid #0A84FF !important; }
[data-theme="dark"] .streamlit-expanderHeader { background: rgba(44,44,46,0.75) !important; border: 1px solid rgba(255,255,255,0.07) !important; }
[data-theme="dark"] .streamlit-expanderContent { background: rgba(44,44,46,0.5) !important; border: 1px solid rgba(255,255,255,0.06) !important; }
[data-theme="dark"] [data-baseweb="select"] > div { background: rgba(44,44,46,0.85) !important; border-color: rgba(255,255,255,0.1) !important; }
[data-theme="dark"] [data-testid="stDataFrame"], [data-theme="dark"] iframe { border: 1px solid rgba(255,255,255,0.07) !important; }

[data-theme="dark"] .glass-card { background: rgba(44,44,46,0.75) !important; border: 1px solid rgba(255,255,255,0.08) !important; box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important; }
[data-theme="dark"] .glass-card .supplier-name  { color: #F5F5F7 !important; }
[data-theme="dark"] .glass-card .supplier-label { color: #636366 !important; }
[data-theme="dark"] .step-track { background: rgba(44,44,46,0.75) !important; border: 1px solid rgba(255,255,255,0.07) !important; }
[data-theme="dark"] .step-dot.idle { background: rgba(255,255,255,0.1) !important; }
[data-theme="dark"] .step-text .num { color: #E5E5EA !important; }
[data-theme="dark"] .step-text .sub { color: #636366 !important; }
[data-theme="dark"] .section-title { color: #F5F5F7 !important; }
[data-theme="dark"] .page-header .subtitle { color: #8E8E93 !important; }
[data-theme="dark"] .apple-divider { background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important; }
[data-theme="dark"] .ref-label { color: #636366 !important; }
[data-theme="dark"] .suspect-card { background: rgba(255,149,0,0.05) !important; border-color: rgba(255,149,0,0.2) !important; }
[data-theme="dark"] .suspect-card .suspect-title  { color: #FF9F0A !important; }
[data-theme="dark"] .suspect-card .suspect-reason { color: #FFCC80 !important; }
</style>
""", unsafe_allow_html=True)


# ── State init ────────────────────────────────────────────────────────────────
def init_state():
    for k, v in {
        "step": 1, "supplier_data": {}, "normalized_items": [],
        "edited_items": [], "api_key_ok": False,
        "preferences": {"corrections": [], "version": 1},
        "preferences_context": "",
        "prefs_loaded": False,
        "catalog": [],
        "catalog_loaded": False,
        "approved_supplier": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Lê API key dos Secrets e configura Gemini automaticamente ─────────────────
def _get_sb_creds():
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        return url, key
    except Exception:
        return "", ""

def _init_gemini():
    """Configura Gemini a partir dos Secrets — sem expor a chave ao usuário."""
    if st.session_state.api_key_ok:
        return True
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            configure_gemini(api_key)
            st.session_state.api_key_ok = True
            return True
    except Exception:
        pass
    return False

_init_gemini()

# ── Auto-carrega preferências e catálogo do Supabase na primeira execução ─────
if not st.session_state.prefs_loaded:
    _sb_url, _sb_key = _get_sb_creds()
    if _sb_url and _sb_key:
        try:
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
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px; border-bottom:1px solid rgba(128,128,128,0.15); margin-bottom:4px;">
        <div style="font-size:1.15rem;font-weight:700;letter-spacing:-0.03em;">🗂 Mapa de Compras</div>
        <div style="font-size:0.72rem;color:#8E8E93;margin-top:2px;letter-spacing:0.02em;">GRUPO EBD · DEPTO. COMPRAS</div>
    </div>
    """, unsafe_allow_html=True)

    # Status da API — apenas indicador, sem campo de input
    if st.session_state.api_key_ok:
        st.markdown(
            '<div style="font-size:0.8rem;color:#1A7F37;font-weight:500;padding:8px 0 4px;">'
            '● API Gemini conectada</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="font-size:0.8rem;color:#C0392B;font-weight:500;padding:8px 0 4px;">'
            '● API Gemini não configurada</div>',
            unsafe_allow_html=True
        )
        st.caption("Configure GEMINI_API_KEY em `.streamlit/secrets.toml`")

    # Preferências — exibe contagem se houver correções na sessão (sem uploader de JSON)
    n_corr_sess = len(st.session_state.get("preferences", {}).get("corrections", []))
    if n_corr_sess > 0:
        st.markdown(
            f'<div style="font-size:0.78rem;color:#636366;padding:2px 0 8px;">'
            f'🧠 {n_corr_sess} correção(ões) ativa(s)</div>',
            unsafe_allow_html=True
        )

    st.markdown("### Cabeçalho")
    numero_seq  = st.text_input("Nº Sequencial", value="2026001001")
    filial      = st.text_input("Filial", value="São Gonçalo")
    responsavel = st.text_input("Responsável", value="")
    data_compra = st.date_input("Data", value=date.today())

    st.markdown("### Fornecedores")
    n_suppliers = st.radio("Quantidade", [3, 4], horizontal=True)

    supplier_names = []
    for i in range(n_suppliers):
        name = st.text_input(f"Fornecedor {i+1}", key=f"sup_name_{i}",
                             placeholder=f"Nome do fornecedor {i+1}")
        supplier_names.append(name)

    st.markdown("### Orçamento Aprovado")
    active_suppliers_sidebar = [s for s in supplier_names if s]
    approved_options = ["Nenhum (não preencher)"] + active_suppliers_sidebar
    approved_selection = st.selectbox(
        "Fornecedor aprovado",
        options=approved_options, index=0,
        help="Se selecionado, a coluna 'Preço Autorizado' será preenchida automaticamente."
    )
    if approved_selection == "Nenhum (não preencher)":
        st.session_state.approved_supplier = None
    else:
        st.session_state.approved_supplier = approved_selection

    st.markdown('<div style="margin-top:2rem;font-size:0.72rem;color:#8E8E93;text-align:center;">v2.0 · 2026</div>', unsafe_allow_html=True)


# ── Helpers UI ────────────────────────────────────────────────────────────────
step = st.session_state.step

def step_tracker():
    steps = [("Upload", "PDFs/Imgs"), ("Extração", "IA"), ("Revisão", "Itens"), ("Download", "Excel")]
    parts = []
    for i, (label, sub) in enumerate(steps, 1):
        if i < step:    cls = "done";   icon = "✓"
        elif i == step: cls = "active"; icon = str(i)
        else:           cls = "idle";   icon = str(i)
        parts.append(f"""
        <div class="step-item">
            <div class="step-dot {cls}">{icon}</div>
            <div class="step-text">
                <div class="num">{label}</div>
                <div class="sub">{sub}</div>
            </div>
        </div>""")
    st.markdown(f'<div class="step-track">{"".join(parts)}</div>', unsafe_allow_html=True)


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Gerador de Mapa de Compras</h1>
    <div class="subtitle">Faça upload dos orçamentos em PDF ou imagem · A IA extrai, normaliza e compara · Baixe o Excel pronto</div>
</div>
""", unsafe_allow_html=True)

step_tracker()


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 1 · Upload
# ─────────────────────────────────────────────────────────────────────────────
if step == 1:
    st.markdown('<div class="section-eyebrow">Passo 1 de 4</div><div class="section-title">Upload dos orçamentos</div>', unsafe_allow_html=True)
    st.info("Carregue um PDF ou imagem (PNG/JPEG) por fornecedor. PDFs escaneados e imagens são processados via Gemini Vision.")

    uploaded_files = {}
    any_uploaded   = False

    for i in range(n_suppliers):
        sname = supplier_names[i] or f"Fornecedor {i+1}"
        st.markdown(f"""
        <div class="glass-card">
            <div class="supplier-label">Orçamento {i+1}</div>
            <div class="supplier-name">{sname}</div>
        </div>
        """, unsafe_allow_html=True)

        f = st.file_uploader(
            "",
            type=["pdf", "png", "jpg", "jpeg"],
            key=f"file_{i}",
            label_visibility="collapsed",
        )
        if f:
            raw = f.read()
            if isinstance(raw, str):
                raw = raw.encode("latin-1")
            uploaded_files[sname] = {"bytes": raw, "name": f.name, "is_image": _is_image_file(f.name)}

            if _is_image_file(f.name):
                st.markdown(
                    f'<div class="img-badge">🖼 {f.name} &nbsp;·&nbsp; imagem</div>',
                    unsafe_allow_html=True
                )
            else:
                pages = get_pdf_page_count(raw)
                st.markdown(
                    f'<div class="success-pill">✓ {f.name} &nbsp;·&nbsp; {pages} pág.</div>',
                    unsafe_allow_html=True
                )
            any_uploaded = True

        st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="ref-label">Lista de referência (opcional)</div>', unsafe_allow_html=True)
    st.caption("Cole os itens que você quer comprar. Ex: HIPOCLORITO 5% 5L, 2 BB")
    ref_text = st.text_area(
        "", height=110,
        placeholder="DETERGENTE GOLD 5L, 1 BB\nSACO DE LIXO 100L, 6 PCT\n...",
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Avançar para extração →", type="primary", disabled=not any_uploaded, use_container_width=True):
            st.session_state.uploaded_files = uploaded_files
            st.session_state.ref_text = ref_text
            st.session_state.step = 2
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 2 · Extração IA
# ─────────────────────────────────────────────────────────────────────────────
elif step == 2:
    st.markdown('<div class="section-eyebrow">Passo 2 de 4</div><div class="section-title">Extração e normalização via IA</div>', unsafe_allow_html=True)

    if not st.session_state.api_key_ok:
        st.error("A chave da API Gemini não está configurada. Adicione GEMINI_API_KEY ao arquivo `.streamlit/secrets.toml`.")
        st.stop()

    uploaded_files = st.session_state.get("uploaded_files", {})
    ref_text       = st.session_state.get("ref_text", "")

    reference_list = []
    if ref_text.strip():
        for line in ref_text.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 2:
                item_name = parts[0].strip().upper()
                rest = parts[1].strip().split()
                try:    qty, unit = float(rest[0]), (rest[1].upper() if len(rest) > 1 else "UN")
                except: qty, unit = 1, "UN"
                reference_list.append({"item": item_name, "quantidade": qty, "unidade": unit})

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_extraction = st.button("Iniciar extração com IA →", type="primary", use_container_width=True)

    if run_extraction:
        supplier_items = {}
        prefs_ctx = st.session_state.get("preferences_context", "")

        with st.status("Analisando orçamentos…", expanded=True) as status_box:
            for supplier_name, finfo in uploaded_files.items():
                pdf_bytes = finfo["bytes"]
                fname     = finfo["name"]
                is_image  = finfo["is_image"]

                try:
                    if is_image:
                        mime = detect_image_mime(pdf_bytes)
                        st.write(f"🖼 Processando imagem **{fname}** de **{supplier_name}** via Gemini Vision…")
                        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                        if mime == "image/jpeg":
                            items = extract_items_from_jpeg_images([b64], preferences_context=prefs_ctx)
                        else:
                            items = extract_items_from_images([b64], preferences_context=prefs_ctx)
                    else:
                        st.write(f"📄 Lendo PDF de **{supplier_name}**…")
                        text, is_img_pdf = extract_text_from_pdf(pdf_bytes)
                        if is_img_pdf:
                            st.write("🔍 PDF escaneado detectado — enviando para OCR via Gemini Vision…")
                            images = extract_images_from_pdf(pdf_bytes)
                            items  = extract_items_from_images(images, preferences_context=prefs_ctx)
                        else:
                            st.write(f"🤖 Extraindo itens de **{supplier_name}** com IA…")
                            items = extract_items_from_text(text, preferences_context=prefs_ctx)

                    supplier_items[supplier_name] = items
                    n_suspect = sum(1 for it in items if it.get("is_suspect"))
                    msg = f"✅ **{supplier_name}** — {len(items)} item(s) extraído(s)"
                    if n_suspect:
                        msg += f" · ⚠️ {n_suspect} item(s) suspeito(s)"
                    st.write(msg)

                    with st.expander(f"Ver itens de {supplier_name}", expanded=False):
                        st.json(items)

                except RuntimeError as e:
                    status_box.update(label="Limite de requisições atingido", state="error")
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Erro ao processar {supplier_name}: {e}")

            if supplier_items:
                st.write(f"⚖️ Cruzando e normalizando itens entre os {len(supplier_items)} fornecedores…")
                st.write("_(Este passo pode demorar até 1 minuto — aguarde sem atualizar a página)_")
                try:
                    normalized = normalize_and_match(
                        supplier_items,
                        reference_list or None,
                        preferences_context=prefs_ctx,
                        catalog=st.session_state.get("catalog") or None,
                    )
                    st.session_state.supplier_data    = supplier_items
                    st.session_state.normalized_items = normalized
                    st.session_state.edited_items     = [dict(x) for x in normalized]
                    n_suspect_total = sum(1 for it in normalized if it.get("is_suspect"))
                    label_done = f"✅ {len(normalized)} itens prontos para revisão"
                    if n_suspect_total:
                        label_done += f" · ⚠️ {n_suspect_total} item(s) marcado(s) para revisão"
                    status_box.update(label=label_done, state="complete", expanded=False)
                    st.session_state.step = 3
                    st.rerun()
                except Exception as e:
                    status_box.update(label="Erro na normalização", state="error")
                    st.error(f"Erro na normalização: {e}")
            else:
                status_box.update(label="Nenhum item extraído", state="error")
                st.warning("Nenhum item foi extraído. Verifique os arquivos e tente novamente.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Voltar"):
        st.session_state.step = 1
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 3 · Revisão
# ─────────────────────────────────────────────────────────────────────────────
elif step == 3:
    st.markdown('<div class="section-eyebrow">Passo 3 de 4</div><div class="section-title">Revisão e ajustes</div>', unsafe_allow_html=True)
    st.info("Edite diretamente na tabela. Ajuste itens, quantidades e preços antes de gerar o Excel.")

    items = st.session_state.normalized_items

    # ── Aviso de itens suspeitos (não encontrados no catálogo) ────────────────
    suspects = [it for it in items if it.get("is_suspect")]
    if suspects:
        suspect_names = ", ".join(f"**{s['item']}**" for s in suspects[:5])
        suffix = f" (e mais {len(suspects)-5})" if len(suspects) > 5 else ""
        st.warning(
            f"⚠️ {len(suspects)} item(s) marcado(s) para revisão: "
            f"{suspect_names}{suffix}"
        )
    st.caption(f"⚠️ Unidades permitidas: {', '.join(ALLOWED_UNITS)}")

    if not items:
        st.warning("Nenhum item encontrado. Volte e processe os arquivos novamente.")
    else:
        # Painel de alertas de sanidade
        suspect_items = [it for it in items if it.get("is_suspect")]
        if suspect_items:
            with st.expander(f"⚠️ {len(suspect_items)} alerta(s) de validação — clique para ver", expanded=True):
                for it in suspect_items:
                    reasons_html = "".join(f'<div>• {r}</div>' for r in (it.get("alert_reason") or []))
                    st.markdown(f"""
                    <div class="suspect-card">
                        <div class="suspect-title">🔍 {it.get('item','?')}</div>
                        <div class="suspect-reason">{reasons_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

        active_suppliers = [s for s in supplier_names if s]

        rows = []
        for item in items:
            row = {
                "ID":    item.get("id", ""),
                "Item":  item.get("item", ""),
                "Marca": item.get("marca") or "",
                "Qtd":   item.get("quantidade", 1),
                "UND":   item.get("unidade", "UN"),
                "⚠":    "⚠️" if item.get("is_suspect") else "",
            }
            for sname in active_suppliers:
                fdata = item.get("fornecedores", {}).get(sname, {})
                row[f"R$ {sname}"] = fdata.get("preco_unit") if fdata else None
            row["Observação"] = item.get("observacao") or ""
            rows.append(row)

        edited_df = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "ID":    st.column_config.NumberColumn("ID", min_value=1, step=1, width="small"),
                "Item":  st.column_config.TextColumn("Item", width="large"),
                "Marca": st.column_config.TextColumn("Marca", width="medium"),
                "Qtd":   st.column_config.NumberColumn("Qtd", min_value=0, step=0.5, width="small"),
                "UND":   st.column_config.SelectboxColumn("UND", options=ALLOWED_UNITS, width="small"),
                "⚠":    st.column_config.TextColumn("⚠", width="small", disabled=True),
                **{f"R$ {s}": st.column_config.NumberColumn(f"{s}", min_value=0, format="R$ %.2f", width="medium") for s in active_suppliers},
                "Observação": st.column_config.TextColumn("Obs", width="medium"),
            },
            hide_index=True,
        )

        if st.session_state.approved_supplier:
            st.markdown(
                f'<div style="font-size:0.85rem;color:#0071E3;padding:8px 0;">'
                f'✓ Orçamento aprovado: <b>{st.session_state.approved_supplier}</b> — '
                f'A coluna "Preço Autorizado" será preenchida automaticamente no Excel.</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
        with st.expander("Adicionar item manualmente (e-commerce / 4º fornecedor)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            new_name  = c1.text_input("Nome do item")
            new_marca = c2.text_input("Marca")
            new_qtd   = c3.number_input("Qtd", min_value=0.0, step=1.0)
            new_und   = c4.selectbox("UND", options=ALLOWED_UNITS, index=0)
            price_cols = st.columns(len(active_suppliers))
            new_prices = {}
            for i, sname in enumerate(active_suppliers):
                new_prices[sname] = price_cols[i].number_input(f"Preço {sname}", min_value=0.0, format="%.2f", key=f"np_{i}")
            new_obs = st.text_input("Observação", key="new_obs")
            if st.button("Adicionar item") and new_name:
                new_row = {
                    "ID": len(edited_df) + 1, "Item": new_name.upper(), "Marca": new_marca,
                    "Qtd": new_qtd, "UND": new_und, "⚠": "", "Observação": new_obs,
                }
                for sname in active_suppliers:
                    new_row[f"R$ {sname}"] = new_prices[sname] if new_prices[sname] > 0 else None
                edited_df = pd.concat([edited_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"'{new_name}' adicionado.")

        def df_to_items(df, sup_names):
            result = []
            for _, row in df.iterrows():
                if not row.get("Item"):
                    continue
                forn_dict = {}
                for sname in sup_names:
                    price = row.get(f"R$ {sname}")
                    forn_dict[sname] = {
                        "preco_unit": float(price) if price and not pd.isna(price) else None,
                        "obs": None,
                    }
                und = str(row.get("UND") or "UN").upper()
                if und not in ALLOWED_UNITS:
                    und = "UN"
                result.append({
                    "id":           int(row.get("ID") or len(result) + 1),
                    "item":         str(row.get("Item", "")),
                    "marca":        row.get("Marca") or None,
                    "quantidade":   float(row.get("Qtd") or 1),
                    "unidade":      und,
                    "fornecedores": forn_dict,
                    "observacao":   row.get("Observação") or None,
                })
            return result

        with st.expander("🧠 Correções capturadas nesta sessão", expanded=False):
            ai_items = st.session_state.normalized_items
            corrections_so_far = detect_corrections(
                ai_items, df_to_items(edited_df, active_suppliers), active_suppliers
            )
            if corrections_so_far:
                for c in corrections_so_far:
                    st.markdown(
                        f'<div style="font-size:0.82rem;padding:4px 0;border-bottom:1px solid rgba(128,128,128,0.1);">'
                        f'<b>{c["type"]}</b> · {c["note"]}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("Nenhuma diferença detectada em relação ao output da IA.")

        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_fwd = st.columns([1, 4])
        with col_back:
            if st.button("← Voltar"):
                st.session_state.step = 2
                st.rerun()
        with col_fwd:
            if st.button("Gerar Excel →", type="primary"):
                final = df_to_items(edited_df, active_suppliers)
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


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 4 · Download
# ─────────────────────────────────────────────────────────────────────────────
elif step == 4:
    st.markdown('<div class="section-eyebrow">Passo 4 de 4</div><div class="section-title">Mapa de Compras pronto</div>', unsafe_allow_html=True)

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

                filename = f"MapaCompras_{filial.replace(' ','_')}_{data_compra.strftime('%d%m%Y')}.xlsx"

                total_menor = sum(
                    min(
                        [item.get("fornecedores", {}).get(s, {}).get("preco_unit") or 999999 for s in active_suppliers]
                    ) * item["quantidade"]
                    for item in final_items
                    if any(item.get("fornecedores", {}).get(s, {}).get("preco_unit") for s in active_suppliers)
                )

                total_autorizado = None
                if approved_supplier:
                    total_autorizado = sum(
                        (item.get("fornecedores", {}).get(approved_supplier, {}).get("preco_unit") or 0) * item["quantidade"]
                        for item in final_items
                    )

                st.markdown(f"""
                <div class="glass-card" style="margin-bottom:1.5rem;">
                    <div style="display:flex;gap:40px;align-items:center;flex-wrap:wrap;">
                        <div>
                            <div class="supplier-label">Itens comparados</div>
                            <div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;">{len(final_items)}</div>
                        </div>
                        <div>
                            <div class="supplier-label">Fornecedores</div>
                            <div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;">{len(active_suppliers)}</div>
                        </div>
                        <div>
                            <div class="supplier-label">Total (menor preço)</div>
                            <div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#1A7F37;">R$ {total_menor:,.2f}</div>
                        </div>
                        {f'''<div>
                            <div class="supplier-label">Total Autorizado ({approved_supplier})</div>
                            <div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#0071E3;">R$ {total_autorizado:,.2f}</div>
                        </div>''' if total_autorizado is not None else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
                        for k in ["step", "supplier_data", "normalized_items", "edited_items", "final_items", "uploaded_files"]:
                            st.session_state[k] = 1 if k == "step" else ({} if k in ["supplier_data", "uploaded_files"] else [])
                        st.rerun()

                prefs_now    = st.session_state.get("preferences", {})
                n_corr_total = len(prefs_now.get("corrections", []))
                if n_corr_total > 0:
                    st.markdown(
                        f'<div style="font-size:0.85rem;color:#1A7F37;padding:8px 0;">'
                        f'🧠 {n_corr_total} preferência(s) salva(s) automaticamente — '
                        f'serão aplicadas na próxima extração.</div>',
                        unsafe_allow_html=True
                    )

                st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-eyebrow" style="margin-top:1rem;">Preview</div>', unsafe_allow_html=True)

                preview_rows = []
                for item in final_items:
                    row = {"#": item["id"], "Item": item["item"], "Qtd": item["quantidade"], "UND": item["unidade"]}
                    prices = []
                    for s in active_suppliers:
                        p = item.get("fornecedores", {}).get(s, {}).get("preco_unit")
                        row[s] = f"R$ {p:,.2f}" if p else "—"
                        if p: prices.append(p)
                    row["✦ Menor"] = f"R$ {min(prices):,.2f}" if prices else "—"
                    row["Total"]   = f"R$ {item['quantidade'] * min(prices):,.2f}" if prices else "—"
                    if approved_supplier:
                        ap = item.get("fornecedores", {}).get(approved_supplier, {}).get("preco_unit")
                        row["P. Autorizado"] = f"R$ {ap:,.2f}" if ap else "—"
                    preview_rows.append(row)

                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Erro ao gerar Excel: {e}")
                st.exception(e)

    if st.button("← Voltar para revisão"):
        st.session_state.step = 3
        st.rerun()
