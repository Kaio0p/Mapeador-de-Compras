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

# ── CSS: Apple Liquid Glass ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display',
                 'DM Sans', 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ══ BASE — MODO CLARO ══ */
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

.stAlert { border-radius: 12px !important; border: none !important; }
div[data-testid="stAlert"] { background: rgba(0,113,227,0.07) !important; border-left: 3px solid #0071E3 !important; border-radius: 12px !important; }

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.75) !important; border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.07) !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 12px 16px !important; transition: background 0.2s !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.5) !important; border: 1px solid rgba(0,0,0,0.06) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 16px !important;
}

.stProgress > div > div { background: linear-gradient(90deg, #0071E3, #34C759) !important; border-radius: 999px !important; }
.stProgress > div { background: rgba(0,0,0,0.06) !important; border-radius: 999px !important; height: 4px !important; }

[data-testid="stDataFrame"], iframe {
    border-radius: 12px !important; overflow: hidden !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important; border: 1px solid rgba(0,0,0,0.06) !important;
}
.stSpinner > div { color: #0071E3 !important; }
[data-baseweb="select"] > div { border-radius: 10px !important; border-color: rgba(0,0,0,0.1) !important; background: rgba(255,255,255,0.85) !important; }

/* ══ COMPONENTES CUSTOMIZADOS ══ */
.glass-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 18px; padding: 20px 24px; margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07), 0 1px 0 rgba(255,255,255,0.9) inset;
    transition: box-shadow 0.25s ease, transform 0.2s ease;
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

.agent-badge {
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: 980px; padding: 3px 10px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
    margin-right: 6px; vertical-align: middle;
}
.agent-gemini { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25); color: #1D4ED8; }
.agent-cohere { background: rgba(249,115,22,0.1); border: 1px solid rgba(249,115,22,0.25); color: #C2410C; }
.agent-audit  { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.25);  color: #B91C1C; }

/* ══ OVERRIDES MODO ESCURO ══ */
.stApp[data-theme="dark"] { background: #1C1C1E !important; }
[data-theme="dark"] [data-testid="stSidebar"] { background: rgba(28,28,30,0.92) !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }
[data-theme="dark"] h1, [data-theme="dark"] h2 { color: #F5F5F7 !important; }
[data-theme="dark"] h3 { color: #E5E5EA !important; }
[data-theme="dark"] p, [data-theme="dark"] label, [data-theme="dark"] .stMarkdown { color: #AEAEB2 !important; }
[data-theme="dark"] .stCaption, [data-theme="dark"] small { color: #636366 !important; }
[data-theme="dark"] .stTextInput input, [data-theme="dark"] .stDateInput input,
[data-theme="dark"] .stNumberInput input, [data-theme="dark"] .stTextArea textarea {
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
[data-theme="dark"] .section-title { color: #F5F5F7 !important; }
[data-theme="dark"] .page-header .subtitle { color: #8E8E93 !important; }
[data-theme="dark"] .apple-divider { background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important; }
[data-theme="dark"] .ref-label { color: #636366 !important; }
[data-theme="dark"] .suspect-card { background: rgba(255,149,0,0.05) !important; border-color: rgba(255,149,0,0.2) !important; }
[data-theme="dark"] .suspect-card .suspect-title  { color: #FF9F0A !important; }
[data-theme="dark"] .suspect-card .suspect-reason { color: #FFCC80 !important; }
[data-theme="dark"] .agent-gemini { background: rgba(96,165,250,0.08) !important; color: #60A5FA !important; }
[data-theme="dark"] .agent-cohere { background: rgba(249,115,22,0.08) !important; color: #FB923C !important; }
[data-theme="dark"] .agent-audit  { background: rgba(248,113,113,0.08) !important; color: #F87171 !important; }
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
    st.markdown("""
    <div style="padding:4px 0 16px; border-bottom:1px solid rgba(128,128,128,0.15); margin-bottom:4px;">
        <div style="font-size:1.15rem;font-weight:700;letter-spacing:-0.03em;">🗂 Mapa de Compras</div>
        <div style="font-size:0.72rem;color:#8E8E93;margin-top:2px;letter-spacing:0.02em;">GRUPO EBD · DEPTO. COMPRAS</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status dos agentes ─────────────────────────────────────────────────────
    st.markdown("### Agentes IA")

    # Gemini (extração + auditoria)
    n_keys    = _system_status.get("gemini_key_count", 0)
    gem_color = "#1A7F37" if st.session_state.api_key_ok else "#C0392B"
    gem_label = (
        "Gemini conectado ({} chave{})".format(n_keys, "s" if n_keys != 1 else "")
        if st.session_state.api_key_ok
        else "Gemini não configurado"
    )
    st.markdown(
        '<div style="font-size:0.8rem;color:{};font-weight:500;padding:4px 0 2px;">● {}</div>'.format(
            gem_color, gem_label
        ),
        unsafe_allow_html=True,
    )
    if not st.session_state.api_key_ok:
        st.caption("Adicione GEMINI_API_KEYS em `.streamlit/secrets.toml`")

    # Cohere (normalização)
    coh_color = "#1A7F37" if st.session_state.cohere_ok else "#C0392B"
    coh_label = "Cohere conectado (normalização)" if st.session_state.cohere_ok else "Cohere não configurado"
    st.markdown(
        '<div style="font-size:0.8rem;color:{};font-weight:500;padding:2px 0 8px;">● {}</div>'.format(
            coh_color, coh_label
        ),
        unsafe_allow_html=True,
    )
    if not st.session_state.cohere_ok:
        st.caption("Adicione COHERE_API_KEY em `.streamlit/secrets.toml`")

    # Preferências ativas
    n_corr = len(st.session_state.get("preferences", {}).get("corrections", []))
    if n_corr > 0:
        st.markdown(
            '<div style="font-size:0.78rem;color:#636366;padding:2px 0 8px;">🧠 {} correção(ões) ativa(s)</div>'.format(n_corr),
            unsafe_allow_html=True,
        )

    st.markdown("### Cabeçalho")
    numero_seq  = st.text_input("Nº Sequencial", value="2026001001")
    filial      = st.selectbox("Filial", options=_FILIAIS, index=2)
    responsavel = st.text_input("Responsável", value="")
    data_compra = st.date_input("Data", value=date.today())

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

    st.markdown(
        '<div style="margin-top:2rem;font-size:0.72rem;color:#8E8E93;text-align:center;">v3.0 · Multi-Agente · 2026</div>',
        unsafe_allow_html=True,
    )


# ── Helpers UI ────────────────────────────────────────────────────────────────
step = st.session_state.step


def step_tracker():
    steps = [("Upload", "PDFs/Imgs"), ("Extração", "IA"), ("Revisão", "Itens"), ("Download", "Excel")]
    parts = []
    for i, (label, sub) in enumerate(steps, 1):
        if i < step:    cls = "done";   icon = "✓"
        elif i == step: cls = "active"; icon = str(i)
        else:           cls = "idle";   icon = str(i)
        parts.append(
            '<div class="step-item">'
            '<div class="step-dot {cls}">{icon}</div>'
            '<div class="step-text"><div class="num">{label}</div><div class="sub">{sub}</div></div>'
            '</div>'.format(cls=cls, icon=icon, label=label, sub=sub)
        )
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


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Gerador de Mapa de Compras</h1>
    <div class="subtitle">
        Faça upload dos orçamentos · Gemini extrai · Cohere normaliza · Gemini audita · Você revisa · Baixe o Excel
    </div>
</div>
""", unsafe_allow_html=True)

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
        '<p style="color:#636366;font-size:0.88rem;margin-bottom:1.5rem;">'
        'Carregue um PDF ou imagem por fornecedor. '
        'PDFs escaneados são detectados automaticamente e enviados para OCR via Gemini Vision.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── CSS extra para os painéis de upload ──────────────────────────────────
    st.markdown("""
    <style>
    /* Painel de upload — container do Streamlit que envolve cada coluna */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        height: 100%;
    }

    /* Zona de drop grande dentro do painel */
    .upload-panel [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        backdrop-filter: none !important;
    }

    /* Dropzone em si — altura aumentada */
    [data-testid="stFileUploaderDropzone"] {
        min-height: 140px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(0,113,227,0.03) !important;
        border: 1.5px dashed rgba(0,113,227,0.22) !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(0,113,227,0.06) !important;
        border-color: rgba(0,113,227,0.5) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        font-size: 0.82rem !important;
        color: #8E8E93 !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 1.5rem !important;
        display: block !important;
        margin-bottom: 4px !important;
    }

    /* Cartão painel */
    .upanel {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255,255,255,0.9);
        border-radius: 20px;
        padding: 20px 20px 16px 20px;
        box-shadow: 0 4px 28px rgba(0,0,0,0.07), 0 1px 0 rgba(255,255,255,0.9) inset;
        margin-bottom: 4px;
        transition: box-shadow 0.25s ease, transform 0.2s ease;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .upanel:hover {
        box-shadow: 0 8px 36px rgba(0,0,0,0.1), 0 1px 0 rgba(255,255,255,0.9) inset;
        transform: translateY(-2px);
    }
    .upanel-eyebrow {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #0071E3;
        margin-bottom: 0;
        line-height: 1;
    }
    .upanel-name {
        font-size: 1.1rem;
        font-weight: 650;
        letter-spacing: -0.025em;
        color: #1C1C1E;
        margin: 0;
        line-height: 1.2;
    }
    .upanel-name.placeholder {
        color: #AEAEB2;
        font-weight: 400;
    }
    .upanel-status {
        font-size: 0.78rem;
        font-weight: 500;
        color: #1A7F37;
        background: rgba(52,199,89,0.1);
        border: 1px solid rgba(52,199,89,0.2);
        border-radius: 980px;
        padding: 3px 12px;
        display: inline-block;
        margin-top: 4px;
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

    # ── Grid: 3 colunas (ou 2+2 para 4 fornecedores) ────────────────────────
    if n_suppliers <= 3:
        cols = st.columns(n_suppliers, gap="medium")
        col_groups = [cols]
    else:
        # 4 fornecedores: 2 linhas de 2
        cols_top = st.columns(2, gap="medium")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        cols_bot = st.columns(2, gap="medium")
        col_groups = [cols_top, cols_bot]

    all_cols = []
    if n_suppliers <= 3:
        all_cols = col_groups[0]
    else:
        all_cols = list(col_groups[0]) + list(col_groups[1])

    for i in range(n_suppliers):
        sname     = supplier_names[i] or "Fornecedor {}".format(i + 1)
        has_name  = bool(supplier_names[i])
        name_cls  = "upanel-name" if has_name else "upanel-name placeholder"
        name_disp = sname if has_name else "Fornecedor {}".format(i + 1)

        with all_cols[i]:
            # Cabeçalho do painel em HTML puro
            st.markdown(
                '<div class="upanel">'
                '<div class="upanel-eyebrow">Orçamento {n}</div>'
                '<div class="{cls}">{name}</div>'
                '</div>'.format(n=i + 1, cls=name_cls, name=name_disp),
                unsafe_allow_html=True,
            )
            # File uploader logo abaixo — dentro da mesma coluna
            f = st.file_uploader(
                "Arraste ou clique para selecionar",
                type=["pdf", "png", "jpg", "jpeg"],
                key="file_{}".format(i),
                label_visibility="collapsed",
            )
            if f:
                raw = f.read()
                if isinstance(raw, str):
                    raw = raw.encode("latin-1")
                is_img = _is_image_file(f.name)
                uploaded_files[sname] = {"bytes": raw, "name": f.name, "is_image": is_img}

                if is_img:
                    st.markdown(
                        '<div class="upanel-status img">🖼 {} · imagem</div>'.format(f.name[:28]),
                        unsafe_allow_html=True,
                    )
                else:
                    pages = get_pdf_page_count(raw)
                    st.markdown(
                        '<div class="upanel-status">✓ {} · {} pág.</div>'.format(f.name[:26], pages),
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

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button(
            "Avançar para extração →",
            type="primary", disabled=not any_uploaded, use_container_width=True,
        ):
            st.session_state.uploaded_files = uploaded_files
            st.session_state.ref_text = ref_text
            st.session_state.step = 2
            st.rerun()


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

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_extraction = st.button(
            "Iniciar extração com IA →", type="primary", use_container_width=True
        )

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

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Voltar"):
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

        # Converter DataFrame → lista de itens
        def df_to_items(df, sup_names):
            result = []
            for _, row in df.iterrows():
                if not row.get("Item"):
                    continue
                forn_dict = {}
                for sname in sup_names:
                    price = row.get("R$ {}".format(sname))
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

        # Painel de correções capturadas
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

        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_fwd = st.columns([1, 4])
        with col_back:
            if st.button("← Voltar"):
                st.session_state.step = 2
                st.rerun()
        with col_fwd:
            if st.button("✅ Aprovar e Gerar Excel →", type="primary"):
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
                        '<div class="supplier-label">Total Autorizado ({})</div>'
                        '<div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#0071E3;">R$ {:,.2f}</div>'
                        '</div>'
                    ).format(approved_supplier, total_autorizado)

                st.markdown(
                    '<div class="glass-card" style="margin-bottom:1.5rem;">'
                    '<div style="display:flex;gap:40px;align-items:center;flex-wrap:wrap;">'
                    '<div><div class="supplier-label">Itens comparados</div>'
                    '<div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;">{}</div></div>'
                    '<div><div class="supplier-label">Fornecedores</div>'
                    '<div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;">{}</div></div>'
                    '<div><div class="supplier-label">Total (menor preço)</div>'
                    '<div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#1A7F37;">R$ {:,.2f}</div></div>'
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

    if st.button("← Voltar para revisão"):
        st.session_state.step = 3
        st.rerun()

