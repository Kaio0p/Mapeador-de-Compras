# -*- coding: utf-8 -*-
"""
streamlit_app.py -- Gerador de Mapa de Compras (Arquitetura Multi-Agente)
=========================================================================
Design: Apple HIG - Liquid Glass - SF Pro system stack

Fluxo de agentes:
  PDF nativo    -> Groq LPU  (extract_items_from_text   -- 128k context, ultra-rapido)
  PDF escaneado -> Gemini    (extract_items_from_images -- OCR multi-chave)
  Imagem direta -> Gemini    (extract_items_from_jpeg/png_images)
  Normalizacao  -> Groq LPU  (normalize_and_match       -- fusao + Regra de Proporcao)
  Auditoria     -> Gemini    (audit_purchase_map         -- cross-reference com originais)
  Revisao       -> Humano    (st.data_editor             -- Human-in-the-Loop)
  Download      -> Excel     (generate_excel)
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

# -- Importacoes dos modulos --------------------------------------------------
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
from modules.groq_processor import (
    extract_items_from_text,
    normalize_and_match,
)
from modules.llm_manager import (
    get_system_status,
    configure_gemini_with_key,
    get_random_gemini_key,
    get_groq_client,
)
from modules.excel_generator import generate_excel
from modules.preferences_manager import (
    detect_corrections, merge_corrections, build_prompt_context,
    load_from_supabase, save_to_supabase,
    load_catalog_from_supabase,
)

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Mapa de Compras - EBD",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS: Apple Liquid Glass --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display',
                 'DM Sans', 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* == BASE - MODO CLARO == */
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
.stDownloadButton > button {
    background: linear-gradient(180deg, #34C759 0%, #28A745 100%) !important;
    color: #fff !important; border: none !important; border-radius: 980px !important;
    padding: 10px 26px !important; font-weight: 600 !important; font-size: 0.9rem !important;
    box-shadow: 0 2px 14px rgba(40,167,69,0.38) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.6) !important; backdrop-filter: blur(16px) !important;
    border: 1.5px dashed rgba(0,113,227,0.25) !important; border-radius: 14px !important;
    padding: 1.2rem !important;
}
.stAlert { border-radius: 12px !important; border: none !important; }
div[data-testid="stAlert"] { background: rgba(0,113,227,0.07) !important; border-left: 3px solid #0071E3 !important; border-radius: 12px !important; }
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.75) !important; border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.07) !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 12px 16px !important;
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
[data-baseweb="select"] > div { border-radius: 10px !important; border-color: rgba(0,0,0,0.1) !important; background: rgba(255,255,255,0.85) !important; }

/* == COMPONENTES CUSTOMIZADOS == */
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
    font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
}
.step-dot.done   { background: #34C759; color: #fff; }
.step-dot.active { background: #0071E3; color: #fff; box-shadow: 0 0 0 4px rgba(0,113,227,0.2); }
.step-dot.idle   { background: rgba(0,0,0,0.08); color: #8E8E93; }
.step-text .num { font-weight: 600; color: #1C1C1E; font-size: 0.82rem; }
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
.agent-groq   { background: rgba(249,115,22,0.1); border: 1px solid rgba(249,115,22,0.25); color: #C2410C; }
.agent-gemini { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25); color: #1D4ED8; }
.agent-audit  { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.25);  color: #B91C1C; }

/* == OVERRIDES MODO ESCURO == */
.stApp[data-theme="dark"] { background: #1C1C1E !important; }
[data-theme="dark"] [data-testid="stSidebar"] { background: rgba(28,28,30,0.92) !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }
[data-theme="dark"] h1, [data-theme="dark"] h2 { color: #F5F5F7 !important; }
[data-theme="dark"] h3 { color: #E5E5EA !important; }
[data-theme="dark"] p, [data-theme="dark"] label, [data-theme="dark"] .stMarkdown { color: #AEAEB2 !important; }
[data-theme="dark"] .stTextInput input, [data-theme="dark"] .stTextArea textarea {
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
[data-theme="dark"] .streamlit-expanderContent { background: rgba(44,44,46,0.5) !important; }
[data-theme="dark"] .glass-card { background: rgba(44,44,46,0.75) !important; border: 1px solid rgba(255,255,255,0.08) !important; }
[data-theme="dark"] .glass-card .supplier-name  { color: #F5F5F7 !important; }
[data-theme="dark"] .section-title { color: #F5F5F7 !important; }
[data-theme="dark"] .step-dot.idle { background: rgba(255,255,255,0.1) !important; }
[data-theme="dark"] .step-text .num { color: #E5E5EA !important; }
[data-theme="dark"] .suspect-card { background: rgba(255,149,0,0.05) !important; }
[data-theme="dark"] .suspect-card .suspect-title  { color: #FF9F0A !important; }
[data-theme="dark"] .suspect-card .suspect-reason { color: #FFCC80 !important; }
[data-theme="dark"] .agent-groq   { background: rgba(249,115,22,0.08) !important; color: #FB923C !important; }
[data-theme="dark"] .agent-gemini { background: rgba(96,165,250,0.08) !important; color: #60A5FA !important; }
[data-theme="dark"] .agent-audit  { background: rgba(248,113,113,0.08) !important; color: #F87171 !important; }
</style>
""", unsafe_allow_html=True)


# -- State init ----------------------------------------------------------------
def init_state():
    defaults = {
        "step": 1,
        "supplier_data": {},
        "normalized_items": [],
        "edited_items": [],
        "api_key_ok": False,
        "groq_ok": False,
        "preferences": {"corrections": [], "version": 1},
        "preferences_context": "",
        "prefs_loaded": False,
        "catalog": [],
        "catalog_loaded": False,
        "approved_supplier": None,
        # Guarda os textos originais dos PDFs nativos para a auditoria do Gemini
        "original_texts": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# -- Inicializacao das APIs ---------------------------------------------------
def _get_sb_creds():
    try:
        return st.secrets.get("SUPABASE_URL", ""), st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        return "", ""


def _init_apis():
    status = get_system_status()

    # Gemini
    if not st.session_state.api_key_ok and status["gemini_configured"]:
        key = get_random_gemini_key()
        if key:
            try:
                configure_gemini_with_key(key)
                st.session_state.api_key_ok = True
            except Exception:
                pass

    # Groq
    if not st.session_state.groq_ok and status["groq_configured"]:
        try:
            get_groq_client()
            st.session_state.groq_ok = True
        except Exception:
            pass

    return status


_system_status = _init_apis()

# -- Auto-carrega preferencias e catalogo do Supabase -------------------------
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


# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px; border-bottom:1px solid rgba(128,128,128,0.15); margin-bottom:4px;">
        <div style="font-size:1.15rem;font-weight:700;letter-spacing:-0.03em;">🗂 Mapa de Compras</div>
        <div style="font-size:0.72rem;color:#8E8E93;margin-top:2px;">GRUPO EBD - DEPTO. COMPRAS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Agentes IA")

    groq_color = "#1A7F37" if st.session_state.groq_ok else "#C0392B"
    groq_label = "Groq conectado (texto+norm.)" if st.session_state.groq_ok else "Groq nao configurado"
    st.markdown(
        '<div style="font-size:0.8rem;color:{};font-weight:500;padding:4px 0 2px;">● {}</div>'.format(
            groq_color, groq_label
        ),
        unsafe_allow_html=True,
    )
    if not st.session_state.groq_ok:
        st.caption("Adicione GROQ_API_KEY em `.streamlit/secrets.toml`")

    n_keys    = _system_status.get("gemini_key_count", 0)
    gem_color = "#1A7F37" if st.session_state.api_key_ok else "#C0392B"
    gem_label = (
        "Gemini Vision+Auditoria ({} chave{})".format(n_keys, "s" if n_keys != 1 else "")
        if st.session_state.api_key_ok
        else "Gemini nao configurado"
    )
    st.markdown(
        '<div style="font-size:0.8rem;color:{};font-weight:500;padding:2px 0 8px;">● {}</div>'.format(
            gem_color, gem_label
        ),
        unsafe_allow_html=True,
    )
    if not st.session_state.api_key_ok:
        st.caption("Adicione GEMINI_API_KEYS em `.streamlit/secrets.toml`")

    n_corr = len(st.session_state.get("preferences", {}).get("corrections", []))
    if n_corr > 0:
        st.markdown(
            '<div style="font-size:0.78rem;color:#636366;padding:2px 0 8px;">🧠 {} correcao(oes) ativa(s)</div>'.format(n_corr),
            unsafe_allow_html=True,
        )

    st.markdown("### Cabecalho")
    numero_seq  = st.text_input("No. Sequencial", value="2026001001")
    filial      = st.text_input("Filial", value="Sao Goncalo")
    responsavel = st.text_input("Responsavel", value="")
    data_compra = st.date_input("Data", value=date.today())

    st.markdown("### Fornecedores")
    n_suppliers = st.radio("Quantidade", [3, 4], horizontal=True)

    supplier_names = []
    for i in range(n_suppliers):
        name = st.text_input("Fornecedor {}".format(i + 1), key="sup_name_{}".format(i),
                             placeholder="Nome do fornecedor {}".format(i + 1))
        supplier_names.append(name)

    st.markdown("### Orcamento Aprovado")
    active_suppliers_sidebar = [s for s in supplier_names if s]
    approved_options  = ["Nenhum (nao preencher)"] + active_suppliers_sidebar
    approved_selection = st.selectbox(
        "Fornecedor aprovado", options=approved_options, index=0,
        help="Se selecionado, preenche 'Preco Autorizado' no Excel automaticamente."
    )
    if approved_selection == "Nenhum (nao preencher)":
        st.session_state.approved_supplier = None
    else:
        st.session_state.approved_supplier = approved_selection

    st.markdown(
        '<div style="margin-top:2rem;font-size:0.72rem;color:#8E8E93;text-align:center;">v3.0 - Multi-Agente - 2026</div>',
        unsafe_allow_html=True,
    )


# -- Helpers UI ---------------------------------------------------------------
step = st.session_state.step


def step_tracker():
    steps = [("Upload", "PDFs/Imgs"), ("Extracao", "IA"), ("Revisao", "Itens"), ("Download", "Excel")]
    parts = []
    for i, (label, sub) in enumerate(steps, 1):
        if i < step:    cls = "done";   icon = "✓"
        elif i == step: cls = "active"; icon = str(i)
        else:           cls = "idle";   icon = str(i)
        parts.append(
            '<div class="step-item">'
            '<div class="step-dot {}">{}</div>'
            '<div class="step-text"><div class="num">{}</div><div class="sub">{}</div></div>'
            '</div>'.format(cls, icon, label, sub)
        )
    st.markdown('<div class="step-track">{}</div>'.format("".join(parts)), unsafe_allow_html=True)


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def _agent_badge(agent: str) -> str:
    badges = {
        "groq":   '<span class="agent-badge agent-groq">⚡ Groq</span>',
        "gemini": '<span class="agent-badge agent-gemini">👁 Gemini Vision</span>',
        "audit":  '<span class="agent-badge agent-audit">🔍 Auditor Gemini</span>',
    }
    return badges.get(agent, "")


# -- Page header --------------------------------------------------------------
st.markdown("""
<div class="page-header">
    <h1>Gerador de Mapa de Compras</h1>
    <div class="subtitle">
        PDFs nativos via Groq (128k) · PDFs escaneados e imagens via Gemini Vision ·
        Auditoria cross-reference via Gemini · Revise e aprove · Baixe o Excel
    </div>
</div>
""", unsafe_allow_html=True)

step_tracker()


# ============================================================================
# PASSO 1 - Upload
# ============================================================================
if step == 1:
    st.markdown(
        '<div class="section-eyebrow">Passo 1 de 4</div>'
        '<div class="section-title">Upload dos orcamentos</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Carregue um PDF ou imagem (PNG/JPEG) por fornecedor.  \n"
        "**PDFs nativos** sao processados pelo **Groq** (ultra-rapido, 128k tokens).  \n"
        "**PDFs escaneados e imagens** sao processados pelo **Gemini Vision** (OCR)."
    )

    uploaded_files = {}
    any_uploaded   = False

    for i in range(n_suppliers):
        sname = supplier_names[i] or "Fornecedor {}".format(i + 1)
        st.markdown(
            '<div class="glass-card">'
            '<div class="supplier-label">Orcamento {}</div>'
            '<div class="supplier-name">{}</div>'
            '</div>'.format(i + 1, sname),
            unsafe_allow_html=True,
        )

        f = st.file_uploader(
            "", type=["pdf", "png", "jpg", "jpeg"],
            key="file_{}".format(i), label_visibility="collapsed",
        )
        if f:
            raw = f.read()
            if isinstance(raw, str):
                raw = raw.encode("latin-1")
            is_img = _is_image_file(f.name)
            uploaded_files[sname] = {"bytes": raw, "name": f.name, "is_image": is_img}

            if is_img:
                st.markdown(
                    '<div class="img-badge">🖼 {} &nbsp;·&nbsp; imagem {}</div>'.format(
                        f.name, _agent_badge("gemini")
                    ),
                    unsafe_allow_html=True,
                )
            else:
                pages = get_pdf_page_count(raw)
                st.markdown(
                    '<div class="success-pill">✓ {} &nbsp;·&nbsp; {} pag.</div>'.format(
                        f.name, pages
                    ),
                    unsafe_allow_html=True,
                )
            any_uploaded = True

        st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="ref-label">Lista de referencia (opcional)</div>', unsafe_allow_html=True)
    st.caption("Cole os itens que voce quer comprar. Ex: HIPOCLORITO 5% 5L, 2 BB")
    ref_text = st.text_area(
        "", height=110,
        placeholder="DETERGENTE GOLD 5L, 1 BB\nSACO DE LIXO 100L, 6 PCT\n...",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button(
            "Avancar para extracao →",
            type="primary", disabled=not any_uploaded, use_container_width=True,
        ):
            st.session_state.uploaded_files = uploaded_files
            st.session_state.ref_text = ref_text
            st.session_state.step = 2
            st.rerun()


# ============================================================================
# PASSO 2 - Extracao IA (Fluxo Maestro Multi-Agente)
# ============================================================================
elif step == 2:
    st.markdown(
        '<div class="section-eyebrow">Passo 2 de 4</div>'
        '<div class="section-title">Extracao, Normalizacao e Auditoria via IA</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.groq_ok and not st.session_state.api_key_ok:
        st.error(
            "Nenhum agente de IA configurado. "
            "Adicione GROQ_API_KEY e/ou GEMINI_API_KEYS ao arquivo `.streamlit/secrets.toml`."
        )
        st.stop()

    if not st.session_state.groq_ok:
        st.warning(
            "GROQ_API_KEY nao configurada. PDFs nativos nao poderao ser processados. "
            "Apenas OCR via Gemini Vision estara disponivel."
        )

    uploaded_files = st.session_state.get("uploaded_files", {})
    ref_text       = st.session_state.get("ref_text", "")

    # Parse da lista de referencia
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
            "Iniciar extracao com IA →", type="primary", use_container_width=True
        )

    if run_extraction:
        supplier_items = {}
        # original_texts guarda o texto de PDFs nativos para a auditoria do Gemini
        original_texts = {}
        prefs_ctx      = st.session_state.get("preferences_context", "")

        with st.status("Analisando orcamentos...", expanded=True) as status_box:

            # ----------------------------------------------------------------
            # ETAPA 1 - Extracao por fornecedor (roteamento inteligente)
            # ----------------------------------------------------------------
            for supplier_name, finfo in uploaded_files.items():
                pdf_bytes = finfo["bytes"]
                fname     = finfo["name"]
                is_image  = finfo["is_image"]

                try:
                    if is_image:
                        # Imagem direta -> Gemini Vision (OCR)
                        mime = detect_image_mime(pdf_bytes)
                        st.write(
                            "🖼 {} Processando imagem **{}** de **{}**...".format(
                                _agent_badge("gemini"), fname, supplier_name
                            )
                        )
                        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                        if mime == "image/jpeg":
                            items = extract_items_from_jpeg_images(
                                [b64], preferences_context=prefs_ctx
                            )
                        else:
                            items = extract_items_from_images(
                                [b64], preferences_context=prefs_ctx
                            )

                    else:
                        # PDF - detecta se e nativo ou escaneado
                        st.write("📄 Lendo PDF de **{}**...".format(supplier_name))
                        text, is_img_pdf = extract_text_from_pdf(pdf_bytes)

                        if is_img_pdf:
                            # PDF escaneado -> Gemini Vision (OCR)
                            st.write(
                                "🔍 {} PDF escaneado detectado - enviando para OCR...".format(
                                    _agent_badge("gemini")
                                )
                            )
                            images = extract_images_from_pdf(pdf_bytes)
                            items  = extract_items_from_images(
                                images, preferences_context=prefs_ctx
                            )
                            # Para PDFs escaneados guardamos descricao breve (sem texto real)
                            original_texts[supplier_name] = (
                                "[PDF escaneado - texto extraido via OCR Gemini Vision]"
                            )
                        else:
                            # PDF nativo -> Groq LPU (128k context, ultra-rapido)
                            if not st.session_state.groq_ok:
                                st.error(
                                    "PDF nativo de **{}** requer Groq (nao configurado). "
                                    "Configure GROQ_API_KEY.".format(supplier_name)
                                )
                                continue
                            st.write(
                                "⚡ {} Extraindo itens de **{}** via Groq (128k)...".format(
                                    _agent_badge("groq"), supplier_name
                                )
                            )
                            items = extract_items_from_text(
                                text, preferences_context=prefs_ctx
                            )
                            # Guarda texto original para a auditoria do Gemini
                            original_texts[supplier_name] = text

                    supplier_items[supplier_name] = items

                    n_suspect = sum(1 for it in items if it.get("is_suspect"))
                    msg = "✅ **{}** -- {} item(s) extraido(s)".format(supplier_name, len(items))
                    if n_suspect:
                        msg += " · ⚠️ {} suspeito(s)".format(n_suspect)
                    st.write(msg)

                    with st.expander("Ver itens brutos de {}".format(supplier_name), expanded=False):
                        st.json(items)

                except RuntimeError as e:
                    status_box.update(label="Limite de requisicoes atingido", state="error")
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error("Erro ao processar {}: {}".format(supplier_name, e))

            if not supplier_items:
                status_box.update(label="Nenhum item extraido", state="error")
                st.warning("Nenhum item foi extraido. Verifique os arquivos e tente novamente.")
                st.stop()

            # ----------------------------------------------------------------
            # ETAPA 2 - Normalizacao e Fuzzy Match via Groq LPU
            # ----------------------------------------------------------------
            st.write(
                "⚖️ {} Cruzando e normalizando itens entre {} fornecedor(es)... "
                "_(pode levar ate 30s)_".format(
                    _agent_badge("groq"), len(supplier_items)
                )
            )
            try:
                normalized = normalize_and_match(
                    supplier_items,
                    reference_list or None,
                    preferences_context=prefs_ctx,
                    catalog=st.session_state.get("catalog") or None,
                )
            except Exception as e:
                status_box.update(label="Erro na normalizacao", state="error")
                st.error("Erro na normalizacao: {}".format(e))
                st.stop()

            # ----------------------------------------------------------------
            # ETAPA 3 - Auditoria Final via Gemini (cross-reference com originais)
            # ----------------------------------------------------------------
            st.write(
                "🔍 {} Auditando mapa -- cruzando com textos originais dos orcamentos...".format(
                    _agent_badge("audit")
                )
            )
            normalized = audit_purchase_map(normalized, original_texts=original_texts)

            # ----------------------------------------------------------------
            # Finaliza
            # ----------------------------------------------------------------
            st.session_state.supplier_data    = supplier_items
            st.session_state.normalized_items = normalized
            st.session_state.edited_items     = [dict(x) for x in normalized]
            st.session_state.original_texts   = original_texts

            n_suspect_total = sum(1 for it in normalized if it.get("is_suspect"))
            label_done = "✅ {} itens prontos para revisao".format(len(normalized))
            if n_suspect_total:
                label_done += " · 🚨 {} item(s) para revisar".format(n_suspect_total)
            status_box.update(label=label_done, state="complete", expanded=False)
            st.session_state.step = 3
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Voltar"):
        st.session_state.step = 1
        st.rerun()


# ============================================================================
# PASSO 3 - Revisao Human-in-the-Loop
# ============================================================================
elif step == 3:
    st.markdown(
        '<div class="section-eyebrow">Passo 3 de 4</div>'
        '<div class="section-title">Revisao e Aprovacao -- Human-in-the-Loop</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "📋 Revise os itens abaixo. Linhas marcadas com 🚨 foram sinalizadas pelo "
        "**Auditor Gemini** (cross-reference com os documentos originais). "
        "Edite diretamente na tabela e clique em **Aprovar e Gerar Excel** quando estiver satisfeito."
    )

    items = st.session_state.normalized_items

    if not items:
        st.warning("Nenhum item encontrado. Volte e processe os arquivos novamente.")
    else:
        # Painel de alertas do Auditor
        suspect_items = [it for it in items if it.get("is_suspect")]
        if suspect_items:
            with st.expander(
                "🚨 {} alerta(s) do Auditor Gemini -- clique para revisar".format(len(suspect_items)),
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
                alert_col = "🚨 " + (r0[:60] + "..." if len(r0) > 60 else r0)
            elif is_suspect:
                alert_col = "🚨 Verificar"
            else:
                alert_col = ""

            row = {
                "ID":          item.get("id", ""),
                "Item":        item.get("item", ""),
                "Marca":       item.get("marca") or "",
                "Qtd":         item.get("quantidade", 1),
                "UND":         item.get("unidade", "UN"),
                "🚨 Alerta":   alert_col,
            }
            for sname in active_suppliers:
                fdata = item.get("fornecedores", {}).get(sname, {})
                row["R$ {}".format(sname)] = fdata.get("preco_unit") if fdata else None
            row["Observacao"] = item.get("observacao") or ""
            rows.append(row)

        col_cfg = {
            "ID":   st.column_config.NumberColumn("ID", min_value=1, step=1, width="small"),
            "Item": st.column_config.TextColumn("Item", width="large"),
            "Marca":st.column_config.TextColumn("Marca", width="medium"),
            "Qtd":  st.column_config.NumberColumn("Qtd", min_value=0, step=0.5, width="small"),
            "UND":  st.column_config.SelectboxColumn("UND", options=ALLOWED_UNITS, width="small"),
            "🚨 Alerta": st.column_config.TextColumn("🚨 Alerta do Auditor", width="large", disabled=True),
            "Observacao": st.column_config.TextColumn("Obs", width="medium"),
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
                '✓ Orcamento aprovado: <b>{}</b> -- '
                'A coluna "Preco Autorizado" sera preenchida automaticamente no Excel.</div>'.format(
                    st.session_state.approved_supplier
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

        # Adicionar item manualmente
        with st.expander("Adicionar item manualmente (e-commerce / 4o fornecedor)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            new_name  = c1.text_input("Nome do item")
            new_marca = c2.text_input("Marca")
            new_qtd   = c3.number_input("Qtd", min_value=0.0, step=1.0)
            new_und   = c4.selectbox("UND", options=ALLOWED_UNITS, index=0)
            price_cols = st.columns(len(active_suppliers))
            new_prices = {}
            for i, sname in enumerate(active_suppliers):
                new_prices[sname] = price_cols[i].number_input(
                    "Preco {}".format(sname), min_value=0.0, format="%.2f", key="np_{}".format(i)
                )
            new_obs = st.text_input("Observacao", key="new_obs")
            if st.button("Adicionar item") and new_name:
                new_row = {
                    "ID": len(edited_df) + 1, "Item": new_name.upper(),
                    "Marca": new_marca, "Qtd": new_qtd, "UND": new_und,
                    "🚨 Alerta": "", "Observacao": new_obs,
                }
                for sname in active_suppliers:
                    new_row["R$ {}".format(sname)] = (
                        new_prices[sname] if new_prices[sname] > 0 else None
                    )
                edited_df = pd.concat([edited_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success("'{}' adicionado.".format(new_name))

        # Converter DataFrame para lista de itens
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
                    "observacao":   row.get("Observacao") or None,
                })
            return result

        # Painel de correcoes capturadas
        with st.expander("🧠 Correcoes capturadas nesta sessao", expanded=False):
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
                st.caption("Nenhuma diferenca detectada em relacao ao output da IA.")

        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_fwd = st.columns([1, 4])
        with col_back:
            if st.button("← Voltar"):
                st.session_state.step = 2
                st.rerun()
        with col_fwd:
            if st.button("✅ Aprovar e Gerar Excel →", type="primary"):
                final = df_to_items(edited_df, active_suppliers)

                # Salva correcoes do usuario para aprendizado futuro
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


# ============================================================================
# PASSO 4 - Download
# ============================================================================
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
        with st.spinner("Gerando planilha..."):
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
                    '<div><div class="supplier-label">Total (menor preco)</div>'
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
                                  "edited_items", "final_items", "uploaded_files",
                                  "original_texts"]:
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
                        '🧠 {} preferencia(s) salva(s) automaticamente -- '
                        'serao aplicadas na proxima extracao.</div>'.format(n_corr_total),
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

    if st.button("← Voltar para revisao"):
        st.session_state.step = 3
        st.rerun()
