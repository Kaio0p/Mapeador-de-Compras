# -*- coding: utf-8 -*-
"""
streamlit_app.py — Gerador de Mapa de Compras
Design: Apple HIG · Liquid Glass · SF Pro system stack
"""
import sys, json
import streamlit as st
import pandas as pd
from datetime import date

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from modules.pdf_extractor   import extract_text_from_pdf, extract_images_from_pdf, get_pdf_page_count
from modules.gemini_processor import configure as configure_gemini, extract_items_from_text, extract_items_from_images, normalize_and_match
from modules.excel_generator  import generate_excel
from modules.preferences_manager import (
    detect_corrections, load_preferences, merge_corrections,
    preferences_to_json_bytes, build_prompt_context
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mapa de Compras · EBD",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apple Liquid Glass CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Reset global ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display',
                 'DM Sans', 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Background: Apple light ── */
.stApp {
    background: #F2F2F7 !important;
    min-height: 100vh;
}

/* ── Sidebar: frosted glass ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border-right: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 2px 0 32px rgba(0,0,0,0.05) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* ── Main block ── */
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Typography ── */
h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #1C1C1E !important;
    margin-bottom: 0.2rem !important;
    line-height: 1.15 !important;
}
h2 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #1C1C1E !important;
}
h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #3A3A3C !important;
}
p, label, .stMarkdown, .stCaption {
    color: #3A3A3C !important;
    font-size: 0.9rem !important;
    line-height: 1.55 !important;
}
.stCaption, small { color: #8E8E93 !important; font-size: 0.78rem !important; }

/* ── Sidebar labels / section headers ── */
[data-testid="stSidebar"] h3 {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8E8E93 !important;
    margin: 1.4rem 0 0.5rem !important;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(0,0,0,0.07) !important;
    margin: 1rem 0 !important;
}

/* ── Inputs ── */
.stTextInput input, .stDateInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 0.88rem !important;
    color: #1C1C1E !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #0071E3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15), 0 1px 3px rgba(0,0,0,0.04) !important;
    outline: none !important;
}

/* ── Buttons: Apple-style pill ── */
.stButton > button {
    background: rgba(255,255,255,0.9) !important;
    color: #0071E3 !important;
    border: 1px solid rgba(0,113,227,0.25) !important;
    border-radius: 980px !important;
    padding: 9px 22px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
    transition: all 0.18s cubic-bezier(0.25,0.46,0.45,0.94) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: rgba(0,113,227,0.06) !important;
    border-color: rgba(0,113,227,0.5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,113,227,0.18) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* Primary button: solid Apple blue */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, #1A8DFF 0%, #0071E3 100%) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 2px 12px rgba(0,113,227,0.38) !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(180deg, #1E96FF 0%, #0077ED 100%) !important;
    box-shadow: 0 4px 20px rgba(0,113,227,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(180deg, #34C759 0%, #28A745 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 980px !important;
    padding: 10px 26px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 2px 14px rgba(40,167,69,0.38) !important;
    transition: all 0.18s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(40,167,69,0.45) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(16px) !important;
    border: 1.5px dashed rgba(0,113,227,0.25) !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,113,227,0.55) !important;
    background: rgba(0,113,227,0.03) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: 0.85rem !important;
    color: #636366 !important;
}

/* ── Alerts / info boxes ── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stAlert"][kind="info"],
div[data-testid="stAlert"] {
    background: rgba(0,113,227,0.07) !important;
    border-left: 3px solid #0071E3 !important;
    border-radius: 12px !important;
}
div[data-baseweb="notification"] {
    border-radius: 12px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.75) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.07) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 12px 16px !important;
    transition: background 0.2s !important;
}
.streamlit-expanderHeader:hover {
    background: rgba(255,255,255,0.95) !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.5) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 16px !important;
}

/* ── Radio buttons ── */
.stRadio > div { gap: 8px !important; }
.stRadio label {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-size: 0.85rem !important;
    transition: all 0.15s !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #0071E3, #34C759) !important;
    border-radius: 999px !important;
}
.stProgress > div {
    background: rgba(0,0,0,0.06) !important;
    border-radius: 999px !important;
    height: 4px !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"], iframe {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
}

/* ── Spinner ── */
.stSpinner > div { color: #0071E3 !important; }

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: rgba(0,0,0,0.1) !important;
    background: rgba(255,255,255,0.85) !important;
}

/* ── Custom component classes ── */
.glass-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 18px;
    padding: 20px 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07), 0 1px 0 rgba(255,255,255,0.9) inset;
    margin-bottom: 12px;
    transition: box-shadow 0.25s ease, transform 0.2s ease;
}
.glass-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.1), 0 1px 0 rgba(255,255,255,0.9) inset;
    transform: translateY(-1px);
}
.glass-card .supplier-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #8E8E93;
    margin-bottom: 4px;
}
.glass-card .supplier-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1C1C1E;
    letter-spacing: -0.02em;
}

.success-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,199,89,0.12);
    border: 1px solid rgba(52,199,89,0.25);
    color: #1A7F37;
    border-radius: 980px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 8px;
}

/* Step tracker */
.step-track {
    display: flex;
    align-items: center;
    gap: 0;
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 14px;
    padding: 10px 20px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    position: relative;
}
.step-item:not(:last-child)::after {
    content: '';
    position: absolute;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 1px;
    height: 20px;
    background: rgba(0,0,0,0.1);
}
.step-dot {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
    transition: all 0.3s ease;
}
.step-dot.done   { background: #34C759; color: #fff; }
.step-dot.active { background: #0071E3; color: #fff; box-shadow: 0 0 0 4px rgba(0,113,227,0.2); }
.step-dot.idle   { background: rgba(0,0,0,0.08); color: #8E8E93; }
.step-text {
    font-size: 0.82rem;
    line-height: 1.2;
}
.step-text .num  { font-weight: 600; color: #1C1C1E; letter-spacing: -0.01em; }
.step-text .sub  { font-size: 0.72rem; color: #8E8E93; }

/* Header strip */
.page-header {
    margin-bottom: 1.6rem;
}
.page-header h1 { margin-bottom: 4px !important; }
.page-header .subtitle {
    font-size: 0.88rem;
    color: #636366;
    letter-spacing: -0.01em;
}

/* Section title */
.section-eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #0071E3;
    margin-bottom: 6px;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 650;
    letter-spacing: -0.025em;
    color: #1C1C1E;
    margin-bottom: 1rem;
}

/* Divider */
.apple-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.08), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* Ref textarea */
.ref-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8E8E93;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── State init ────────────────────────────────────────────────────────────────
def init_state():
    for k, v in {
        "step": 1, "supplier_data": {}, "normalized_items": [],
        "edited_items": [], "api_key_ok": False,
        "preferences": {"corrections": [], "version": 1},
        "preferences_context": "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px; border-bottom:1px solid rgba(0,0,0,0.07); margin-bottom:4px;">
        <div style="font-size:1.15rem;font-weight:700;letter-spacing:-0.03em;color:#1C1C1E;">🗂 Mapa de Compras</div>
        <div style="font-size:0.72rem;color:#8E8E93;margin-top:2px;letter-spacing:0.02em;">GRUPO EBD · DEPTO. COMPRAS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Gemini API")
    try:
        _default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        _default_key = ""

    api_key_input = st.text_input(
        "Chave API", type="password", value=_default_key,
        placeholder="AIza...",
        help="Obtenha grátis em aistudio.google.com"
    )
    if api_key_input:
        try:
            configure_gemini(api_key_input)
            st.markdown('<div style="font-size:0.8rem;color:#1A7F37;font-weight:500;margin-top:-6px;">● API conectada</div>', unsafe_allow_html=True)
            st.session_state.api_key_ok = True
        except Exception as e:
            st.markdown(f'<div style="font-size:0.8rem;color:#C0392B;font-weight:500;">● Erro: {e}</div>', unsafe_allow_html=True)
            st.session_state.api_key_ok = False

    st.markdown("### Cabeçalho")
    numero_seq  = st.text_input("Nº Sequencial", value="2026001001")
    filial      = st.text_input("Filial", value="São Gonçalo")
    responsavel = st.text_input("Responsável", value="")
    data_compra = st.date_input("Data", value=date.today())

    st.markdown("### Preferências")
    prefs_file = st.file_uploader(
        "Carregar histórico de correções (.json)",
        type=["json"], key="prefs_upload", label_visibility="collapsed",
        help="Arquivo gerado ao final de sessões anteriores"
    )
    if prefs_file:
        loaded = load_preferences(prefs_file.read())
        st.session_state.preferences = loaded
        st.session_state.preferences_context = build_prompt_context(loaded)
        n_corr = len(loaded.get("corrections", []))
        st.markdown(
            f'<div style="font-size:0.8rem;color:#1A7F37;font-weight:500;margin-top:-6px;">'
            f'● {n_corr} correção(ões) carregada(s)</div>',
            unsafe_allow_html=True
        )
    elif st.session_state.get("preferences", {}).get("corrections"):
        n_corr = len(st.session_state.preferences["corrections"])
        st.markdown(
            f'<div style="font-size:0.8rem;color:#636366;margin-top:-6px;">'
            f'● {n_corr} correção(ões) na sessão</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Fornecedores")
    n_suppliers = st.radio("Quantidade", [3, 4], horizontal=True)

    supplier_names = []
    for i in range(n_suppliers):
        name = st.text_input(f"Fornecedor {i+1}", key=f"sup_name_{i}",
                             placeholder=f"Nome do fornecedor {i+1}")
        supplier_names.append(name)

    st.markdown('<div style="margin-top:2rem;font-size:0.72rem;color:#C7C7CC;text-align:center;">v1.0 · 2026</div>', unsafe_allow_html=True)


# ── Helpers UI ────────────────────────────────────────────────────────────────
step = st.session_state.step

def step_tracker():
    steps = [("Upload", "PDFs"), ("Extração", "IA"), ("Revisão", "Itens"), ("Download", "Excel")]
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


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Gerador de Mapa de Compras</h1>
    <div class="subtitle">Faça upload dos orçamentos em PDF · A IA extrai, normaliza e compara · Baixe o Excel pronto</div>
</div>
""", unsafe_allow_html=True)

step_tracker()


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 1 · Upload
# ─────────────────────────────────────────────────────────────────────────────
if step == 1:
    st.markdown('<div class="section-eyebrow">Passo 1 de 4</div><div class="section-title">Upload dos orçamentos</div>', unsafe_allow_html=True)
    st.info("Carregue um PDF por fornecedor. PDFs escaneados são detectados automaticamente e enviados para OCR via Gemini Vision.")

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

        f = st.file_uploader("", type=["pdf"], key=f"pdf_{i}", label_visibility="collapsed")
        if f:
            raw = f.read()
            if isinstance(raw, str): raw = raw.encode("latin-1")
            uploaded_files[sname] = raw
            pages = get_pdf_page_count(raw)
            st.markdown(
                f'<div class="success-pill">✓ {f.name} &nbsp;·&nbsp; {pages} pág.</div>',
                unsafe_allow_html=True
            )
            any_uploaded = True

        st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="ref-label">Lista de referência (opcional)</div>', unsafe_allow_html=True)
    st.caption("Cole os itens que você quer comprar. Ex: HIPOCLORITO 5% 5L, 2 BB")
    ref_text = st.text_area("", height=110, placeholder="DETERGENTE GOLD 5L, 1 BB\nSACO DE LIXO 100L, 6 FD\n...", label_visibility="collapsed")

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
        st.error("Configure a chave da API Gemini na barra lateral antes de continuar.")
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
        progress = st.progress(0)
        status   = st.empty()
        total    = len(uploaded_files) + 1

        for n, (supplier_name, pdf_bytes) in enumerate(uploaded_files.items(), 1):
            status.markdown(f"<div style='font-size:0.88rem;color:#636366;'>⚙ Processando <b>{supplier_name}</b>… <span style='color:#8E8E93'>(pode levar alguns segundos)</span></div>", unsafe_allow_html=True)
            try:
                text, is_image = extract_text_from_pdf(pdf_bytes)
                if is_image:
                    status.markdown(f"<div style='font-size:0.88rem;color:#636366;'>🔍 OCR via Gemini Vision — <b>{supplier_name}</b>…</div>", unsafe_allow_html=True)
                    images = extract_images_from_pdf(pdf_bytes)
                    items  = extract_items_from_images(images, preferences_context=st.session_state.get('preferences_context', ''))
                else:
                    items = extract_items_from_text(text, preferences_context=st.session_state.get('preferences_context', ''))

                supplier_items[supplier_name] = items
                progress.progress(n / total)
                with st.expander(f"✓ {supplier_name} — {len(items)} itens extraídos", expanded=False):
                    st.json(items)
            except RuntimeError as e:
                # Erro de quota com mensagem amigável
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Erro em {supplier_name}: {e}")
                progress.progress(n / total)

        if supplier_items:
            status.markdown("<div style='font-size:0.88rem;color:#636366;'>Cruzando e normalizando itens entre fornecedores…</div>", unsafe_allow_html=True)
            try:
                normalized = normalize_and_match(supplier_items, reference_list or None,
                    preferences_context=st.session_state.get('preferences_context', ''))
                st.session_state.supplier_data     = supplier_items
                st.session_state.normalized_items  = normalized
                st.session_state.edited_items      = [dict(x) for x in normalized]
                progress.progress(1.0)
                status.markdown(f"<div style='font-size:0.88rem;color:#1A7F37;font-weight:500;'>✓ {len(normalized)} itens normalizados e prontos para revisão.</div>", unsafe_allow_html=True)
                st.session_state.step = 3
                st.rerun()
            except Exception as e:
                st.error(f"Erro na normalização: {e}")
        else:
            st.warning("Nenhum item extraído. Verifique os PDFs.")

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
    if not items:
        st.warning("Nenhum item encontrado. Volte e processe os PDFs novamente.")
    else:
        active_suppliers = [s for s in supplier_names if s]

        rows = []
        for item in items:
            row = {
                "ID":    item.get("id", ""),
                "Item":  item.get("item", ""),
                "Marca": item.get("marca") or "",
                "Qtd":   item.get("quantidade", 1),
                "UND":   item.get("unidade", "UN"),
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
                "UND":   st.column_config.TextColumn("UND", width="small"),
                **{f"R$ {s}": st.column_config.NumberColumn(f"{s}", min_value=0, format="R$ %.2f", width="medium") for s in active_suppliers},
                "Observação": st.column_config.TextColumn("Obs", width="medium"),
            },
            hide_index=True,
        )

        st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
        with st.expander("Adicionar item manualmente (e-commerce / 4º fornecedor)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            new_name  = c1.text_input("Nome do item")
            new_marca = c2.text_input("Marca")
            new_qtd   = c3.number_input("Qtd", min_value=0.0, step=1.0)
            new_und   = c4.text_input("UND", value="UN")
            price_cols = st.columns(len(active_suppliers))
            new_prices = {}
            for i, sname in enumerate(active_suppliers):
                new_prices[sname] = price_cols[i].number_input(f"Preço {sname}", min_value=0.0, format="%.2f", key=f"np_{i}")
            new_obs = st.text_input("Observação", key="new_obs")
            if st.button("Adicionar item") and new_name:
                new_row = {"ID": len(edited_df) + 1, "Item": new_name.upper(), "Marca": new_marca,
                           "Qtd": new_qtd, "UND": new_und, "Observação": new_obs}
                for sname in active_suppliers:
                    new_row[f"R$ {sname}"] = new_prices[sname] if new_prices[sname] > 0 else None
                edited_df = pd.concat([edited_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"'{new_name}' adicionado.")

        def df_to_items(df, sup_names):
            result = []
            for _, row in df.iterrows():
                if not row.get("Item"): continue
                forn_dict = {}
                for sname in sup_names:
                    price = row.get(f"R$ {sname}")
                    forn_dict[sname] = {"preco_unit": float(price) if price and not pd.isna(price) else None, "obs": None}
                result.append({
                    "id": int(row.get("ID") or len(result) + 1),
                    "item": str(row.get("Item", "")),
                    "marca": row.get("Marca") or None,
                    "quantidade": float(row.get("Qtd") or 1),
                    "unidade": str(row.get("UND") or "UN"),
                    "fornecedores": forn_dict,
                    "observacao": row.get("Observação") or None,
                })
            return result

        # ── Captura de correções (diff IA vs usuário) ─────────────────────
        with st.expander("🧠 Correções capturadas nesta sessão", expanded=False):
            ai_items = st.session_state.normalized_items
            corrections_so_far = detect_corrections(
                ai_items,
                df_to_items(edited_df, active_suppliers),
                active_suppliers
            )
            if corrections_so_far:
                for c in corrections_so_far:
                    st.markdown(
                        f'<div style="font-size:0.82rem;padding:4px 0;border-bottom:1px solid rgba(0,0,0,0.06);">'
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
                # Detecta e salva correções antes de avançar
                new_corr = detect_corrections(
                    st.session_state.normalized_items, final, active_suppliers
                )
                if new_corr:
                    updated_prefs, n_added = merge_corrections(
                        st.session_state.get("preferences", {"corrections": []}),
                        new_corr
                    )
                    st.session_state.preferences = updated_prefs
                    st.session_state.preferences_context = build_prompt_context(updated_prefs)
                st.session_state.final_items = final
                st.session_state.step = 4
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 4 · Download
# ─────────────────────────────────────────────────────────────────────────────
elif step == 4:
    st.markdown('<div class="section-eyebrow">Passo 4 de 4</div><div class="section-title">Mapa de Compras pronto</div>', unsafe_allow_html=True)

    final_items      = st.session_state.get("final_items", [])
    active_suppliers = [s for s in supplier_names if s]

    if not final_items:
        st.warning("Nenhum item para gerar. Volte ao passo 3.")
    else:
        with st.spinner("Gerando planilha…"):
            try:
                xlsx_bytes = generate_excel(
                    items=final_items,
                    supplier_names=active_suppliers,
                    numero_sequencial=numero_seq,
                    filial=filial,
                    responsavel=responsavel,
                    data_compra=data_compra,
                )
                filename = f"MapaCompras_{filial.replace(' ','_')}_{data_compra.strftime('%d%m%Y')}.xlsx"

                # Resumo
                prices_all = [
                    p for item in final_items
                    for s in active_suppliers
                    for p in [item.get("fornecedores", {}).get(s, {}).get("preco_unit")]
                    if p
                ]
                total_menor = sum(
                    min(
                        [item.get("fornecedores", {}).get(s, {}).get("preco_unit") or 999999 for s in active_suppliers]
                    ) * item["quantidade"]
                    for item in final_items
                    if any(item.get("fornecedores", {}).get(s, {}).get("preco_unit") for s in active_suppliers)
                )

                st.markdown(f"""
                <div class="glass-card" style="margin-bottom:1.5rem;">
                    <div style="display:flex;gap:40px;align-items:center;">
                        <div>
                            <div class="supplier-label">Itens comparados</div>
                            <div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#1C1C1E;">{len(final_items)}</div>
                        </div>
                        <div>
                            <div class="supplier-label">Fornecedores</div>
                            <div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#1C1C1E;">{len(active_suppliers)}</div>
                        </div>
                        <div>
                            <div class="supplier-label">Total (menor preço)</div>
                            <div style="font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#1A7F37;">R$ {total_menor:,.2f}</div>
                        </div>
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
                        for k in ["step","supplier_data","normalized_items","edited_items","final_items","uploaded_files"]:
                            st.session_state[k] = 1 if k=="step" else ({} if k in ["supplier_data","uploaded_files"] else [])
                        st.rerun()


                # Preferências acumuladas
                prefs_now = st.session_state.get("preferences", {})
                n_corr_total = len(prefs_now.get("corrections", []))
                if n_corr_total > 0:
                    prefs_bytes = preferences_to_json_bytes(prefs_now)
                    st.download_button(
                        label=f"🧠 Salvar histórico de aprendizado ({n_corr_total} correções)",
                        data=prefs_bytes,
                        file_name="mapa_compras_preferencias.json",
                        mime="application/json",
                        use_container_width=True,
                        help="Carregue este arquivo na próxima sessão para que a IA aplique suas preferências automaticamente"
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
                    preview_rows.append(row)

                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Erro ao gerar Excel: {e}")
                st.exception(e)

    if st.button("← Voltar para revisão"):
        st.session_state.step = 3
        st.rerun()
