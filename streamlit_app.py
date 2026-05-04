"""
app.py — Gerador de Mapa de Compras
Pipeline: Upload PDFs → Extração IA → Revisão → Excel
"""
import streamlit as st
import pandas as pd
from datetime import date, datetime
import json

from modules.pdf_extractor import (
    extract_text_from_pdf, extract_images_from_pdf, get_pdf_page_count
)
from modules.gemini_processor import (
    configure as configure_gemini,
    extract_items_from_text,
    extract_items_from_images,
    normalize_and_match
)
from modules.excel_generator import generate_excel

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mapa de Compras | EBD Grupo",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Compatibilidade de encoding — força UTF-8 no stdout/stderr
# ─────────────────────────────────────────────
import sys
import io as _io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# CSS customizado
st.markdown("""
<style>
    .stApp { background-color: #f0f4f9; }
    .block-container { padding-top: 2rem; }
    h1 { color: #1F4E79; }
    h2, h3 { color: #2E75B6; }
    .supplier-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #2E75B6;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .success-box {
        background: #e8f5e9;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #4CAF50;
        margin: 8px 0;
    }
    .step-badge {
        background: #1F4E79;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers de estado
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 1,
        "supplier_data": {},       # {nome: lista_itens_brutos}
        "normalized_items": [],    # lista normalizada
        "edited_items": [],        # após edição do usuário
        "api_key_ok": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1F4E79/FFFFFF?text=EBD+Grupo", use_column_width=True)
    st.markdown("---")
    st.markdown("### ⚙️ Configuração")

    # API Key
    try:
        _default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        _default_key = ""

    api_key_input = st.text_input(
        "Chave API Gemini",
        type="password",
        value=_default_key,
        help="Obtenha gratuitamente em aistudio.google.com"
    )
    if api_key_input:
        try:
            configure_gemini(api_key_input)
            st.success("✅ API conectada")
            st.session_state.api_key_ok = True
        except Exception as e:
            st.error(f"Erro: {e}")
            st.session_state.api_key_ok = False

    st.markdown("---")
    st.markdown("### 📍 Cabeçalho do Mapa")
    numero_seq = st.text_input("Número Sequencial", value="2026001001")
    filial = st.text_input("Filial", value="São Gonçalo")
    responsavel = st.text_input("Responsável", value="")
    data_compra = st.date_input("Data", value=date.today())

    st.markdown("---")
    st.markdown("### 🗂️ Fornecedores")
    n_suppliers = st.radio("Quantidade de fornecedores", [3, 4], horizontal=True)

    supplier_names = []
    for i in range(n_suppliers):
        name = st.text_input(f"Fornecedor {i+1}", value=["", "", "", ""][i], key=f"sup_name_{i}")
        supplier_names.append(name)

    st.markdown("---")
    st.caption("v1.0 · Grupo EBD · Depto. Compras")


# ─────────────────────────────────────────────
# Header principal
# ─────────────────────────────────────────────
st.markdown("# 📋 Gerador de Mapa de Compras")
st.markdown("Upload dos PDFs de orçamento → extração via IA → revisão → Excel pronto")
st.markdown("---")

# Indicador de passos
step = st.session_state.step
cols_steps = st.columns(4)
step_labels = ["1️⃣ Upload PDFs", "2️⃣ Extração IA", "3️⃣ Revisão", "4️⃣ Download Excel"]
for i, (col, label) in enumerate(zip(cols_steps, step_labels)):
    with col:
        if i + 1 == step:
            st.markdown(f"**🔵 {label}**")
        elif i + 1 < step:
            st.markdown(f"✅ {label}")
        else:
            st.markdown(f"⚪ {label}")

st.markdown("---")


# ─────────────────────────────────────────────
# PASSO 1: Upload dos PDFs
# ─────────────────────────────────────────────
if step == 1:
    st.markdown("## Passo 1 — Upload dos Orçamentos")
    st.info("Faça upload de um PDF por fornecedor. PDFs escaneados são detectados automaticamente e tratados com OCR via IA.")

    uploaded_files = {}
    any_uploaded = False

    for i in range(n_suppliers):
        sname = supplier_names[i] or f"Fornecedor {i+1}"
        with st.container():
            st.markdown(f'<div class="supplier-card"><b>🏪 {sname}</b></div>', unsafe_allow_html=True)
            f = st.file_uploader(
                f"PDF do orçamento — {sname}",
                type=["pdf"],
                key=f"pdf_{i}",
                label_visibility="collapsed"
            )
            if f:
                raw_bytes = f.read()
                # Garante que é bytes puros (não tenta decodificar como texto)
                if isinstance(raw_bytes, str):
                    raw_bytes = raw_bytes.encode("latin-1")
                uploaded_files[sname] = raw_bytes
                pages = get_pdf_page_count(uploaded_files[sname])
                st.markdown(f'<div class="success-box">✅ <b>{f.name}</b> — {pages} página(s)</div>', unsafe_allow_html=True)
                any_uploaded = True

    # Lista de referência opcional
    st.markdown("#### 📄 Lista de referência (opcional)")
    st.caption("Cole aqui a lista de itens a comprar. Ex: HIPOCLORITO 5% 5L, 2 BB | SACO DE LIXO 100L, 6 FD")
    ref_text = st.text_area("Lista de referência", height=120, placeholder="Item, Qtd UND\nEx: DETERGENTE GOLD 5L, 1 BB")

    col_next, _ = st.columns([1, 3])
    with col_next:
        if st.button("▶️ Avançar para Extração", type="primary", disabled=not any_uploaded):
            st.session_state.uploaded_files = uploaded_files
            st.session_state.ref_text = ref_text
            st.session_state.step = 2
            st.rerun()


# ─────────────────────────────────────────────
# PASSO 2: Extração com IA
# ─────────────────────────────────────────────
elif step == 2:
    st.markdown("## Passo 2 — Extração e Normalização via IA")

    if not st.session_state.api_key_ok:
        st.error("⚠️ Configure a chave da API Gemini na barra lateral antes de continuar.")
        st.stop()

    uploaded_files = st.session_state.get("uploaded_files", {})
    ref_text = st.session_state.get("ref_text", "")

    # Parseia lista de referência
    reference_list = []
    if ref_text.strip():
        for line in ref_text.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 2:
                item_name = parts[0].strip().upper()
                rest = parts[1].strip().split()
                try:
                    qty = float(rest[0]) if rest else 1
                    unit = rest[1].upper() if len(rest) > 1 else "UN"
                except:
                    qty, unit = 1, "UN"
                reference_list.append({"item": item_name, "quantidade": qty, "unidade": unit})

    if st.button("🤖 Iniciar Extração com IA", type="primary"):
        supplier_items = {}

        progress = st.progress(0)
        status = st.empty()

        total = len(uploaded_files) + 1  # +1 para normalização
        step_n = 0

        for supplier_name, pdf_bytes in uploaded_files.items():
            status.markdown(f"⚙️ Processando **{supplier_name}**...")
            try:
                text, is_image = extract_text_from_pdf(pdf_bytes)

                if is_image:
                    status.markdown(f"🔍 **{supplier_name}** parece escaneado — usando OCR via Gemini Vision...")
                    images = extract_images_from_pdf(pdf_bytes)
                    items = extract_items_from_images(images)
                else:
                    items = extract_items_from_text(text)

                supplier_items[supplier_name] = items
                step_n += 1
                progress.progress(step_n / total)

                with st.expander(f"✅ {supplier_name} — {len(items)} itens extraídos", expanded=False):
                    st.json(items)

            except Exception as e:
                st.error(f"Erro ao processar {supplier_name}: {e}")
                step_n += 1
                progress.progress(step_n / total)

        if supplier_items:
            status.markdown("🔄 Normalizando e cruzando itens entre fornecedores...")
            try:
                normalized = normalize_and_match(supplier_items, reference_list or None)
                st.session_state.supplier_data = supplier_items
                st.session_state.normalized_items = normalized
                st.session_state.edited_items = [dict(item) for item in normalized]
                step_n += 1
                progress.progress(1.0)
                status.markdown(f"✅ Normalização concluída — **{len(normalized)} itens** prontos para revisão.")
                st.session_state.step = 3
                st.rerun()
            except Exception as e:
                st.error(f"Erro na normalização: {e}")
        else:
            st.warning("Nenhum item foi extraído. Verifique os PDFs e tente novamente.")

    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("◀️ Voltar"):
            st.session_state.step = 1
            st.rerun()


# ─────────────────────────────────────────────
# PASSO 3: Revisão e edição
# ─────────────────────────────────────────────
elif step == 3:
    st.markdown("## Passo 3 — Revisão e Edição")
    st.info("Revise os itens extraídos. Edite diretamente na tabela. Você pode ajustar itens, quantidades e preços antes de gerar o Excel.")

    items = st.session_state.normalized_items
    if not items:
        st.warning("Nenhum item encontrado. Volte e processe os PDFs novamente.")
    else:
        active_suppliers = [s for s in supplier_names if s]

        # Monta DataFrame para edição
        rows = []
        for item in items:
            row = {
                "ID": item.get("id", ""),
                "Item": item.get("item", ""),
                "Marca": item.get("marca") or "",
                "Qtd": item.get("quantidade", 1),
                "UND": item.get("unidade", "UN"),
            }
            for sname in active_suppliers:
                fdata = item.get("fornecedores", {}).get(sname, {})
                row[f"Preço {sname}"] = fdata.get("preco_unit") if fdata else None

            row["Observação"] = item.get("observacao") or ""
            rows.append(row)

        df = pd.DataFrame(rows)

        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "ID": st.column_config.NumberColumn("ID", min_value=1, step=1, width="small"),
                "Item": st.column_config.TextColumn("Item", width="large"),
                "Marca": st.column_config.TextColumn("Marca", width="medium"),
                "Qtd": st.column_config.NumberColumn("Qtd", min_value=0, step=0.5, width="small"),
                "UND": st.column_config.TextColumn("UND", width="small"),
                **{
                    f"Preço {s}": st.column_config.NumberColumn(
                        f"💰 {s}", min_value=0, format="R$ %.2f", width="medium"
                    )
                    for s in active_suppliers
                },
                "Observação": st.column_config.TextColumn("Obs", width="medium"),
            },
            hide_index=True,
        )

        # Botão para adicionar item manualmente
        st.markdown("#### ➕ Adicionar item manualmente")
        with st.expander("Adicionar item extra (ex: item de e-commerce ou 4º fornecedor)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            new_item_name = c1.text_input("Nome do item")
            new_marca = c2.text_input("Marca")
            new_qtd = c3.number_input("Qtd", min_value=0.0, step=1.0)
            new_und = c4.text_input("UND", value="UN")

            price_cols = st.columns(len(active_suppliers))
            new_prices = {}
            for i, sname in enumerate(active_suppliers):
                new_prices[sname] = price_cols[i].number_input(f"Preço {sname}", min_value=0.0, format="%.2f", key=f"np_{i}")

            new_obs = st.text_input("Observação")
            if st.button("Adicionar item") and new_item_name:
                new_row = {
                    "ID": len(edited_df) + 1,
                    "Item": new_item_name.upper(),
                    "Marca": new_marca,
                    "Qtd": new_qtd,
                    "UND": new_und,
                    "Observação": new_obs,
                }
                for sname in active_suppliers:
                    new_row[f"Preço {sname}"] = new_prices[sname] if new_prices[sname] > 0 else None
                new_row_df = pd.DataFrame([new_row])
                edited_df = pd.concat([edited_df, new_row_df], ignore_index=True)
                st.success(f"Item '{new_item_name}' adicionado!")

        # Converte df editado de volta para formato interno
        def df_to_items(df, supplier_names):
            result = []
            for _, row in df.iterrows():
                if not row.get("Item"):
                    continue
                forn_dict = {}
                for sname in supplier_names:
                    price = row.get(f"Preço {sname}")
                    forn_dict[sname] = {
                        "preco_unit": float(price) if price and not pd.isna(price) else None,
                        "obs": None
                    }
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

        col_back, col_next = st.columns([1, 4])
        with col_back:
            if st.button("◀️ Voltar"):
                st.session_state.step = 2
                st.rerun()
        with col_next:
            if st.button("▶️ Gerar Excel", type="primary"):
                final_items = df_to_items(edited_df, active_suppliers)
                st.session_state.final_items = final_items
                st.session_state.step = 4
                st.rerun()


# ─────────────────────────────────────────────
# PASSO 4: Download do Excel
# ─────────────────────────────────────────────
elif step == 4:
    st.markdown("## Passo 4 — Download do Mapa de Compras")

    final_items = st.session_state.get("final_items", [])
    active_suppliers = [s for s in supplier_names if s]

    if not final_items:
        st.warning("Nenhum item para gerar. Volte ao passo 3.")
    else:
        with st.spinner("Gerando Excel..."):
            try:
                xlsx_bytes = generate_excel(
                    items=final_items,
                    supplier_names=active_suppliers,
                    numero_sequencial=numero_seq,
                    filial=filial,
                    responsavel=responsavel,
                    data_compra=data_compra,
                )

                filename = f"MapaCompras_{filial.replace(' ', '_')}_{data_compra.strftime('%d%m%Y')}.xlsx"

                st.success(f"✅ Excel gerado com **{len(final_items)} itens** e **{len(active_suppliers)} fornecedores**!")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="⬇️ Baixar Mapa de Compras.xlsx",
                        data=xlsx_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True
                    )
                with col2:
                    if st.button("🔄 Gerar novo mapa", use_container_width=True):
                        for key in ["step", "supplier_data", "normalized_items",
                                    "edited_items", "final_items", "uploaded_files"]:
                            st.session_state[key] = (1 if key == "step" else
                                                      {} if key in ["supplier_data", "uploaded_files"] else [])
                        st.rerun()

                # Preview da tabela
                st.markdown("### 👁️ Preview do mapa gerado")
                preview_rows = []
                for item in final_items:
                    row = {
                        "ID": item["id"],
                        "Item": item["item"],
                        "Qtd": item["quantidade"],
                        "UND": item["unidade"],
                    }
                    prices = []
                    for sname in active_suppliers:
                        p = item.get("fornecedores", {}).get(sname, {}).get("preco_unit")
                        row[sname] = f"R$ {p:.2f}" if p else "—"
                        if p:
                            prices.append(p)
                    row["Menor Preço"] = f"R$ {min(prices):.2f}" if prices else "—"
                    row["Total (Menor)"] = f"R$ {item['quantidade'] * min(prices):.2f}" if prices else "—"
                    preview_rows.append(row)

                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Erro ao gerar Excel: {e}")
                st.exception(e)

    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("◀️ Voltar para revisão"):
            st.session_state.step = 3
            st.rerun()
