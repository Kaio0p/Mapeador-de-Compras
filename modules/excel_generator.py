# -*- coding: utf-8 -*-
"""
excel_generator.py
Preenche o template real do Mapa de Compras.
Carrega template/mapa_compras_template.xlsx e escreve apenas nas células de dados.
Toda formatação, fórmulas e estrutura vêm do template original intocados.

Mapeamento do template:
  Cabeçalho:
    C6 = Número Sequencial | F6 = Filial | H6 = Responsável | R6 = Data

  Fornecedores (linha 10): I10 J10 K10 L10

  Dados (linhas 11-53):
    E=Item  F=Marca  G=Qtd  H=UND
    I=Forn1  J=Forn2  K=Forn3  L=Forn4
    P=Preço Autorizado (preenchido quando um fornecedor é aprovado)
    R=Observação

  Fórmulas intocadas: N(menor preço), O(total menor), Q(total aut), totais linha 54
"""
import io, os
from datetime import date
import openpyxl

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "template", "mapa_compras_template.xlsx"
)

DATA_START = 11
DATA_END   = 53
SUP_COLS   = ["I", "J", "K", "L"]
DATA_COLS  = ["E", "F", "G", "H", "I", "J", "K", "L", "P", "R"]


def generate_excel(
    items,
    supplier_names,
    numero_sequencial,
    filial,
    responsavel,
    data_compra,
    approved_supplier=None,
):
    """
    Gera o Excel do Mapa de Compras a partir do template.
    
    Args:
        items: Lista de itens normalizados
        supplier_names: Lista com nomes dos fornecedores
        numero_sequencial: Número sequencial do mapa
        filial: Nome da filial
        responsavel: Nome do responsável
        data_compra: Data da compra
        approved_supplier: Nome do fornecedor aprovado (opcional).
            Se fornecido, preenche a coluna P (Preço Autorizado) com os preços desse fornecedor.
    """
    wb = openpyxl.load_workbook(_TEMPLATE_PATH)
    ws = wb.active

    # Cabeçalho
    ws["C6"] = f" Número Sequencial: {numero_sequencial}"
    ws["F6"] = f"               Filial: {filial}"
    ws["H6"] = f"                                                                                                        Responsável pela compra: {responsavel}"
    ws["R6"] = f"Data: {data_compra.strftime('%d/%m/%Y')}"

    # Nomes dos fornecedores
    padded = (list(supplier_names) + ["", "", "", ""])[:4]
    for col, name in zip(SUP_COLS, padded):
        ws[f"{col}10"] = name.upper() if name else ""

    # Limpa linhas de dados
    for row in range(DATA_START, DATA_END + 1):
        for col in DATA_COLS:
            ws[f"{col}{row}"] = None

    # Preenche itens
    max_items = DATA_END - DATA_START + 1
    for i, item in enumerate(items[:max_items]):
        row = DATA_START + i
        ws[f"E{row}"] = (item.get("item") or "").upper()
        ws[f"F{row}"] = item.get("marca") or None

        qtd = item.get("quantidade")
        ws[f"G{row}"] = float(qtd) if qtd is not None else None
        ws[f"H{row}"] = item.get("unidade") or None

        fornecedores = item.get("fornecedores") or {}
        for col, fname in zip(SUP_COLS, padded):
            if not fname:
                ws[f"{col}{row}"] = None
                continue
            fdata = fornecedores.get(fname) or {}
            price = fdata.get("preco_unit")
            ws[f"{col}{row}"] = float(price) if price is not None else None

        # Preço Autorizado (coluna P) — se um fornecedor foi aprovado
        if approved_supplier and approved_supplier in fornecedores:
            fdata = fornecedores.get(approved_supplier) or {}
            approved_price = fdata.get("preco_unit")
            ws[f"P{row}"] = float(approved_price) if approved_price is not None else None
        else:
            ws[f"P{row}"] = None

        # Observação consolidada
        obs_parts = []
        if item.get("observacao"):
            obs_parts.append(item["observacao"])
        for fname in padded:
            if not fname:
                continue
            fdata = fornecedores.get(fname) or {}
            if fdata.get("obs"):
                obs_parts.append(f"[{fname[:4]}] {fdata['obs']}")
        ws[f"R{row}"] = " | ".join(obs_parts) if obs_parts else None

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
