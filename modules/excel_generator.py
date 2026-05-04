"""
excel_generator.py
Gera o Excel de Mapa de Compras a partir dos dados normalizados.
Replica a estrutura exata do template 4_JAN_SG.xlsx.
"""
import io
from datetime import date
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter


# ── Paleta de cores do template original ──
COLOR_HEADER_BG   = "1F4E79"   # azul escuro
COLOR_HEADER_FG   = "FFFFFF"
COLOR_SUB_BG      = "2E75B6"   # azul médio (linha de fornecedores)
COLOR_ROW_ODD     = "DEEAF1"   # azul clarinho
COLOR_ROW_EVEN    = "FFFFFF"
COLOR_TOTAL_BG    = "BDD7EE"
COLOR_MENOR_BG    = "C6EFCE"   # verde (menor preço)
COLOR_MENOR_FG    = "375623"
COLOR_TITLE_FG    = "1F4E79"

THIN  = Side(style="thin",   color="B8CCE4")
MED   = Side(style="medium", color="1F4E79")
BORDER_CELL = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BORDER_HEADER = Border(left=MED, right=MED, top=MED, bottom=MED)


def _font(bold=False, size=10, color="000000", name="Arial"):
    return Font(name=name, bold=bold, size=size, color=color)

def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def _align(h="center", v="center", wrap=False):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

def _money_fmt():
    return 'R$ #,##0.00;[Red]-R$ #,##0.00;"-"'

def _apply_border(ws, row, col):
    ws.cell(row=row, column=col).border = BORDER_CELL


def generate_excel(
    items: list[dict],
    supplier_names: list[str],      # até 4 nomes, na ordem
    numero_sequencial: str,
    filial: str,
    responsavel: str,
    data_compra: date,
) -> bytes:
    """
    items: lista de dicts normalizados pelo gemini_processor.normalize_and_match()
    Retorna bytes do .xlsx pronto para download.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Mapa de Compras"

    # ── Larguras de coluna (espelhando template) ──
    col_widths = {
        1: 4,    # A (vazio)
        2: 5,    # B (vazio)
        3: 6,    # C (ID parte 1)
        4: 6,    # D (ID parte 2)
        5: 32,   # E (Itens)
        6: 16,   # F (Marca)
        7: 8,    # G (Qtd)
        8: 8,    # H (UND)
        9: 14,   # I (Forn 1)
        10: 14,  # J (Forn 2)
        11: 14,  # K (Forn 3)
        12: 14,  # L (Forn 4)
        13: 10,  # M (Nº NF)
        14: 14,  # N (Menor Preço)
        15: 18,  # O (Vl Total Menor)
        16: 16,  # P (Preço Autorizado)
        17: 18,  # Q (Vl Total Autor.)
        18: 6,   # R (obs p1)
        19: 20,  # S (obs p2)
        20: 4,   # T
    }
    for col, width in col_widths.items():
        ws.column_dimensions[get_column_letter(col)].width = width

    ws.row_dimensions[4].height = 30
    ws.row_dimensions[6].height = 20
    ws.row_dimensions[9].height = 22
    ws.row_dimensions[10].height = 18

    # ── LINHA 1-3: Espaço / Logo ──
    ws.row_dimensions[1].height = 8
    ws.row_dimensions[2].height = 8
    ws.row_dimensions[3].height = 8

    # ── LINHA 4: Título ──
    ws.merge_cells("C4:S4")
    c = ws["C4"]
    c.value = "MAPA DE COMPRAS"
    c.font = Font(name="Arial", bold=True, size=16, color=COLOR_TITLE_FG)
    c.alignment = _align("center", "center")
    c.fill = _fill("EBF3FB")

    ws.row_dimensions[5].height = 6

    # ── LINHA 6: Cabeçalho meta ──
    ws.row_dimensions[6].height = 22

    def _header_cell(coord, value):
        c = ws[coord]
        c.value = value
        c.font = _font(bold=False, size=9, color="1F4E79")
        c.alignment = _align("left", "center")
        c.fill = _fill("EBF3FB")
        c.border = Border(bottom=Side(style="thin", color="1F4E79"))

    _header_cell("C6", f" Número Sequencial: {numero_sequencial}")
    ws.merge_cells("C6:E6")
    _header_cell("F6", f"  Filial: {filial}")
    ws.merge_cells("F6:I6")
    _header_cell("J6", f"  Responsável: {responsavel}")
    ws.merge_cells("J6:M6")
    _header_cell("N6", " Aprovação GAD: ___________________________")
    ws.merge_cells("N6:P6")
    _header_cell("Q6", f"Data: {data_compra.strftime('%d/%m/%Y')}")
    ws.merge_cells("Q6:S6")

    ws.row_dimensions[7].height = 5
    ws.row_dimensions[8].height = 5

    # ── LINHAS 9-10: Cabeçalho da tabela ──
    def _th(ws, row, col, value, merge_to_col=None, merge_to_row=None):
        c = ws.cell(row=row, column=col, value=value)
        c.font = Font(name="Arial", bold=True, size=9, color=COLOR_HEADER_FG)
        c.fill = _fill(COLOR_HEADER_BG)
        c.alignment = _align("center", "center", wrap=True)
        c.border = BORDER_HEADER
        if merge_to_col and not merge_to_row:
            ws.merge_cells(
                start_row=row, start_column=col,
                end_row=row+1, end_column=merge_to_col
            )
        elif merge_to_row and not merge_to_col:
            ws.merge_cells(
                start_row=row, start_column=col,
                end_row=merge_to_row, end_column=col
            )

    # Linha 9: headers principais (merges verticais para maioria, horizontais para fornecedores)
    _th(ws, 9, 3, "ID",                  merge_to_col=4)   # C9:D10
    _th(ws, 9, 5, "Itens",               merge_to_row=10)  # E9:E10
    _th(ws, 9, 6, "Marca",               merge_to_row=10)  # F9:F10
    _th(ws, 9, 7, "Qtd",                 merge_to_row=10)  # G9:G10
    _th(ws, 9, 8, "UND",                 merge_to_row=10)  # H9:H10

    # Fornecedores: merge horizontal (linha 9 apenas), nomes na 10
    for col_idx in range(9, 13):
        c = ws.cell(row=9, column=col_idx, value=f"Fornecedor {col_idx-8}")
        c.font = Font(name="Arial", bold=True, size=9, color=COLOR_HEADER_FG)
        c.fill = _fill(COLOR_HEADER_BG)
        c.alignment = _align("center", "center")
        c.border = BORDER_HEADER

    _th(ws, 9, 13, "Nº NF",              merge_to_row=10)
    _th(ws, 9, 14, "Menor Preço",         merge_to_row=10)
    _th(ws, 9, 15, "Valor Total\n(Menor Preço)", merge_to_row=10)
    _th(ws, 9, 16, "Preço\nAutorizado",   merge_to_row=10)
    _th(ws, 9, 17, "Valor Total\n(Autorizado)", merge_to_col=18)  # Q9:R10 -> Q:S
    _th(ws, 9, 19, "Observação",          merge_to_col=20)

    # Linha 10: nomes dos fornecedores
    padded_suppliers = (supplier_names + ["", "", "", ""])[:4]
    for i, name in enumerate(padded_suppliers):
        col = 9 + i
        c = ws.cell(row=10, column=col, value=name.upper() if name else "")
        c.font = Font(name="Arial", bold=True, size=8, color=COLOR_HEADER_FG)
        c.fill = _fill(COLOR_SUB_BG)
        c.alignment = _align("center", "center")
        c.border = BORDER_HEADER

    # ── LINHAS DE DADOS ──
    DATA_START = 11
    MAX_ROWS = 28   # template tem 28 linhas de dado (linhas 11-38)

    for row_idx in range(MAX_ROWS):
        excel_row = DATA_START + row_idx
        ws.row_dimensions[excel_row].height = 16
        bg = COLOR_ROW_ODD if row_idx % 2 == 0 else COLOR_ROW_EVEN

        if row_idx < len(items):
            item = items[row_idx]
            suppliers_data = item.get("fornecedores", {})
        else:
            item = None
            suppliers_data = {}

        # Merge C:D para ID
        ws.merge_cells(
            start_row=excel_row, start_column=3,
            end_row=excel_row, end_column=4
        )
        id_cell = ws.cell(row=excel_row, column=3)
        id_cell.value = item["id"] if item else None
        id_cell.font = _font(bold=True, size=9, color="1F4E79")
        id_cell.alignment = _align("center", "center")
        id_cell.fill = _fill(bg)
        id_cell.border = BORDER_CELL

        # E: Item
        c_item = ws.cell(row=excel_row, column=5)
        c_item.value = item["item"] if item else None
        c_item.font = _font(size=9)
        c_item.alignment = _align("left", "center", wrap=True)
        c_item.fill = _fill(bg)
        c_item.border = BORDER_CELL

        # F: Marca
        c_marca = ws.cell(row=excel_row, column=6)
        c_marca.value = item.get("marca") if item else None
        c_marca.font = _font(size=9)
        c_marca.alignment = _align("center", "center")
        c_marca.fill = _fill(bg)
        c_marca.border = BORDER_CELL

        # G: Qtd
        c_qtd = ws.cell(row=excel_row, column=7)
        c_qtd.value = item.get("quantidade") if item else None
        c_qtd.font = _font(size=9)
        c_qtd.alignment = _align("center", "center")
        c_qtd.number_format = "#,##0.##"
        c_qtd.fill = _fill(bg)
        c_qtd.border = BORDER_CELL

        # H: UND
        c_und = ws.cell(row=excel_row, column=8)
        c_und.value = item.get("unidade") if item else None
        c_und.font = _font(size=9)
        c_und.alignment = _align("center", "center")
        c_und.fill = _fill(bg)
        c_und.border = BORDER_CELL

        # I-L: Preços dos fornecedores
        forn_prices = []
        for fi, fname in enumerate(padded_suppliers):
            col = 9 + fi
            cell = ws.cell(row=excel_row, column=col)
            price = None
            obs_forn = None
            if fname and item:
                fdata = suppliers_data.get(fname, {})
                if fdata:
                    price = fdata.get("preco_unit")
                    obs_forn = fdata.get("obs")
            cell.value = price
            cell.font = _font(size=9)
            cell.alignment = _align("right", "center")
            cell.fill = _fill(bg)
            cell.border = BORDER_CELL
            if price is not None:
                cell.number_format = _money_fmt()
                forn_prices.append((fi, price))

        # M: Nº NF
        c_nf = ws.cell(row=excel_row, column=13)
        c_nf.fill = _fill(bg)
        c_nf.border = BORDER_CELL
        c_nf.alignment = _align("center", "center")

        # N: Menor Preço — fórmula MIN(I:L)
        col_letters = [get_column_letter(9 + fi) for fi in range(len(padded_suppliers))]
        min_range = ",".join([f"{l}{excel_row}" for l in col_letters if padded_suppliers[col_letters.index(l)]])

        c_menor = ws.cell(row=excel_row, column=14)
        if item and forn_prices:
            valid_cols = [get_column_letter(9+fi) for fi, _ in forn_prices]
            c_menor.value = f"=MIN({','.join([f'{c}{excel_row}' for c in valid_cols])})"
        else:
            c_menor.value = None
        c_menor.font = Font(name="Arial", bold=True, size=9, color=COLOR_MENOR_FG)
        c_menor.fill = _fill(COLOR_MENOR_BG if item and forn_prices else bg)
        c_menor.alignment = _align("right", "center")
        c_menor.number_format = _money_fmt()
        c_menor.border = BORDER_CELL

        # O: Valor Total Menor Preço = Qtd × Menor Preço
        c_vt_menor = ws.cell(row=excel_row, column=15)
        if item and forn_prices:
            c_vt_menor.value = f"=G{excel_row}*N{excel_row}"
        else:
            c_vt_menor.value = None
        c_vt_menor.font = Font(name="Arial", bold=True, size=9, color=COLOR_MENOR_FG)
        c_vt_menor.fill = _fill(COLOR_MENOR_BG if item and forn_prices else bg)
        c_vt_menor.alignment = _align("right", "center")
        c_vt_menor.number_format = _money_fmt()
        c_vt_menor.border = BORDER_CELL

        # P: Preço Autorizado (começa igual ao menor, editável)
        c_aut = ws.cell(row=excel_row, column=16)
        if item and forn_prices:
            c_aut.value = f"=N{excel_row}"
        else:
            c_aut.value = None
        c_aut.font = _font(size=9, color="7F5200")
        c_aut.fill = _fill("FFF2CC" if item and forn_prices else bg)
        c_aut.alignment = _align("right", "center")
        c_aut.number_format = _money_fmt()
        c_aut.border = BORDER_CELL

        # Q-R: Valor Total Autorizado (merge Q:S)
        ws.merge_cells(
            start_row=excel_row, start_column=17,
            end_row=excel_row, end_column=18
        )
        c_vt_aut = ws.cell(row=excel_row, column=17)
        if item and forn_prices:
            c_vt_aut.value = f"=G{excel_row}*P{excel_row}"
        else:
            c_vt_aut.value = None
        c_vt_aut.font = Font(name="Arial", bold=True, size=9, color="7F5200")
        c_vt_aut.fill = _fill("FFF2CC" if item and forn_prices else bg)
        c_vt_aut.alignment = _align("right", "center")
        c_vt_aut.number_format = _money_fmt()
        c_vt_aut.border = BORDER_CELL

        # S-T: Observação (merge)
        ws.merge_cells(
            start_row=excel_row, start_column=19,
            end_row=excel_row, end_column=20
        )
        c_obs = ws.cell(row=excel_row, column=19)
        if item:
            # Consolida obs de todos fornecedores
            obs_parts = []
            for fname in padded_suppliers:
                if fname and item:
                    fdata = suppliers_data.get(fname, {})
                    if fdata and fdata.get("obs"):
                        obs_parts.append(f"[{fname[:3]}] {fdata['obs']}")
            main_obs = item.get("observacao") or ""
            if obs_parts:
                c_obs.value = main_obs + " | ".join(obs_parts)
            else:
                c_obs.value = main_obs or None
        c_obs.font = _font(size=8, color="595959")
        c_obs.alignment = _align("left", "center", wrap=True)
        c_obs.fill = _fill(bg)
        c_obs.border = BORDER_CELL

    # ── LINHA DE TOTAIS ──
    total_row = DATA_START + MAX_ROWS
    ws.row_dimensions[total_row].height = 18

    c_total_label = ws.cell(row=total_row, column=3, value="TOTAL")
    ws.merge_cells(
        start_row=total_row, start_column=3,
        end_row=total_row, end_column=8
    )
    c_total_label.font = Font(name="Arial", bold=True, size=10, color=COLOR_HEADER_FG)
    c_total_label.fill = _fill(COLOR_HEADER_BG)
    c_total_label.alignment = _align("center", "center")

    for fi in range(4):
        col = 9 + fi
        c = ws.cell(row=total_row, column=col)
        col_l = get_column_letter(col)
        c.value = f"=SUM({col_l}{DATA_START}:{col_l}{DATA_START+MAX_ROWS-1})"
        c.font = Font(name="Arial", bold=True, size=9, color=COLOR_HEADER_FG)
        c.fill = _fill(COLOR_HEADER_BG)
        c.alignment = _align("right", "center")
        c.number_format = _money_fmt()
        c.border = BORDER_HEADER

    for col in [13, 14]:
        c = ws.cell(row=total_row, column=col)
        c.fill = _fill(COLOR_HEADER_BG)
        c.border = BORDER_HEADER

    c_total_menor = ws.cell(row=total_row, column=15)
    c_total_menor.value = f"=SUM(O{DATA_START}:O{DATA_START+MAX_ROWS-1})"
    c_total_menor.font = Font(name="Arial", bold=True, size=10, color=COLOR_HEADER_FG)
    c_total_menor.fill = _fill("375623")
    c_total_menor.alignment = _align("right", "center")
    c_total_menor.number_format = _money_fmt()
    c_total_menor.border = BORDER_HEADER

    ws.cell(row=total_row, column=16).fill = _fill(COLOR_HEADER_BG)
    ws.cell(row=total_row, column=16).border = BORDER_HEADER

    c_total_aut = ws.cell(row=total_row, column=17)
    ws.merge_cells(
        start_row=total_row, start_column=17,
        end_row=total_row, end_column=18
    )
    c_total_aut.value = f"=SUM(Q{DATA_START}:Q{DATA_START+MAX_ROWS-1})"
    c_total_aut.font = Font(name="Arial", bold=True, size=10, color=COLOR_HEADER_FG)
    c_total_aut.fill = _fill("7F5200")
    c_total_aut.alignment = _align("right", "center")
    c_total_aut.number_format = _money_fmt()
    c_total_aut.border = BORDER_HEADER

    for col in [19, 20]:
        c = ws.cell(row=total_row, column=col)
        c.fill = _fill(COLOR_HEADER_BG)
        c.border = BORDER_HEADER

    # ── Congela painel no cabeçalho ──
    ws.freeze_panes = "E11"

    # ── Print area ──
    ws.print_area = f"B1:T{total_row+2}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToWidth = 1

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
