# -*- coding: utf-8 -*-
"""
preferences_manager.py
Sistema de aprendizado por correção.

Fluxo:
  1. Compara output da IA (normalized_items) com edições do usuário (edited_df)
  2. Detecta diffs: renomeações, trocas de unidade, ajustes de preço, mudanças de qtd
  3. Armazena como correções estruturadas em JSON
  4. Nas próximas sessões, injeta correções nos prompts como few-shot examples + regras
"""
import json
from datetime import datetime
from difflib import SequenceMatcher


# ── Tipos de correção detectados ──────────────────────────────────────────────

def _similar(a: str, b: str) -> float:
    """Similaridade entre strings (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_corrections(
    ai_items: list[dict],
    user_items: list[dict],
    supplier_names: list[str],
) -> list[dict]:
    """
    Compara os itens da IA com os editados pelo usuário.
    Retorna lista de correções detectadas.
    """
    corrections = []
    timestamp   = datetime.now().isoformat()

    # Mapeia por ID para comparação
    ai_map   = {str(item.get("id", i)): item for i, item in enumerate(ai_items)}
    user_map = {str(item.get("id", i)): item for i, item in enumerate(user_items)}

    all_ids = set(ai_map.keys()) | set(user_map.keys())

    for id_key in sorted(all_ids):
        ai_item   = ai_map.get(id_key)
        user_item = user_map.get(id_key)

        if not ai_item or not user_item:
            continue   # item adicionado/removido — não gera regra de normalização

        ai_name   = (ai_item.get("item") or "").strip().upper()
        user_name = (user_item.get("item") or "").strip().upper()
        ai_unit   = (ai_item.get("unidade") or "").strip().upper()
        user_unit = (user_item.get("unidade") or "").strip().upper()
        ai_qty    = float(ai_item.get("quantidade") or 0)
        user_qty  = float(user_item.get("quantidade") or 0)

        # ── Correção de nomenclatura ──────────────────────────────────────────
        if ai_name != user_name and _similar(ai_name, user_name) > 0.3:
            corrections.append({
                "timestamp": timestamp,
                "type": "nomenclature",
                "original": ai_name,
                "corrected": user_name,
                "note": f"Renomeado de '{ai_name}' para '{user_name}'"
            })

        # ── Correção de unidade ───────────────────────────────────────────────
        if ai_unit and user_unit and ai_unit != user_unit:
            # Captura também variação de preço associada (quando a unidade muda o preço muda)
            price_examples = []
            for sname in supplier_names:
                ai_fdata   = (ai_item.get("fornecedores") or {}).get(sname, {})
                user_fdata = (user_item.get("fornecedores") or {}).get(sname, {})
                ai_price   = ai_fdata.get("preco_unit") if ai_fdata else None
                user_price = user_fdata.get("preco_unit") if user_fdata else None
                if ai_price and user_price and abs(ai_price - user_price) > 0.01:
                    price_examples.append({
                        "fornecedor": sname,
                        "preco_original": ai_price,
                        "preco_corrigido": user_price,
                    })

            corrections.append({
                "timestamp": timestamp,
                "type": "unit_conversion",
                "item_reference": user_name or ai_name,
                "original_unit": ai_unit,
                "corrected_unit": user_unit,
                "original_qty": ai_qty,
                "corrected_qty": user_qty,
                "price_corrections": price_examples,
                "note": (
                    f"Unidade: {ai_unit} → {user_unit} | "
                    f"Qtd: {ai_qty} → {user_qty}"
                )
            })

        # ── Correção de preço (sem troca de unidade) ──────────────────────────
        elif ai_unit == user_unit:
            for sname in supplier_names:
                ai_fdata   = (ai_item.get("fornecedores") or {}).get(sname, {})
                user_fdata = (user_item.get("fornecedores") or {}).get(sname, {})
                ai_price   = ai_fdata.get("preco_unit") if ai_fdata else None
                user_price = user_fdata.get("preco_unit") if user_fdata else None
                if ai_price and user_price and abs(ai_price - user_price) > 0.01:
                    ratio = user_price / ai_price if ai_price else None
                    corrections.append({
                        "timestamp": timestamp,
                        "type": "price_correction",
                        "item_reference": user_name or ai_name,
                        "fornecedor": sname,
                        "unidade": user_unit,
                        "preco_original": ai_price,
                        "preco_corrigido": user_price,
                        "fator": round(ratio, 4) if ratio else None,
                        "note": f"Preço corrigido de R${ai_price:.2f} → R${user_price:.2f}"
                    })

    return [c for c in corrections if c]   # remove vazios


# ── Persistência ──────────────────────────────────────────────────────────────

def load_preferences(json_bytes: bytes) -> dict:
    """Carrega preferences de bytes (arquivo JSON uploadado)."""
    try:
        data = json.loads(json_bytes.decode("utf-8"))
        # Garante estrutura mínima
        data.setdefault("corrections", [])
        data.setdefault("version", 1)
        return data
    except Exception:
        return {"corrections": [], "version": 1}


def merge_corrections(existing: dict, new_corrections: list[dict]) -> dict:
    """Adiciona novas correções ao histórico, evitando duplicatas óbvias."""
    prefs = dict(existing)
    prefs.setdefault("corrections", [])

    existing_notes = {c.get("note", "") for c in prefs["corrections"]}
    added = 0
    for c in new_corrections:
        if c.get("note", "") not in existing_notes:
            prefs["corrections"].append(c)
            existing_notes.add(c.get("note", ""))
            added += 1

    prefs["last_updated"] = datetime.now().isoformat()
    prefs["total_corrections"] = len(prefs["corrections"])
    return prefs, added


def preferences_to_json_bytes(prefs: dict) -> bytes:
    """Serializa preferences para download."""
    return json.dumps(prefs, ensure_ascii=False, indent=2).encode("utf-8")


# ── Geração de contexto para injeção nos prompts ──────────────────────────────

def build_prompt_context(prefs: dict, max_examples: int = 15) -> str:
    """
    Gera bloco de texto a ser injetado nos prompts do Gemini.
    Usa as correções mais recentes como few-shot examples + regras derivadas.
    """
    if not prefs or not prefs.get("corrections"):
        return ""

    corrections = prefs["corrections"]
    # Mais recentes primeiro
    corrections = sorted(corrections, key=lambda c: c.get("timestamp", ""), reverse=True)

    nomenclature = [c for c in corrections if c["type"] == "nomenclature"][:max_examples]
    unit_conv    = [c for c in corrections if c["type"] == "unit_conversion"][:max_examples]
    price_corr   = [c for c in corrections if c["type"] == "price_correction"][:5]

    parts = ["=== PREFERÊNCIAS E CORREÇÕES DO USUÁRIO (aplique obrigatoriamente) ===\n"]

    if nomenclature:
        parts.append("NOMENCLATURA — Exemplos de como o usuário quer que os itens sejam nomeados:")
        for c in nomenclature:
            parts.append(f'  • "{c["original"]}" → "{c["corrected"]}"')
        parts.append("")

    if unit_conv:
        parts.append("UNIDADES — Como o usuário quer que as unidades sejam tratadas:")
        for c in unit_conv:
            item_ref = c.get("item_reference", "")
            parts.append(
                f'  • {item_ref}: unidade "{c["original_unit"]}" foi corrigida para "{c["corrected_unit"]}" '
                f'(qtd: {c["original_qty"]} → {c["corrected_qty"]})'
            )
            if c.get("price_corrections"):
                for pc in c["price_corrections"]:
                    parts.append(
                        f'    Preço [{pc["fornecedor"]}]: R${pc["preco_original"]:.2f} → R${pc["preco_corrigido"]:.2f}'
                    )
        parts.append("")

    if price_corr:
        parts.append("PREÇOS — Correções de cálculo aplicadas pelo usuário:")
        for c in price_corr:
            parts.append(
                f'  • {c.get("item_reference","")} [{c.get("fornecedor","")}] '
                f'{c.get("unidade","")}: R${c["preco_original"]:.2f} → R${c["preco_corrigido"]:.2f}'
                + (f' (fator: {c["fator"]}x)' if c.get("fator") else "")
            )
        parts.append("")

    # Deriva regras gerais a partir dos padrões observados
    derived_rules = _derive_rules(nomenclature, unit_conv)
    if derived_rules:
        parts.append("REGRAS GERAIS derivadas das correções:")
        for rule in derived_rules:
            parts.append(f"  • {rule}")
        parts.append("")

    parts.append("=== FIM DAS PREFERÊNCIAS ===\n")
    return "\n".join(parts)


def _derive_rules(nomenclature: list, unit_conv: list) -> list[str]:
    """Deriva regras gerais a partir dos padrões de correção."""
    rules = []

    # Detecta padrão: gramatura removida de nomes
    gram_removed = [
        c for c in nomenclature
        if any(x in c["original"] for x in ["75G","80G","90G","56G","60G"])
        and not any(x in c["corrected"] for x in ["75G","80G","90G","56G","60G"])
    ]
    if gram_removed:
        rules.append("Não incluir gramatura no nome de papéis/itens de escritório")

    # Detecta padrão: quantidades entre parênteses removidas
    paren_removed = [
        c for c in nomenclature
        if "(" in c["original"] and "(" not in c["corrected"]
    ]
    if paren_removed:
        rules.append("Não incluir detalhes de embalagem entre parênteses no nome do item")

    # Detecta padrão: UN → PCT para itens vendidos em pacote
    un_to_pct = [c for c in unit_conv if c["original_unit"] == "UN" and c["corrected_unit"] == "PCT"]
    if un_to_pct:
        rules.append(
            "Itens vendidos em pacote/conjunto (ex: pilhas c/4, parafusos c/50) "
            "devem usar a unidade do pacote (PCT), não ser explodidos em unidades individuais"
        )

    # Detecta renomeações de tamanho de pilha
    pilha_renames = [
        c for c in nomenclature
        if "PILHA" in c["original"] or "PILHA" in c["corrected"]
    ]
    if pilha_renames:
        for c in pilha_renames:
            if "PEQUENA" in c["original"] and "NORMAL" in c["corrected"]:
                rules.append("Pilha AA = 'NORMAL' (não 'PEQUENA'). Pilha AAA = 'PALITO'")
                break

    return rules
