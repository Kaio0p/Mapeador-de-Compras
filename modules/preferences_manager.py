# -*- coding: utf-8 -*-
"""
preferences_manager.py
Sistema de aprendizado por correção com persistência automática via Supabase.

A tabela 'mapa_compras_preferencias' tem uma única linha (id=1) com o campo
jsonb 'data' contendo todo o histórico de correções. O app carrega ao iniciar
e salva automaticamente ao fim de cada sessão — zero fricção para o usuário.
"""
import json, urllib.request, urllib.error
from datetime import datetime
from difflib import SequenceMatcher


# ── Supabase REST helpers (sem dependência extra — usa só urllib) ─────────────

def _sb_headers(supabase_key: str) -> dict:
    return {
        "apikey":        supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

def load_from_supabase(supabase_url: str, supabase_key: str) -> dict:
    """Carrega preferências do Supabase. Retorna dict vazio se não existir ainda."""
    url = f"{supabase_url}/rest/v1/mapa_compras_preferencias?id=eq.1&select=data"
    req = urllib.request.Request(url, headers=_sb_headers(supabase_key))
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            rows = json.loads(resp.read().decode())
            if rows:
                return rows[0].get("data", {"corrections": [], "version": 1})
    except Exception:
        pass
    return {"corrections": [], "version": 1}

def save_to_supabase(supabase_url: str, supabase_key: str, prefs: dict) -> bool:
    """Upsert das preferências no Supabase (cria ou atualiza a linha id=1)."""
    url     = f"{supabase_url}/rest/v1/mapa_compras_preferencias"
    payload = json.dumps({"id": 1, "data": prefs, "updated_at": "now()"}).encode()
    headers = _sb_headers(supabase_key)
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8):
            return True
    except urllib.error.HTTPError as e:
        print(f"[Preferences] Supabase save error: {e.code} {e.read().decode()}")
        return False
    except Exception as e:
        print(f"[Preferences] Supabase save error: {e}")
        return False


# ── Detecção de diffs ─────────────────────────────────────────────────────────

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def detect_corrections(
    ai_items: list[dict],
    user_items: list[dict],
    supplier_names: list[str],
) -> list[dict]:
    corrections = []
    timestamp   = datetime.now().isoformat()

    ai_map   = {str(item.get("id", i)): item for i, item in enumerate(ai_items)}
    user_map = {str(item.get("id", i)): item for i, item in enumerate(user_items)}

    for id_key in sorted(set(ai_map) | set(user_map)):
        ai_item   = ai_map.get(id_key)
        user_item = user_map.get(id_key)
        if not ai_item or not user_item:
            continue

        ai_name   = (ai_item.get("item")    or "").strip().upper()
        user_name = (user_item.get("item")  or "").strip().upper()
        ai_unit   = (ai_item.get("unidade") or "").strip().upper()
        user_unit = (user_item.get("unidade") or "").strip().upper()
        ai_qty    = float(ai_item.get("quantidade")   or 0)
        user_qty  = float(user_item.get("quantidade") or 0)

        if ai_name != user_name and _similar(ai_name, user_name) > 0.3:
            corrections.append({
                "timestamp": timestamp, "type": "nomenclature",
                "original": ai_name, "corrected": user_name,
                "note": f"Renomeado: \'{ai_name}\' → \'{user_name}\'"
            })

        if ai_unit and user_unit and ai_unit != user_unit:
            price_ex = []
            for sname in supplier_names:
                ap = (ai_item.get("fornecedores") or {}).get(sname, {}).get("preco_unit")
                up = (user_item.get("fornecedores") or {}).get(sname, {}).get("preco_unit")
                if ap and up and abs(ap - up) > 0.01:
                    price_ex.append({"fornecedor": sname, "preco_original": ap, "preco_corrigido": up})
            corrections.append({
                "timestamp": timestamp, "type": "unit_conversion",
                "item_reference": user_name or ai_name,
                "original_unit": ai_unit, "corrected_unit": user_unit,
                "original_qty": ai_qty, "corrected_qty": user_qty,
                "price_corrections": price_ex,
                "note": f"Unidade: {ai_unit}→{user_unit} | Qtd: {ai_qty}→{user_qty} [{user_name or ai_name}]"
            })
        elif ai_unit == user_unit:
            for sname in supplier_names:
                ap = (ai_item.get("fornecedores") or {}).get(sname, {}).get("preco_unit")
                up = (user_item.get("fornecedores") or {}).get(sname, {}).get("preco_unit")
                if ap and up and abs(ap - up) > 0.01:
                    corrections.append({
                        "timestamp": timestamp, "type": "price_correction",
                        "item_reference": user_name or ai_name, "fornecedor": sname,
                        "unidade": user_unit, "preco_original": ap, "preco_corrigido": up,
                        "fator": round(up / ap, 4) if ap else None,
                        "note": f"Preço [{sname}] {user_name or ai_name}: R${ap:.2f}→R${up:.2f}"
                    })

    return corrections


# ── Merge de correções ────────────────────────────────────────────────────────

def merge_corrections(existing: dict, new_corrections: list[dict]) -> tuple[dict, int]:
    prefs = dict(existing)
    prefs.setdefault("corrections", [])
    seen  = {c.get("note", "") for c in prefs["corrections"]}
    added = 0
    for c in new_corrections:
        if c.get("note", "") not in seen:
            prefs["corrections"].append(c)
            seen.add(c.get("note", ""))
            added += 1
    prefs["last_updated"]     = datetime.now().isoformat()
    prefs["total_corrections"] = len(prefs["corrections"])
    return prefs, added


# ── Geração de contexto para injeção nos prompts ──────────────────────────────

def build_prompt_context(prefs: dict, max_examples: int = 20) -> str:
    if not prefs or not prefs.get("corrections"):
        return ""

    corrections = sorted(prefs["corrections"], key=lambda c: c.get("timestamp", ""), reverse=True)
    nomenclature = [c for c in corrections if c["type"] == "nomenclature"][:max_examples]
    unit_conv    = [c for c in corrections if c["type"] == "unit_conversion"][:max_examples]
    price_corr   = [c for c in corrections if c["type"] == "price_correction"][:5]

    parts = ["=== PREFERÊNCIAS E CORREÇÕES DO USUÁRIO (aplique obrigatoriamente) ===\n"]

    if nomenclature:
        parts.append("NOMENCLATURA — Como o usuário quer que os itens sejam nomeados:")
        for c in nomenclature:
            parts.append(f'  • "{c["original"]}" → "{c["corrected"]}"'  )
        parts.append("")

    if unit_conv:
        parts.append("UNIDADES — Como tratar unidades e quantidades:")
        for c in unit_conv:
            parts.append(
                f'  • {c.get("item_reference","")}: "{c["original_unit"]}" → "{c["corrected_unit"]}" ' 
                f'(qtd {c["original_qty"]} → {c["corrected_qty"]})' 
            )
            for pc in c.get("price_corrections", []):
                parts.append(f'    [{pc["fornecedor"]}] R${pc["preco_original"]:.2f} → R${pc["preco_corrigido"]:.2f}')
        parts.append("")

    if price_corr:
        parts.append("PREÇOS — Correções de cálculo:")
        for c in price_corr:
            parts.append(
                f'  • {c.get("item_reference","")} [{c.get("fornecedor","")}]: ' 
                f'R${c["preco_original"]:.2f} → R${c["preco_corrigido"]:.2f}' 
                + (f' (fator {c["fator"]}x)' if c.get("fator") else "")
            )
        parts.append("")

    derived = _derive_rules(nomenclature, unit_conv)
    if derived:
        parts.append("REGRAS GERAIS derivadas:")
        for r in derived:
            parts.append(f"  • {r}")
        parts.append("")

    parts.append("=== FIM DAS PREFERÊNCIAS ===\n")
    return "\n".join(parts)


def _derive_rules(nomenclature, unit_conv):
    rules = []
    if any(any(g in c["original"] for g in ["75G","80G","90G","56G","60G"])
           and not any(g in c["corrected"] for g in ["75G","80G","90G","56G","60G"])
           for c in nomenclature):
        rules.append("Não incluir gramatura no nome de papéis/itens de escritório")
    if any("(" in c["original"] and "(" not in c["corrected"] for c in nomenclature):
        rules.append("Não incluir detalhes de embalagem entre parênteses no nome")
    if any(c["original_unit"] == "UN" and c["corrected_unit"] == "PCT" for c in unit_conv):
        rules.append("Itens em pacote/conjunto (c/4, c/50 etc.) devem usar PCT, não UN individual")
    for c in nomenclature:
        if "PEQUENA" in c["original"] and "NORMAL" in c["corrected"] and "PILHA" in c["original"]:
            rules.append("Pilha AA = NORMAL, Pilha AAA = PALITO")
            break
    return rules


# ── Compat: load/save local (fallback se Supabase não configurado) ─────────────

def load_preferences(json_bytes: bytes) -> dict:
    try:
        data = json.loads(json_bytes.decode("utf-8"))
        data.setdefault("corrections", [])
        data.setdefault("version", 1)
        return data
    except Exception:
        return {"corrections": [], "version": 1}

def preferences_to_json_bytes(prefs: dict) -> bytes:
    return json.dumps(prefs, ensure_ascii=False, indent=2).encode("utf-8")
