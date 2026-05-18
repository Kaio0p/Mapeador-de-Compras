# -*- coding: utf-8 -*-
"""
preferences_manager.py
Sistema de aprendizado por correção com persistência automática via Supabase.

A tabela 'mapa_compras_preferencias' tem uma única linha (id=1) com o campo
jsonb 'data' contendo todo o histórico de correções. O app carrega ao iniciar
e salva automaticamente ao fim de cada sessão — zero fricção para o usuário.

O catálogo 'catalogo_produtos' suporta os campos:
  - nome_oficial      (str)  — nome padronizado do produto
  - unidade_padrao    (str)  — unidade de medida padrão (UN/CX/PCT/BB/KG)
  - marca_referencia  (str)  — marca de referência preferida (opcional)
  - marca             (str)  — alias de marca_referencia (retrocompatibilidade)
"""
import json
import logging
import urllib.request
import urllib.error
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
    except urllib.error.HTTPError as e:
        logging.error("[Supabase] HTTP %d ao carregar preferências: %s", e.code, e.read().decode())
    except urllib.error.URLError as e:
        logging.error("[Supabase] Falha de conexão ao carregar preferências: %s", e.reason)
    except Exception as e:
        logging.error("[Supabase] Erro inesperado ao carregar preferências: %s", e)
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
        logging.error("[Preferences] Supabase save error: %d %s", e.code, e.read().decode())
        return False
    except Exception as e:
        logging.error("[Preferences] Supabase save error: %s", e)
        return False


def load_catalog_from_supabase(supabase_url: str, supabase_key: str) -> list:
    """
    Carrega o catálogo oficial de produtos da tabela catalogo_produtos.

    Campos retornados:
      - nome_oficial      — nome padronizado
      - unidade_padrao    — unidade de medida (UN/CX/PCT/BB/KG)
      - marca_referencia  — marca preferida (se disponível na tabela)

    Retorna [] em caso de erro — o sistema continua funcionando sem catálogo.
    """
    # Tenta buscar todos os campos relevantes (incluindo marca_referencia se existir)
    url = (
        f"{supabase_url}/rest/v1/catalogo_produtos"
        "?select=nome_oficial,unidade_padrao,marca_referencia"
    )
    req = urllib.request.Request(url, headers=_sb_headers(supabase_key))
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
            return []
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode()
        except Exception:
            pass
        # Se a coluna marca_referencia não existir, tenta sem ela (retrocompatibilidade)
        if e.code in (400, 404) and "marca_referencia" in err_body:
            logging.warning(
                "[Supabase] Coluna 'marca_referencia' não encontrada no catálogo. "
                "Tentando sem ela (retrocompatibilidade)..."
            )
            return _load_catalog_without_marca(supabase_url, supabase_key)
        logging.error("[Supabase] HTTP %d ao carregar catálogo: %s", e.code, err_body)
    except urllib.error.URLError as e:
        logging.error("[Supabase] Falha de conexão ao carregar catálogo: %s", e.reason)
    except Exception as e:
        logging.error("[Supabase] Erro inesperado ao carregar catálogo: %s", e)
    return []


def _load_catalog_without_marca(supabase_url: str, supabase_key: str) -> list:
    """Fallback: carrega catálogo sem o campo marca_referencia."""
    url = f"{supabase_url}/rest/v1/catalogo_produtos?select=nome_oficial,unidade_padrao"
    req = urllib.request.Request(url, headers=_sb_headers(supabase_key))
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        logging.error("[Supabase] Erro ao carregar catálogo (fallback): %s", e)
    return []


# ── Detecção de diffs ─────────────────────────────────────────────────────────

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_corrections(
    ai_items: list,
    user_items: list,
    supplier_names: list,
) -> list:
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
        ai_marca  = (ai_item.get("marca")   or "").strip()
        user_marca = (user_item.get("marca") or "").strip()

        # Correção de nomenclatura
        if ai_name != user_name and _similar(ai_name, user_name) > 0.3:
            corrections.append({
                "timestamp": timestamp,
                "type":      "nomenclature",
                "original":  ai_name,
                "corrected": user_name,
                "note":      f"Renomeado: '{ai_name}' → '{user_name}'",
            })

        # Correção de marca
        if ai_marca and user_marca and ai_marca.upper() != user_marca.upper():
            corrections.append({
                "timestamp":    timestamp,
                "type":         "brand_correction",
                "item_reference": user_name or ai_name,
                "original":     ai_marca,
                "corrected":    user_marca,
                "note":         f"Marca [{user_name or ai_name}]: '{ai_marca}' → '{user_marca}'",
            })

        # Correção de unidade (com preços associados)
        if ai_unit and user_unit and ai_unit != user_unit:
            price_ex = []
            for sname in supplier_names:
                ap = (ai_item.get("fornecedores") or {}).get(sname, {}).get("preco_unit")
                up = (user_item.get("fornecedores") or {}).get(sname, {}).get("preco_unit")
                if ap and up and abs(ap - up) > 0.01:
                    price_ex.append({
                        "fornecedor":      sname,
                        "preco_original":  ap,
                        "preco_corrigido": up,
                    })
            corrections.append({
                "timestamp":     timestamp,
                "type":          "unit_conversion",
                "item_reference": user_name or ai_name,
                "original_unit": ai_unit,
                "corrected_unit": user_unit,
                "original_qty":  ai_qty,
                "corrected_qty": user_qty,
                "price_corrections": price_ex,
                "note": (
                    f"Unidade: {ai_unit}→{user_unit} | "
                    f"Qtd: {ai_qty}→{user_qty} [{user_name or ai_name}]"
                ),
            })

        elif ai_unit == user_unit:
            # Mesma unidade — verifica correções de preço por fornecedor
            for sname in supplier_names:
                ap = (ai_item.get("fornecedores") or {}).get(sname, {}).get("preco_unit")
                up = (user_item.get("fornecedores") or {}).get(sname, {}).get("preco_unit")
                if ap is not None and up is not None and abs(ap - up) > 0.01:
                    corrections.append({
                        "timestamp":      timestamp,
                        "type":           "price_correction",
                        "item_reference": user_name or ai_name,
                        "fornecedor":     sname,
                        "unidade":        user_unit,
                        "preco_original": ap,
                        "preco_corrigido": up,
                        "fator":          round(up / ap, 4) if ap else None,
                        "note": (
                            f"Preço [{sname}] {user_name or ai_name}: "
                            f"R${ap:.2f}→R${up:.2f}"
                        ),
                    })

    return corrections


# ── Merge de correções ────────────────────────────────────────────────────────

def merge_corrections(existing: dict, new_corrections: list) -> tuple:
    prefs = dict(existing)
    prefs.setdefault("corrections", [])
    seen  = {c.get("note", "") for c in prefs["corrections"]}
    added = 0
    for c in new_corrections:
        if c.get("note", "") not in seen:
            prefs["corrections"].append(c)
            seen.add(c.get("note", ""))
            added += 1
    prefs["last_updated"]      = datetime.now().isoformat()
    prefs["total_corrections"] = len(prefs["corrections"])
    return prefs, added


# ── Geração de contexto para injeção nos prompts ──────────────────────────────

def build_prompt_context(prefs: dict, max_examples: int = 20) -> str:
    """
    Gera um bloco de texto com todas as preferências aprendidas para
    injeção nos prompts do Gemini e do Cohere.
    """
    if not prefs or not prefs.get("corrections"):
        return ""

    corrections = sorted(
        prefs["corrections"],
        key=lambda c: c.get("timestamp", ""),
        reverse=True,
    )
    nomenclature  = [c for c in corrections if c["type"] == "nomenclature"][:max_examples]
    unit_conv     = [c for c in corrections if c["type"] == "unit_conversion"][:max_examples]
    price_corr    = [c for c in corrections if c["type"] == "price_correction"][:5]
    brand_corr    = [c for c in corrections if c["type"] == "brand_correction"][:max_examples]

    parts = ["=== PREFERÊNCIAS E CORREÇÕES DO USUÁRIO (aplique obrigatoriamente) ===\n"]

    if nomenclature:
        parts.append("NOMENCLATURA — Como o usuário quer que os itens sejam nomeados:")
        for c in nomenclature:
            parts.append(f'  • "{c["original"]}" → "{c["corrected"]}"')
        parts.append("")

    if brand_corr:
        parts.append("MARCAS — Correções de marca feitas pelo usuário:")
        for c in brand_corr:
            parts.append(
                f'  • [{c.get("item_reference", "")}] marca "{c["original"]}" → "{c["corrected"]}"'
            )
        parts.append("")

    if unit_conv:
        parts.append("UNIDADES — Como tratar unidades e quantidades:")
        for c in unit_conv:
            parts.append(
                f'  • {c.get("item_reference", "")}: '
                f'"{c["original_unit"]}" → "{c["corrected_unit"]}" '
                f'(qtd {c["original_qty"]} → {c["corrected_qty"]})'
            )
            for pc in c.get("price_corrections", []):
                parts.append(
                    f'    [{pc["fornecedor"]}] '
                    f'R${pc["preco_original"]:.2f} → R${pc["preco_corrigido"]:.2f}'
                )
        parts.append("")

    if price_corr:
        parts.append("PREÇOS — Correções de cálculo:")
        for c in price_corr:
            parts.append(
                f'  • {c.get("item_reference", "")} [{c.get("fornecedor", "")}]: '
                f'R${c["preco_original"]:.2f} → R${c["preco_corrigido"]:.2f}'
                + (f' (fator {c["fator"]}x)' if c.get("fator") else "")
            )
        parts.append("")

    derived = _derive_rules(nomenclature, unit_conv, brand_corr)
    if derived:
        parts.append("REGRAS GERAIS derivadas:")
        for r in derived:
            parts.append(f"  • {r}")
        parts.append("")

    parts.append("=== FIM DAS PREFERÊNCIAS ===\n")
    return "\n".join(parts)


def _derive_rules(nomenclature: list, unit_conv: list, brand_corr: list = None) -> list:
    """Deriva regras gerais a partir do histórico de correções."""
    rules = []

    # Regras derivadas das correções de nomenclatura
    if any(
        any(g in c["original"] for g in ["75G", "80G", "90G", "56G", "60G"])
        and not any(g in c["corrected"] for g in ["75G", "80G", "90G", "56G", "60G"])
        for c in nomenclature
    ):
        rules.append("Não incluir gramatura no nome de papéis/itens de escritório")

    if any("(" in c["original"] and "(" not in c["corrected"] for c in nomenclature):
        rules.append("Não incluir detalhes de embalagem entre parênteses no nome")

    if any(c["original_unit"] == "UN" and c["corrected_unit"] == "PCT" for c in unit_conv):
        rules.append(
            "Itens em pacote/conjunto (c/4, c/50 etc.) devem usar PCT, não UN individual"
        )

    # Pilhas: AA = PEQUENA, AAA = PALITO
    for c in nomenclature:
        if "PILHA" in c.get("original", "") or "PILHA" in c.get("corrected", ""):
            if "AA" in c.get("corrected", "") and "PEQUENA" in c.get("corrected", ""):
                rules.append(
                    "Pilha AA = PEQUENA, Pilha AAA = PALITO. "
                    "Sempre incluir C/4 ou C/2 no nome."
                )
                break
            if "NORMAL" in c.get("corrected", ""):
                rules.append(
                    "Pilha AA = PEQUENA (ou NORMAL), Pilha AAA = PALITO. "
                    "Incluir C/4 ou C/2."
                )
                break

    # Regras derivadas de correções de marca
    if brand_corr:
        for c in brand_corr:
            item_ref = c.get("item_reference", "")
            orig     = c.get("original", "")
            corr     = c.get("corrected", "")
            if item_ref and orig and corr:
                rules.append(
                    f"Para '{item_ref}': usar marca '{corr}' (não '{orig}')"
                )

    # ── Regras fixas — sempre incluídas (mais relevantes e detalhadas) ─────────
    rules.append(
        "PILHAS: unidade=PCT, preço do PACOTE inteiro (não dividir por 4 nem por 2). "
        "Nome inclui 'C/4' ou 'C/2'. AA=PEQUENA, AAA=PALITO."
    )
    rules.append(
        "PAPEL A4: preço por RESMA individual. "
        "Se o fornecedor cotou em lote (ex: 50 resmas a R$24,90 cada), "
        "o preço unitário é R$24,90 — NÃO multiplique por 2 ou por 50."
    )
    rules.append(
        "BALDES como PRODUTO: unidade=UN (não BB). "
        "BB é reservado para EMBALAGEM de líquido (bombona, galão)."
    )
    rules.append(
        "COPOS DESCARTÁVEIS cotados em peças (100 copos, 200 copos): unidade=UN."
    )
    rules.append(
        "MARCAS: sempre extrair e preservar conforme documento original. "
        "Referências: FBOX (arquivo morto), BRW (borracha), ECOCOPPO (copo), "
        "CIS (estilete/copo), CHAMEX (papel A4), DURACELL (pilha)."
    )
    rules.append(
        "NOMES DESCRITIVOS: incluir tipo/tamanho/capacidade relevante. "
        "Ex: 'CAIXA DE ARQUIVO MORTO' (não só 'CAIXA'), "
        "'PILHA ALCALINA AA PEQUENA C/4' (não só 'PILHA')."
    )
    rules.append(
        "TODOS os fornecedores que cotaram o item DEVEM ter preco_unit preenchido — "
        "NUNCA null para fornecedor que claramente cotou."
    )

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
