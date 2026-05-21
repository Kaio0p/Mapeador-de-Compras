# -*- coding: utf-8 -*-
"""
cohere_processor.py — Agente Lógico/Normalizador via Cohere command-a-reasoning
===========================================================================
Responsável pela NORMALIZAÇÃO e CRUZAMENTO dos orçamentos extraídos.

Pipeline:
  1. Recebe itens brutos extraídos pelo Gemini (por fornecedor)
  2. Agrupa itens equivalentes entre fornecedores (fuzzy matching semântico)
  3. Padroniza nomes, unidades e quantidades
  4. Aplica Regra de Proporção quando embalagens diferem (COM CAUTELA)
  5. Valida contra catálogo Supabase (se disponível) — nome, unidade e MARCA

Anti-Rate-Limit:
  • Retry exponencial com jitter aleatório
  • Detecção de HTTP 429 e extração de retry_after
  • Backoff com cap de 120s
"""
import json
import re
import time
import random
import logging
from typing import Optional
from difflib import SequenceMatcher

from pydantic import BaseModel, field_validator

from modules.llm_manager import get_cohere_client

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

ALLOWED_UNITS = ["UN", "CX", "PCT", "BB", "KG"]

_COHERE_MODEL = "command-a-reasoning-08-2025"

_PRICE_HIGH_THRESHOLD = 1_000.0


# ── Modelos Pydantic ──────────────────────────────────────────────────────────

class FornecedorData(BaseModel):
    preco_unit: Optional[float] = None
    obs:        Optional[str]   = None

    @field_validator("preco_unit", mode="before")
    @classmethod
    def coerce_float(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = re.sub(r"[R$\s]", "", v).replace(".", "").replace(",", ".").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None


class NormalizedItem(BaseModel):
    id:           int
    item:         str
    marca:        Optional[str]             = None
    quantidade:   float                     = 1.0
    unidade:      str                       = "UN"
    fornecedores: dict[str, FornecedorData] = {}
    observacao:   Optional[str]             = None

    @field_validator("quantidade", mode="before")
    @classmethod
    def coerce_float(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 1.0

    @field_validator("unidade", mode="before")
    @classmethod
    def normalize_unit(cls, v):
        return _normalize_unit(v or "UN")

    @field_validator("item", mode="before")
    @classmethod
    def clean_name(cls, v):
        if not v:
            return ""
        return " ".join(str(v).split()).upper()

    @field_validator("fornecedores", mode="before")
    @classmethod
    def coerce_fornecedores(cls, v):
        if not isinstance(v, dict):
            return {}
        result = {}
        for k, val in v.items():
            if isinstance(val, dict):
                result[k] = val
            else:
                result[k] = {"preco_unit": None, "obs": None}
        return result


# ── Utilitários ───────────────────────────────────────────────────────────────

def _normalize_unit(unit: str) -> str:
    """Mapeia qualquer unidade para uma das 5 permitidas."""
    if not unit:
        return "UN"
    u = unit.strip().upper()
    mapping = {
        "UN": "UN", "UND": "UN", "UNID": "UN", "UNIDADE": "UN",
        "CX": "CX", "CAIXA": "CX",
        "PCT": "PCT", "PACOTE": "PCT", "PC": "PCT", "PAC": "PCT",
        "FD": "PCT", "FARDO": "PCT", "RESMA": "PCT", "RSM": "PCT",
        "BB": "BB", "BOMBONA": "BB", "BD": "BB",
        "GL": "BB", "GALÃO": "BB", "GALAO": "BB",
        "KG": "KG", "KILO": "KG", "QUILO": "KG",
        "L": "UN", "LT": "UN", "LITRO": "UN",
        "M": "UN", "MT": "UN", "METRO": "UN",
        "M2": "UN", "ROLO": "UN", "RL": "UN",
    }
    return mapping.get(u, "UN")


def _parse_json_response(text: str) -> list:
    """Parse tolerante a markdown residual."""
    if not text:
        return []
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                items = obj.get("items", obj.get("data", obj.get("result", None)))
                if isinstance(items, list):
                    return items
            return []
        except json.JSONDecodeError:
            pass
    logger.error("[Cohere] Não foi possível parsear JSON. Primeiros 400 chars: %s", text[:400])
    return []


def _validate_normalized_items(raw_items: list) -> list:
    """Valida e coerce cada item normalizado com Pydantic."""
    validated = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            obj  = NormalizedItem(**raw)
            data = obj.model_dump()
            data["fornecedores"] = {
                k: {"preco_unit": v["preco_unit"], "obs": v["obs"]}
                for k, v in data["fornecedores"].items()
            }
            validated.append(data)
        except Exception as e:
            logger.warning("[Pydantic/Normalização] Item descartado: %s — %s", e, raw)
    return validated


def _sanity_check_normalized(items: list) -> list:
    """Sanity check de preços e unidades nos itens normalizados."""
    for item in items:
        alerts = []
        for fname, fdata in (item.get("fornecedores") or {}).items():
            if not isinstance(fdata, dict):
                continue
            p = fdata.get("preco_unit")
            if p is None:
                continue
            if p > _PRICE_HIGH_THRESHOLD:
                alerts.append("[{}] Preço suspeito: R$ {:.2f}".format(fname, p))
            if p < 0:
                alerts.append("[{}] Preço negativo: R$ {:.2f}".format(fname, p))

        unidade = item.get("unidade", "UN")
        if unidade not in ALLOWED_UNITS:
            item["unidade"] = "UN"
            alerts.append("Unidade '{}' corrigida para 'UN'".format(unidade))

        # Não sobrescreve is_suspect existente — apenas adiciona alertas novos
        existing_reasons = list(item.get("alert_reason") or [])
        for a in alerts:
            if a not in existing_reasons:
                existing_reasons.append(a)

        if existing_reasons:
            item["is_suspect"]   = True
            item["alert_reason"] = existing_reasons
        else:
            item.setdefault("is_suspect", False)
            item.setdefault("alert_reason", [])

    return items


def _fuzzy_match_catalog(item_name: str, catalog: list) -> tuple:
    """
    Fuzzy match contra catálogo. Retorna (nome, score, unidade, marca).

    Lógica de pontuação (da maior para menor prioridade):
      1. Igualdade exata → 1.0
      2. Item é prefixo/substring do nome do catálogo (ex: "PAPEL A4" ⊂ "PAPEL A4 C/500FLS") → 0.95
      3. Nome do catálogo é substring do item (ex: "CAIXA ARQUIVO MORTO" ⊃ "ARQUIVO MORTO") → 0.90
      4. SequenceMatcher ratio normal
    """
    if not catalog or not item_name:
        return None, 0.0, None, None

    best_name  = None
    best_score = 0.0
    best_unit  = None
    best_marca = None
    item_upper = item_name.strip().upper()

    for entry in catalog:
        nome = (entry.get("nome_oficial") or "").strip().upper()
        if not nome:
            continue

        # 1. Igualdade exata
        if item_upper == nome:
            score = 1.0
        # 2. Item é prefixo/substring do nome do catálogo
        #    Ex: "PAPEL A4" → "PAPEL A4 C/500FLS"  (item está contido no catálogo)
        elif item_upper in nome:
            # Quanto maior a cobertura, maior a confiança
            coverage = len(item_upper) / len(nome)
            score = max(0.90 + 0.05 * coverage, SequenceMatcher(None, item_upper, nome).ratio())
        # 3. Nome do catálogo é substring do item
        #    Ex: "CAIXA DE ARQUIVO MORTO PAPELÃO" ⊃ "CAIXA DE ARQUIVO MORTO"
        elif nome in item_upper:
            coverage = len(nome) / len(item_upper)
            score = max(0.85 + 0.05 * coverage, SequenceMatcher(None, item_upper, nome).ratio())
        # 4. Similaridade sequencial
        else:
            score = SequenceMatcher(None, item_upper, nome).ratio()

        if score > best_score:
            best_score = score
            best_name  = entry.get("nome_oficial")
            best_unit  = entry.get("unidade_padrao")
            best_marca = entry.get("marca_referencia") or entry.get("marca") or None

    return best_name, best_score, best_unit, best_marca


def _apply_catalog_matching(items: list, catalog: list) -> list:
    """
    Aplica fuzzy matching contra catálogo Supabase.

    Thresholds:
      RENAME_THRESHOLD (0.82): adota nome, unidade e (marca se vazia) do catálogo
      FUZZY_THRESHOLD  (0.70): adota apenas a unidade do catálogo

    A lógica de substring garante que "PAPEL A4" → "PAPEL A4 C/500FLS" com score ≥ 0.92,
    muito acima do RENAME_THRESHOLD.
    """
    FUZZY_THRESHOLD  = 0.70   # adota unidade do catálogo
    RENAME_THRESHOLD = 0.82   # adota nome + unidade + (marca se vazia)

    for item in items:
        nome = item.get("item", "")
        best_name, score, best_unit, best_marca = _fuzzy_match_catalog(nome, catalog)
        item["catalog_match"] = best_name
        item["catalog_score"] = round(score, 3)

        if score >= RENAME_THRESHOLD and best_name:
            # Alta confiança: adota nome oficial do catálogo
            item["item"] = best_name.upper()
            if best_unit and best_unit.upper() in ALLOWED_UNITS:
                item["unidade"] = best_unit.upper()
            # Adota marca do catálogo SOMENTE se o item ainda não tem marca definida
            if best_marca and not item.get("marca"):
                item["marca"] = best_marca
            logger.debug(
                "[Catálogo] '%s' → '%s' (score=%.3f)",
                nome, best_name, score,
            )

        elif score >= FUZZY_THRESHOLD and best_unit:
            # Confiança média: adota apenas a unidade do catálogo
            if best_unit.upper() in ALLOWED_UNITS:
                item["unidade"] = best_unit.upper()

        # Score baixo: não sinaliza como suspect — apenas registra internamente
        # (muitos itens legítimos ainda não estão cadastrados)

    return items


# ── Retry com backoff exponencial + jitter ────────────────────────────────────

def _call_cohere_with_retry(
    message: str,
    preamble: str = "",
    max_attempts: int = 5,
    base_delay: float = 8.0,
    temperature: float = 0.1,
    max_tokens: int = 16000,
) -> str:
    """
    Chama o Cohere via ClientV2 com retry exponencial + jitter.

    API V2 (SDK >= 5.x):
      - Usa client.chat(model, messages=[...], response_format=...)
      - messages = [{"role": "system", "content": preamble}, {"role": "user", "content": message}]
      - Resposta em response.message.content[0].text
      - response_format={"type": "json_object"} força JSON puro
    """
    client  = get_cohere_client()
    delay   = base_delay
    attempt = 0

    messages = []
    if preamble:
        messages.append({"role": "system", "content": preamble})
    messages.append({"role": "user", "content": message})

    while attempt < max_attempts:
        try:
            response = client.chat(
                model=_COHERE_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.message.content
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text") and block.text:
                        text = block.text.strip()
                        logger.debug("[Cohere] Resposta bruta (primeiros 300 chars): %s", text[:300])
                        return text
            if isinstance(content, str) and content:
                logger.debug("[Cohere] Resposta bruta (string direta, primeiros 300 chars): %s", content[:300])
                return content
            logger.error("[Cohere] Resposta vazia ou sem bloco de texto. content=%r", content)
            return ""

        except Exception as e:
            err_str = str(e).lower()
            attempt += 1

            if "429" in str(e) or ("rate" in err_str and "limit" in err_str):
                retry_after = None
                m = re.search(r'retry.after[\":\s]+(\d+)', str(e), re.IGNORECASE)
                if m:
                    retry_after = float(m.group(1)) + 2.0
                wait = retry_after or (delay + random.uniform(0, 4))
                logger.warning(
                    "[Cohere] 429 Rate Limit (tentativa %d/%d). Aguardando %.1fs...",
                    attempt, max_attempts, wait,
                )
                if attempt >= max_attempts:
                    raise RuntimeError(
                        "Rate limit do Cohere atingido após {} tentativas. "
                        "Aguarde alguns segundos e tente novamente.".format(max_attempts)
                    ) from e
                time.sleep(wait)
                delay = min(delay * 2.0, 120.0)
                continue

            if any(x in err_str for x in ["503", "502", "timeout", "unavailable", "connection"]):
                wait = delay + random.uniform(0, 4)
                logger.warning(
                    "[Cohere] Serviço indisponível (tentativa %d/%d). Aguardando %.1fs...",
                    attempt, max_attempts, wait,
                )
                if attempt >= max_attempts:
                    raise
                time.sleep(wait)
                delay = min(delay * 2.0, 60.0)
                continue

            if "model" in err_str and (
                "not found" in err_str or "invalid" in err_str or "removed" in err_str
            ):
                raise RuntimeError(
                    "Modelo Cohere '{}' não encontrado ou removido. "
                    "Verifique https://docs.cohere.com/docs/models para modelos disponíveis.".format(
                        _COHERE_MODEL
                    )
                ) from e

            logger.error("[Cohere] Erro inesperado (tentativa %d/%d): %s", attempt, max_attempts, e)
            raise


# ── Prompts ───────────────────────────────────────────────────────────────────

_NORMALIZATION_PREAMBLE = """\
Você é um especialista sênior em mapas de compras empresariais brasileiros.
Sua tarefa é cruzar orçamentos de múltiplos fornecedores e criar um mapa unificado em JSON.

════════════════════════════════════════════════════════════════
REGRA #0 — ABSOLUTA E INVIOLÁVEL: PRESERVAR TODOS OS DADOS
════════════════════════════════════════════════════════════════
Se um fornecedor tem um item com preço em seu orçamento, esse preço DEVE aparecer
no campo correspondente do fornecedor no mapa. NUNCA coloque null para um fornecedor
que claramente cotou o item. Esta é a regra mais importante.

Exemplos do que NÃO fazer:
  ✗ JAE cotou BORRACHA por R$1,10 mas você colocou null no JAE → ERRADO
  ✗ JAE cotou PAPEL A4 por R$24,90 (50 resmas) mas você colocou null no JAE → ERRADO
  ✗ JAE cotou PILHA AA C/4 por R$21,60 mas você colocou null no JAE → ERRADO

════════════════════════════════════════════════════════════════
REGRA #1 — AGRUPAMENTO SEMÂNTICO
════════════════════════════════════════════════════════════════
Agrupe itens equivalentes mesmo com nomes diferentes entre fornecedores.
Exemplos de equivalência:
  - "CAIXA ARQUIVO MORTO" = "CX ARQUIVO" = "ARQUIVO MORTO"
  - "PILHA ALCALINA AA C/4" = "PILHA AA PEQUENA C/4" = "PILHA DURACELL AA CARTELA C/4"
  - "BORRACHA PEQUENA" = "BORRACHA BRANCA PEQUENA" = "BORRACHA 40"
  - "BALDE 8L" do forn. A = "BALDE PLÁSTICO 8L" do forn. B

════════════════════════════════════════════════════════════════
REGRA #2 — NOMES DESCRITIVOS E PADRONIZADOS
════════════════════════════════════════════════════════════════
Use nomes COMPLETOS e PADRONIZADOS em MAIÚSCULAS.
  - INCLUA: tipo, tamanho, capacidade, material quando relevante
  - Bons: "CAIXA DE ARQUIVO MORTO", "BORRACHA PEQUENA", "COPO 200ML PP",
          "PILHA ALCALINA AA PEQUENA C/4", "BALDE PLASTICO 8L VERDE", "ESTILETE LARGO"
  - Ruins: "CAIXA", "BORRACHA", "COPO", "PILHA", "BALDE"

════════════════════════════════════════════════════════════════
REGRA #3 — UNIDADES: SOMENTE UN, CX, PCT, BB, KG
════════════════════════════════════════════════════════════════
  - Resma de papel → PCT (cada resma = 1 PCT)
  - Fardo → PCT
  - Pacote c/N unidades (pilha c/4, copo c/100) → PCT
  - COPO DESCARTÁVEL: se cotado em peças (100 copos, 200 copos) → UN
  - Bombona/Galão de LÍQUIDO → BB (ex: hipoclorito 5L em bombona)
  - BALDE como PRODUTO (balde plástico para uso) → UN ← NUNCA BB!
  - Item individual avulso → UN

════════════════════════════════════════════════════════════════
REGRA #4 — PREÇO POR EMBALAGEM DE VENDA (CRÍTICA)
════════════════════════════════════════════════════════════════
O preço unitário é SEMPRE por embalagem de venda, como declarado pelo fornecedor.
  ✓ Pilha c/4 a R$18,64 → preco_unit = 18.64 (NÃO 4.66!)
  ✓ Copo c/100 a R$5,37 → preco_unit = 5.37
  ✓ PAPEL A4: se orçamento mostra 50 resmas a R$24,90 cada → preco_unit = 24.90
  ✗ NUNCA divida o preço pelo conteúdo da embalagem
  ✗ NUNCA multiplique preço unitário pela quantidade do lote

════════════════════════════════════════════════════════════════
REGRA #5 — PILHAS (REGRA ESPECIAL)
════════════════════════════════════════════════════════════════
  - Vendidas em PACOTES (C/2, C/4, etc.) → unidade = PCT
  - Preço: valor do PACOTE inteiro (NÃO dividir por número de pilhas)
  - Nome deve incluir: tipo (AA/AAA), tamanho (PEQUENA/PALITO), e "C/4" ou "C/2"
    • AA = PEQUENA, AAA = PALITO

  NORMALIZAÇÃO DE EMBALAGEM (C/2 ↔ C/4):
  - Quando fornecedores cotaram tamanhos diferentes do MESMO tipo de pilha, unifique tudo
    na embalagem PADRÃO (a da maioria dos fornecedores ou do orçamento aprovado).
  - Exemplo: Forn A cotou C/2 a R$11,11 e Forn B cotou C/4 a R$18,64:
    → Padrão = C/4 (maioria ou aprovado)
    → preco_unit do Forn A = R$11,11 × (4/2) = R$22,22 (normalizado para C/4) ✓
    → Registre na observacao: "MINAS: normalizado de C/2 para C/4 (R$11,11 × 2 = R$22,22)"
  - ATENÇÃO: se o 3º fornecedor também cotou, inclua o preço dele normalizado!

  QUANTIDADE DE PILHAS — REGRA CRÍTICA:
  - A QUANTIDADE no mapa representa a NECESSIDADE DA EMPRESA, não a soma dos fornecedores.
  - Use a quantidade do orçamento aprovado (se houver), ou a quantidade da maioria.
  - NUNCA some as quantidades dos fornecedores entre si.
  - Exemplo CORRETO:
      SMAIS (aprovado) cotou 3 pacotes C/4 → quantidade no mapa = 3
      MINAS cotou 6 pacotes C/2 (equivale a 3 C/4) → confirma 3
      JAE cotou 6 pacotes C/2 (equivale a 3 C/4) → confirma 3
      → quantidade final = 3 PCT C/4 ✓
  - Exemplo ERRADO:
      NÃO some: 3 + 3 + 2 = 8 PCT → ERRADO, isso não representa a necessidade real

════════════════════════════════════════════════════════════════
REGRA #6 — BALDES (REGRA ESPECIAL)
════════════════════════════════════════════════════════════════
  - Balde PLÁSTICO como PRODUTO → unidade = UN (é um item individual!)
  - Balde/Bombona como EMBALAGEM de líquido → unidade = BB
  - NÃO crie itens de balde que não existem nos orçamentos originais

════════════════════════════════════════════════════════════════
REGRA #7 — REGRA DE PROPORÇÃO (COM CAUTELA)
════════════════════════════════════════════════════════════════
Aplique proporção SOMENTE quando fornecedores cotaram tamanhos REALMENTE diferentes:
  EXEMPLO VÁLIDO: Forn A cotou 1KG a R$5, Forn B cotou 500g a R$3 → normalize para 1KG
  EXEMPLO VÁLIDO: Forn A cotou pilha C/2 a R$11,11, Forn B cotou pilha C/4 a R$18,64 → normalize para C/4

  ARMADILHA DO PAPEL A4 — LEIA COM ATENÇÃO:
    Se o Forn A cotou 1 resma a R$25,50 e o Forn B cotou 1 resma a R$23,00
    e o Forn C (JAE) cotou "50 resmas" com preço UNITÁRIO de R$24,90 por resma:
    → O preço UNITÁRIO do JAE já é R$24,90 por resma!
    → NÃO multiplique por 2 ou por 50!
    → O mapa deve ter: MINAS=25.50, SMAIS=23.00, JAE=24.90 (todos por resma)
    → A quantidade no mapa é a necessidade da empresa (ex: 100 resmas)
    → A observacao pode indicar "JAE cotou em lote de 50 resmas"

════════════════════════════════════════════════════════════════
REGRA #8 — FORNECEDORES AUSENTES
════════════════════════════════════════════════════════════════
  - Fornecedor que NÃO cotou o item → preco_unit: null
  - Fornecedor que cotou o item → preco_unit: OBRIGATÓRIO ter o valor!
  - Antes de colocar null, confirme que o fornecedor realmente não tinha o item

════════════════════════════════════════════════════════════════
REGRA #9 — MARCAS
════════════════════════════════════════════════════════════════
  - Preserve as marcas extraídas dos orçamentos
  - Se o catálogo indica uma marca de referência, use-a como prioridade
  - Se vários fornecedores indicam a mesma marca, use-a
  - Se diferem, use a do catálogo → a mais comum → a do fornecedor de referência
  - Marcas conhecidas: FBOX (arquivo morto), BRW (borracha), ECOCOPPO/CIS (copo),
    CIS (estilete), CHAMEX (papel A4), DURACELL (pilha)

════════════════════════════════════════════════════════════════
REGRA #10 — FORMATO DE SAÍDA
════════════════════════════════════════════════════════════════
Retorne SEMPRE um objeto JSON válido com a chave "items".
"""

_NORMALIZATION_MESSAGE_TEMPLATE = """\
Crie o mapa de compras unificado cruzando os orçamentos dos fornecedores abaixo.

FORNECEDORES ({n_fornecedores}): {nomes_fornecedores}

{dados_fornecedores}

LISTA DE REFERÊNCIA (itens que precisamos comprar — use como guia de nomes, quantidades e unidades):
{lista_referencia}

{catalogo_context}

{preferences}

Retorne um objeto JSON com a chave "items" onde cada elemento tem EXATAMENTE esta estrutura:
{{
  "id": inteiro começando em 1,
  "item": "NOME DESCRITIVO COMPLETO MAIÚSCULO",
  "marca": "marca ou null",
  "quantidade": número float (quantidade de embalagens que precisamos comprar),
  "unidade": "UN ou CX ou PCT ou BB ou KG",
  "fornecedores": {{
    "{exemplo_forn}": {{"preco_unit": número_ou_null, "obs": null}},
    ...um campo para CADA fornecedor...
  }},
  "observacao": "nota sobre ajuste de proporção ou null"
}}

CHECKLIST ANTES DE RETORNAR:
1. ✓ Todos os fornecedores aparecem em CADA item (com null se realmente não cotaram)
2. ✓ Nenhum fornecedor que cotou o item foi omitido (preco_unit != null)
3. ✓ Preços de pilhas = valor do pacote inteiro (não dividido por 4 ou 2)
4. ✓ Papel A4: preço por RESMA individual (não multiplicado pela quantidade do lote)
5. ✓ Baldes como produto têm unidade=UN
6. ✓ Copos descartáveis cotados em peças têm unidade=UN
7. ✓ Nomes incluem características relevantes (tamanho, capacidade, etc.)
8. ✓ Marcas preservadas conforme catálogo ou orçamentos originais
9. ✓ NÃO criar itens que não existem nos orçamentos
10. ✓ QUANTIDADE = necessidade da empresa (da lista de referência, ou do aprovado, ou da maioria)
    NÃO é a soma das quantidades de todos os fornecedores
    Pilhas: se SMAIS cotou 3 C/4, MINAS cotou 6 C/2 e JAE cotou 6 C/2 → todos equivalem a 3 C/4 → quantidade = 3

Nomes exatos dos fornecedores a usar como chaves: {nomes_fornecedores}
"""


# ── Função Pública ────────────────────────────────────────────────────────────

def normalize_and_match(
    supplier_items: dict,
    reference_list=None,
    preferences_context: str = "",
    catalog: list = None,
) -> list:
    """
    Normaliza e cruza itens de múltiplos fornecedores via Cohere.

    Pipeline:
      1. Serializa dados de todos os fornecedores (com nomes reais)
      2. Cohere cria mapa unificado aplicando todas as regras
      3. Validação Pydantic + sanity check
      4. Fuzzy match contra catálogo Supabase (nome, unidade e marca)
      5. Guardrail de plausibilidade (retry automático se < 30% dos itens)

    Parâmetros:
      supplier_items      — dict {nome_fornecedor: [lista_de_itens]}
      reference_list      — lista de referência [{item, quantidade, unidade}] (opcional)
      preferences_context — contexto de correções aprendidas
      catalog             — catálogo oficial do Supabase para fuzzy matching
    """
    suppliers         = list(supplier_items.keys())
    total_input_items = sum(len(v) for v in supplier_items.values())

    # Serializa dados com nomes reais dos fornecedores (mais claro para o modelo)
    dados_formatados = ""
    for name, items in supplier_items.items():
        dados_formatados += "\n=== ORÇAMENTO DO FORNECEDOR: {} ===\n".format(name)
        dados_formatados += "Total de itens extraídos: {}\n".format(len(items))
        dados_formatados += json.dumps(items, ensure_ascii=False, indent=2)
        dados_formatados += "\n"

    # Monta contexto do catálogo (com marca de referência)
    catalogo_context = _build_catalog_context_for_prompt(catalog)

    ref_str = (
        json.dumps(reference_list, ensure_ascii=False, indent=2)
        if reference_list
        else "Não fornecida — use as quantidades dos orçamentos como referência."
    )

    message = _NORMALIZATION_MESSAGE_TEMPLATE.format(
        n_fornecedores=len(suppliers),
        nomes_fornecedores=json.dumps(suppliers, ensure_ascii=False),
        dados_fornecedores=dados_formatados,
        lista_referencia=ref_str,
        catalogo_context=catalogo_context,
        preferences=(
            "PREFERÊNCIAS DO USUÁRIO (aplique obrigatoriamente):\n" + preferences_context
            if preferences_context
            else ""
        ),
        exemplo_forn=suppliers[0] if suppliers else "FORNECEDOR",
    )

    raw = _call_cohere_with_retry(
        message=message,
        preamble=_NORMALIZATION_PREAMBLE,
        temperature=0.05,
        max_tokens=16000,
    )

    # Parse — Cohere pode retornar {"items": [...]} ou wrappers alternativos
    items = _parse_response_to_items(raw)

    if not items:
        logger.error(
            "[Cohere/Normalização] Parse retornou lista vazia. "
            "Raw (primeiros 500 chars): %s", raw[:500] if raw else "<vazio>"
        )

    # ── Guardrail de plausibilidade ───────────────────────────────────────────
    # Se temos poucos itens em relação ao esperado, tenta novamente
    min_expected = max(1, int(total_input_items * 0.25))
    if len(items) < min_expected:
        logger.warning(
            "[Cohere/Normalização] Resultado suspeito: %d itens vs %d de entrada "
            "(mín. esperado %d). Retentar...",
            len(items), total_input_items, min_expected,
        )
        raw2 = _call_cohere_with_retry(
            message=message,
            preamble=_NORMALIZATION_PREAMBLE,
            temperature=0.05,
            max_tokens=16000,
        )
        items2 = _parse_response_to_items(raw2)
        if isinstance(items2, list) and len(items2) > len(items):
            logger.info(
                "[Cohere/Normalização] Retry retornou %d itens (anterior: %d)",
                len(items2), len(items),
            )
            items = items2
        else:
            logger.warning(
                "[Cohere/Normalização] Retry também retornou poucos itens: %d",
                len(items2) if items2 else 0,
            )

    # ── Garante que todos os fornecedores existam em todos os itens ──────────
    for item in items:
        forn = item.get("fornecedores") or {}
        for sname in suppliers:
            if sname not in forn:
                forn[sname] = {"preco_unit": None, "obs": None}
        item["fornecedores"] = forn

    # ── Remap fornecedor_1..N → nomes reais (caso o modelo use índices) ──────
    key_map = {"fornecedor_{}".format(i + 1): name for i, name in enumerate(suppliers)}
    for i, name in enumerate(suppliers):
        key_map["fornecedor {}".format(i + 1)] = name
        key_map["forn_{}".format(i + 1)]       = name
        key_map["forn{}".format(i + 1)]        = name
        key_map[name.lower()]                   = name
        key_map[name.upper()]                   = name

    for item in items:
        if "fornecedores" in item and isinstance(item["fornecedores"], dict):
            remapped = {}
            for k, v in item["fornecedores"].items():
                real_key = key_map.get(k, key_map.get(k.lower(), k))
                remapped[real_key] = v
            item["fornecedores"] = remapped

    # Validação Pydantic
    items = _validate_normalized_items(items)
    # Sanity check
    items = _sanity_check_normalized(items)
    # Fuzzy catalog matching (aplica nomes, unidades e marcas do catálogo)
    if catalog:
        items = _apply_catalog_matching(items, catalog)

    return items


def _build_catalog_context_for_prompt(catalog: list) -> str:
    """
    Monta o bloco de contexto do catálogo para inserção no prompt do Cohere.
    Inclui nome_oficial, unidade_padrao e marca_referencia.
    """
    if not catalog:
        return ""
    entries = []
    for entry in catalog[:60]:
        nome = entry.get("nome_oficial", "")
        if not nome:
            continue
        und   = entry.get("unidade_padrao", "UN")
        marca = entry.get("marca_referencia") or entry.get("marca", "")
        if marca:
            entries.append("  - {} ({}) [marca: {}]".format(nome, und, marca))
        else:
            entries.append("  - {} ({})".format(nome, und))

    if not entries:
        return ""

    return (
        "CATÁLOGO OFICIAL DE PRODUTOS (nomes, unidades e marcas de referência — prioridade máxima):\n"
        + "\n".join(entries)
        + "\n\nSe um item extraído corresponder a um do catálogo:\n"
        "  1. Use o NOME EXATO do catálogo\n"
        "  2. Use a UNIDADE do catálogo\n"
        "  3. Use a MARCA do catálogo (se o item não tiver marca definida)\n"
    )


def _parse_response_to_items(raw: str) -> list:
    """Parse robusto da resposta do Cohere para lista de itens."""
    if not raw:
        return []
    items = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("items", "data", "result", "mapa", "compras", "output"):
                candidate = data.get(key)
                if isinstance(candidate, list) and candidate:
                    items = candidate
                    break
            if not items:
                for v in data.values():
                    if isinstance(v, list) and v:
                        items = v
                        break
    except (json.JSONDecodeError, TypeError):
        items = _parse_json_response(raw)

    if not isinstance(items, list):
        items = []
    return items

