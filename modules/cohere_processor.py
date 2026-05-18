# -*- coding: utf-8 -*-
"""
cohere_processor.py — Agente Lógico/Normalizador via Cohere command-r-plus
===========================================================================
Responsável pela NORMALIZAÇÃO e CRUZAMENTO dos orçamentos extraídos.

Vantagens do Cohere command-r-plus para normalização:
  • Especializado em RAG e tarefas estruturadas (JSON)
  • Suporte nativo a JSON mode (response_format)
  • Generous rate limits no free tier vs. Groq
  • Excelente raciocínio lógico e matemático para Regra de Proporção

Roteamento do sistema:
  Extração PDF nativo    → gemini_processor.extract_items_from_text()
  Extração PDF escaneado → gemini_processor.extract_items_from_images()
  Normalização           → cohere_processor.normalize_and_match()   ← ESTE MÓDULO
  Auditoria Final        → gemini_processor.audit_purchase_map()

Anti-Rate-Limit:
  • Retry exponencial com jitter aleatório
  • Detecção de HTTP 429 e extração de retry_after
  • Backoff com cap de 120s

Anti-Alucinação (camadas locais):
  • JSON mode nativo via cohere (response_format)
  • Validação Pydantic pós-parse
  • Sanity check de preços e unidades
  • Guardrail de plausibilidade (retry automático se < 30% dos itens)
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

# Modelo Cohere para normalização
# command-r-plus foi removido em setembro/2025.
# command-a-reasoning-08-2025: modelo de raciocínio da Cohere, ideal para
# lógica matemática (regra de 3, proporções) e estruturação JSON complexa.
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
        "BB": "BB", "BOMBONA": "BB", "BALDE": "BB", "BD": "BB",
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
            # Pode vir embrulhado em {"items": [...]}
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
    """Fuzzy match contra catálogo. Retorna (nome, score, unidade)."""
    if not catalog or not item_name:
        return None, 0.0, None
    best_name  = None
    best_score = 0.0
    best_unit  = None
    item_upper = item_name.upper()
    for entry in catalog:
        nome  = (entry.get("nome_oficial") or "").upper()
        if not nome:
            continue
        score = SequenceMatcher(None, item_upper, nome).ratio()
        if score > best_score:
            best_score = score
            best_name  = entry.get("nome_oficial")
            best_unit  = entry.get("unidade_padrao")
    return best_name, best_score, best_unit


def _apply_catalog_matching(items: list, catalog: list) -> list:
    """Aplica fuzzy matching contra catálogo e sinaliza itens sem match."""
    FUZZY_THRESHOLD = 0.82
    for item in items:
        nome = item.get("item", "")
        best_name, score, best_unit = _fuzzy_match_catalog(nome, catalog)
        item["catalog_match"] = best_name
        item["catalog_score"] = round(score, 3)
        if score >= FUZZY_THRESHOLD:
            if best_unit and best_unit.upper() in ALLOWED_UNITS:
                item["unidade"] = best_unit.upper()
        else:
            item["is_suspect"] = True
            reasons = list(item.get("alert_reason") or [])
            reasons.append(
                "Item não encontrado no catálogo (melhor: '{}', {:.0%})".format(best_name, score)
            )
            item["alert_reason"] = reasons
    return items


# ── Retry com backoff exponencial + jitter ────────────────────────────────────

def _call_cohere_with_retry(
    message: str,
    preamble: str = "",
    max_attempts: int = 5,
    base_delay: float = 8.0,
    temperature: float = 0.1,
    max_tokens: int = 8192,   # aumentado de 4096 para evitar truncamento silencioso
) -> str:
    """
    Chama o Cohere via ClientV2 com retry exponencial + jitter.

    API V2 (SDK >= 5.x):
      - Usa client.chat(model, messages=[...], response_format=...)
      - messages = [{"role": "system", "content": preamble}, {"role": "user", "content": message}]
      - Resposta em response.message.content[0].text
      - response_format={"type": "json_object"} força JSON puro

    Detecta HTTP 429 e extrai retry_after quando disponível.
    """
    client  = get_cohere_client()
    delay   = base_delay
    attempt = 0

    # Monta a lista de mensagens no formato V2
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
            # V2: response.message.content é uma lista de blocos de conteúdo.
            # Modelos de raciocínio (command-a-reasoning-*) retornam dois tipos:
            #   - ThinkingAssistantMessageResponseContentItem  → raciocínio interno (sem .text)
            #   - TextAssistantMessageResponseContentItem      → resposta final    (tem .text)
            # Filtramos pelo bloco que tem o atributo "text".
            content = response.message.content
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text") and block.text:
                        text = block.text.strip()
                        logger.debug("[Cohere] Resposta bruta (primeiros 300 chars): %s", text[:300])
                        return text
            # Fallback para string direta (modelos não-reasoning)
            if isinstance(content, str) and content:
                logger.debug("[Cohere] Resposta bruta (string direta, primeiros 300 chars): %s", content[:300])
                return content
            logger.error("[Cohere] Resposta vazia ou sem bloco de texto. content=%r", content)
            return ""

        except Exception as e:
            err_str = str(e).lower()
            attempt += 1

            # Rate limit
            if "429" in str(e) or "rate" in err_str and "limit" in err_str:
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

            # Serviço indisponível
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

            # Modelo não encontrado / inválido
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

REGRAS ABSOLUTAS:
1. Agrupe itens equivalentes mesmo com nomes diferentes
   Exemplos: "HIPOCLORITO 5% 5L" = "CLORO ATIVO 5L"; "PAPEL A4 RESMA" = "PAPEL A4 C/500FLS"
2. Use nomes CURTOS e PADRONIZADOS em MAIÚSCULAS (sem detalhes desnecessários)
3. Unidades SOMENTE: UN, CX, PCT, BB, KG
   Resma→PCT | Fardo→PCT | Pacote c/N→PCT | Bombona/Balde/Galão→BB | Galão→BB
4. Preço POR EMBALAGEM DE VENDA como declarado pelo fornecedor

5. REGRA DE EQUIVALÊNCIA CONTEXTUAL (REGRA DE 3):
   Observe o cenário de cada item cotado. Se fornecedores diferentes cotaram variações de
   peso/tamanho para a mesma necessidade (exemplo: dois fornecedores cotaram "Elástico 1KG"
   e um fornecedor cotou 2x "Elástico 500g"), NÃO separe em itens diferentes.
   Sua tarefa é identificar a INTENÇÃO DA COMPRA e agrupar todos na mesma linha:

   PASSO 1 — Eleja a UNIDADE PADRÃO:
     • Se o item constar na Lista de Referência, use OBRIGATORIAMENTE a unidade/tamanho
       que está na lista de referência como padrão.
     • Se o item NÃO estiver na lista de referência, eleja a unidade mais comum
       (majoritária) entre os fornecedores como padrão.

   PASSO 2 — Normalize o preço do fornecedor divergente via REGRA DE 3:
     • Exemplo: padrão eleito = 1KG. Fornecedor A cotou 500g a R$8,00.
       Preço normalizado = (R$8,00 / 500g) × 1000g = R$16,00 por KG.
     • Exemplo: padrão eleito = 500g. Fornecedor B cotou 1KG a R$15,00.
       Preço normalizado = (R$15,00 / 1000g) × 500g = R$7,50 por 500g.
     • Use sempre proporção direta: preco_normalizado = preco_original × (qtd_padrao / qtd_original)

   PASSO 3 — Registre o ajuste no campo "observacao" do item:
     • Indique qual fornecedor foi ajustado e a conversão aplicada.
     • Exemplo: "Fornecedor X: 2×500g→1KG (R$16,00 calc. via regra de 3)"

   ATENÇÃO: Esta regra só se aplica quando há INTENÇÃO CLARA de comprar o mesmo produto
   em volumes/pesos diferentes. Produtos genuinamente distintos (ex: detergente 5L e
   detergente 500ml com fins diferentes) devem permanecer em linhas separadas.

6. Fornecedores que não cotaram o item → preco_unit: null
7. Retorne SEMPRE um objeto JSON válido com a chave "items"
"""

_NORMALIZATION_MESSAGE_TEMPLATE = """\
Crie o mapa de compras unificado abaixo.

FORNECEDORES ({n_fornecedores}):
{dados_fornecedores}

LISTA DE REFERÊNCIA (itens que precisamos comprar):
{lista_referencia}

{preferences}

Retorne um objeto JSON com a chave "items" onde cada elemento tem EXATAMENTE esta estrutura:
{{
  "id": inteiro começando em 1,
  "item": "NOME CURTO MAIÚSCULO",
  "marca": "marca ou null",
  "quantidade": número float,
  "unidade": "UN ou CX ou PCT ou BB ou KG",
  "fornecedores": {{
    "NOME_EXATO_DO_FORNECEDOR": {{"preco_unit": número_ou_null, "obs": null}}
  }},
  "observacao": "nota sobre ajuste de proporção ou null"
}}

DADOS DOS FORNECEDORES (use estes nomes EXATAMENTE como chaves em "fornecedores"):
{dados_chave_numerica}

IMPORTANTE:
- Todos os números devem ser float, NUNCA strings
- Retorne APENAS o JSON, sem markdown, sem explicações fora do JSON
- Aplique a REGRA DE PROPORÇÃO sempre que as embalagens diferirem entre fornecedores
"""


# ── Função Pública ────────────────────────────────────────────────────────────

def normalize_and_match(
    supplier_items: dict,
    reference_list=None,
    preferences_context: str = "",
    catalog: list = None,
) -> list:
    """
    Normaliza e cruza itens de múltiplos fornecedores via Cohere command-r-plus.

    Pipeline:
      1. Serializa dados de todos os fornecedores
      2. Cohere cria mapa unificado aplicando Regra de Proporção
      3. Remap fornecedor_1..N → nomes reais
      4. Validação Pydantic + sanity check
      5. Fuzzy match contra catálogo Supabase (se fornecido)
      6. Guardrail de plausibilidade (retry automático se < 30% dos itens)

    Parâmetros:
      supplier_items      — dict {nome_fornecedor: [lista_de_itens]}
      reference_list      — lista de referência [{item, quantidade, unidade}] (opcional)
      preferences_context — contexto de correções aprendidas
      catalog             — catálogo oficial do Supabase para fuzzy matching

    Retorna lista de itens normalizados com campos padrão + is_suspect/alert_reason.
    """
    suppliers         = list(supplier_items.keys())
    total_input_items = sum(len(v) for v in supplier_items.values())

    # Serializa dados com índice numérico (mais robusto para o modelo)
    dados_numerados = "\n\n".join(
        "FORNECEDOR {} — {}:\n{}".format(
            i + 1, name, json.dumps(items, ensure_ascii=False, indent=2)
        )
        for i, (name, items) in enumerate(supplier_items.items())
    )

    # Versão com chaves numéricas para o modelo usar no JSON de saída
    dados_chave_numerica = json.dumps(
        {
            "fornecedor_{}".format(i + 1): {
                "nome_real": name,
                "itens": items,
            }
            for i, (name, items) in enumerate(supplier_items.items())
        },
        ensure_ascii=False,
        indent=2,
    )

    ref_str = (
        json.dumps(reference_list, ensure_ascii=False, indent=2)
        if reference_list
        else "Não fornecida — use as quantidades dos orçamentos como referência."
    )

    message = _NORMALIZATION_MESSAGE_TEMPLATE.format(
        n_fornecedores=len(suppliers),
        dados_fornecedores=dados_numerados,
        lista_referencia=ref_str,
        preferences=(
            "PREFERÊNCIAS DO USUÁRIO (aplique obrigatoriamente):\n" + preferences_context
            if preferences_context
            else ""
        ),
        dados_chave_numerica=dados_chave_numerica,
    )

    raw = _call_cohere_with_retry(
        message=message,
        preamble=_NORMALIZATION_PREAMBLE,
        temperature=0.05,
        max_tokens=8192,
    )

    # Parse — Cohere pode retornar {"items": [...]} ou o array direto
    # ou ainda wrappers alternativos como {"result": [...]} / {"mapa": [...]}
    items = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Tenta chaves conhecidas em ordem de prioridade
            for key in ("items", "data", "result", "mapa", "compras", "output"):
                candidate = data.get(key)
                if isinstance(candidate, list) and candidate:
                    items = candidate
                    break
            # Último recurso: primeiro valor que for lista
            if not items:
                for v in data.values():
                    if isinstance(v, list) and v:
                        items = v
                        break
    except (json.JSONDecodeError, TypeError):
        items = _parse_json_response(raw)

    if not isinstance(items, list):
        items = []

    if not items:
        logger.error(
            "[Cohere/Normalização] Parse retornou lista vazia. "
            "Raw (primeiros 500 chars): %s", raw[:500] if raw else "<vazio>"
        )

    # ── Guardrail de plausibilidade ───────────────────────────────────────────
    min_expected = max(1, int(total_input_items * 0.30))
    if len(items) < min_expected:
        logger.warning(
            "[Cohere/Normalização] Resultado suspeito: %d itens vs %d de entrada (mín. esperado %d). Retentar...",
            len(items), total_input_items, min_expected,
        )
        raw2 = _call_cohere_with_retry(
            message=message,
            preamble=_NORMALIZATION_PREAMBLE,
            temperature=0.05,
            max_tokens=8192,
        )
        try:
            items2 = []
            if raw2:
                data2 = json.loads(raw2)
                if isinstance(data2, list):
                    items2 = data2
                elif isinstance(data2, dict):
                    for key in ("items", "data", "result", "mapa", "compras", "output"):
                        candidate = data2.get(key)
                        if isinstance(candidate, list) and candidate:
                            items2 = candidate
                            break
                    if not items2:
                        for v in data2.values():
                            if isinstance(v, list) and v:
                                items2 = v
                                break
            if not items2:
                items2 = _parse_json_response(raw2)
            if isinstance(items2, list) and len(items2) > len(items):
                logger.info("[Cohere/Normalização] Retry retornou %d itens (anterior: %d)", len(items2), len(items))
                items = items2
            else:
                logger.warning("[Cohere/Normalização] Retry também retornou poucos itens: %d", len(items2))
        except Exception as exc:
            logger.warning("[Cohere/Normalização] Erro no parse do retry: %s", exc)

    # ── Remap fornecedor_1..N → nomes reais ──────────────────────────────────
    # O modelo às vezes usa "fornecedor_1" como chave — remapeamos para o nome real
    key_map = {"fornecedor_{}".format(i + 1): name for i, name in enumerate(suppliers)}
    for item in items:
        if "fornecedores" in item and isinstance(item["fornecedores"], dict):
            remapped = {}
            for k, v in item["fornecedores"].items():
                real_key = key_map.get(k, k)
                remapped[real_key] = v
            item["fornecedores"] = remapped

    # ── Garante que todos os fornecedores existam em todos os itens ──────────
    for item in items:
        forn = item.get("fornecedores") or {}
        for sname in suppliers:
            if sname not in forn:
                forn[sname] = {"preco_unit": None, "obs": None}
        item["fornecedores"] = forn

    # Validação Pydantic
    items = _validate_normalized_items(items)
    # Sanity check
    items = _sanity_check_normalized(items)
    # Fuzzy catalog matching
    if catalog:
        items = _apply_catalog_matching(items, catalog)

    return items
