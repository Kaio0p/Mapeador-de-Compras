# -*- coding: utf-8 -*-
"""
groq_processor.py — Motor de Texto e Normalizacao via Groq (LPU 128k)
=======================================================================
Responsavel pela EXTRACAO de texto e NORMALIZACAO/CRUZAMENTO dos orcamentos.

Roteamento do sistema:
  PDF nativo    -> extract_items_from_text()   (este modulo - Groq LPU)
  PDF escaneado -> extract_items_from_images() (gemini_processor.py - OCR)
  Normalizacao  -> normalize_and_match()       (este modulo - Groq LPU)
  Auditoria     -> audit_purchase_map()        (gemini_processor.py - contexto enorme)

Vantagens do Groq para extracao/normalizacao:
  - LPU inference - tokens/s muito mais rapido que GPU tradicional
  - llama-3.3-70b-versatile tem 128k de contexto - PDF inteiro cabe num unico prompt
  - JSON mode nativo (response_format) elimina markdown/texto extra

Anti-Rate-Limit:
  - Retry exponencial com jitter aleatorio
  - Deteccao de "retry_after" no corpo do erro 429
  - Backoff com cap de 120s

Anti-Alucinacao (camadas locais):
  - response_format={"type": "json_object"} em todas as chamadas
  - Validacao Pydantic pos-parse
  - Sanity check de precos e unidades
  - Guardrail de plausibilidade (retry se menos de 30% dos itens retornados)
  A auditoria semantica final fica no gemini_processor.audit_purchase_map().
"""
import json
import re
import time
import random
import logging
from typing import Optional
from difflib import SequenceMatcher

from pydantic import BaseModel, field_validator

from modules.llm_manager import get_groq_client

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

ALLOWED_UNITS = ["UN", "CX", "PCT", "BB", "KG"]

# Modelo padrao — pode ser sobrescrito via st.secrets["GROQ_MODEL"]
# llama-3.3-70b-versatile: 128k context, excelente raciocinio
_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Limite de tokens do texto enviado ao Groq.
# 128k context window = ~96k tokens de input (~72k chars UTF-8 conservador).
# Usamos 80k chars para deixar margem para o prompt de sistema e a resposta.
_MAX_TEXT_CHARS = 80_000

_PRICE_HIGH_THRESHOLD = 1_000.0


def _get_model() -> str:
    """Retorna o modelo Groq configurado, com fallback para o padrao."""
    try:
        import streamlit as st
        return st.secrets.get("GROQ_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL
    except Exception:
        return _DEFAULT_MODEL


# ── Modelos Pydantic ──────────────────────────────────────────────────────────

class ExtractionItem(BaseModel):
    item:            str
    marca:           Optional[str]   = None
    quantidade:      float           = 1.0
    unidade:         str             = "UN"
    preco_unitario:  Optional[float] = None
    preco_total:     Optional[float] = None
    observacao:      Optional[str]   = None

    @field_validator("quantidade", "preco_unitario", "preco_total", mode="before")
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


# ── Utilitarios ───────────────────────────────────────────────────────────────

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
        "GL": "BB", "GALAO": "BB",
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
            return [obj] if isinstance(obj, dict) else []
        except json.JSONDecodeError:
            pass
    logger.error("[Groq] Nao foi possivel parsear JSON. Primeiros 300 chars: %s", text[:300])
    return []


def _validate_extraction_items(raw_items: list) -> list:
    validated = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            obj = ExtractionItem(**raw)
            validated.append(obj.model_dump())
        except Exception as e:
            logger.warning("[Pydantic/Extracao] Item descartado: %s", e)
    return validated


def _validate_normalized_items(raw_items: list) -> list:
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
            logger.warning("[Pydantic/Normalizacao] Item descartado: %s", e)
    return validated


def _sanity_check_extraction(items: list) -> list:
    for item in items:
        alerts     = []
        p_unit     = item.get("preco_unitario")
        p_total    = item.get("preco_total")
        quantidade = item.get("quantidade") or 1.0

        if p_unit is not None and p_unit > _PRICE_HIGH_THRESHOLD:
            alerts.append(
                "Preco unitario suspeito: R$ {:.2f} (limiar R$ {:.0f})".format(
                    p_unit, _PRICE_HIGH_THRESHOLD
                )
            )
        if p_unit is not None and p_total is not None and p_unit > p_total and quantidade >= 1.0:
            alerts.append(
                "Preco unitario R$ {:.2f} maior que total R$ {:.2f}".format(p_unit, p_total)
            )
        if p_unit is not None and p_total is not None and p_total > 0:
            expected = p_unit * quantidade
            desvio   = abs(expected - p_total) / p_total
            if desvio > 0.25:
                alerts.append(
                    "Total inconsistente: {}xR${:.2f}=R${:.2f} != total R${:.2f} (desvio {:.0f}%)".format(
                        quantidade, p_unit, expected, p_total, desvio * 100
                    )
                )
        unidade = item.get("unidade", "UN")
        if unidade not in ALLOWED_UNITS:
            item["unidade"] = "UN"
            alerts.append("Unidade '{}' nao reconhecida -> corrigida para 'UN'".format(unidade))

        item["is_suspect"]   = bool(alerts)
        item["alert_reason"] = alerts
    return items


def _sanity_check_normalized(items: list) -> list:
    for item in items:
        alerts = []
        for fname, fdata in (item.get("fornecedores") or {}).items():
            if not isinstance(fdata, dict):
                continue
            p = fdata.get("preco_unit")
            if p is None:
                continue
            if p > _PRICE_HIGH_THRESHOLD:
                alerts.append("[{}] Preco suspeito: R$ {:.2f}".format(fname, p))
            if p < 0:
                alerts.append("[{}] Preco negativo: R$ {:.2f}".format(fname, p))

        unidade = item.get("unidade", "UN")
        if unidade not in ALLOWED_UNITS:
            item["unidade"] = "UN"
            alerts.append("Unidade '{}' corrigida para 'UN'".format(unidade))

        item["is_suspect"]   = bool(alerts)
        item["alert_reason"] = alerts
    return items


def _fuzzy_match_catalog(item_name: str, catalog: list) -> tuple:
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
                "Item nao encontrado no catalogo (melhor: '{}', {:.0%})".format(best_name, score)
            )
            item["alert_reason"] = reasons
    return items


# ── Retry com backoff exponencial + jitter ────────────────────────────────────

def _call_groq_with_retry(
    messages: list,
    model: str = None,
    max_attempts: int = 5,
    base_delay: float = 5.0,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """
    Chama o Groq com retry exponencial + jitter aleatorio.
    Sempre usa response_format={"type": "json_object"}.
    """
    client  = get_groq_client()
    model   = model or _get_model()
    delay   = base_delay
    attempt = 0

    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            err_str = str(e).lower()
            attempt += 1

            # Rate limit
            if "429" in str(e) or "rate_limit" in err_str or "rate limit" in err_str:
                retry_after = None
                m = re.search(r'retry.after[\":\s]+(\d+)', str(e), re.IGNORECASE)
                if m:
                    retry_after = float(m.group(1)) + 2.0
                wait = retry_after or (delay + random.uniform(0, 3))
                logger.warning(
                    "[Groq] 429 Rate Limit (tentativa %d/%d). Aguardando %.1fs...",
                    attempt, max_attempts, wait,
                )
                if attempt >= max_attempts:
                    raise RuntimeError(
                        "Rate limit do Groq atingido apos {} tentativas. "
                        "Aguarde alguns segundos e tente novamente.".format(max_attempts)
                    ) from e
                time.sleep(wait)
                delay = min(delay * 2.0, 120.0)
                continue

            # Servico indisponivel
            if any(x in err_str for x in ["503", "502", "timeout", "service unavailable", "connection"]):
                wait = delay + random.uniform(0, 3)
                logger.warning(
                    "[Groq] Servico indisponivel (tentativa %d/%d). Aguardando %.1fs...",
                    attempt, max_attempts, wait,
                )
                if attempt >= max_attempts:
                    raise
                time.sleep(wait)
                delay = min(delay * 2.0, 60.0)
                continue

            # Modelo nao encontrado
            if "model_not_found" in err_str or "model not found" in err_str:
                raise RuntimeError(
                    "Modelo Groq '{}' nao encontrado. "
                    "Verifique GROQ_MODEL nos secrets ou use 'llama-3.3-70b-versatile'.".format(model)
                ) from e

            raise


# ── Prompts ───────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = (
    "Voce e um assistente especializado em analise de orcamentos de compras empresariais brasileiros. "
    "Voce tem acesso a uma janela de contexto de 128k tokens -- use-a integralmente para ler o "
    "documento completo antes de responder. "
    "Responda SEMPRE com JSON valido -- sem markdown, sem texto extra."
)

_EXTRACTION_USER = (
    "Analise o orcamento abaixo e extraia TODOS os itens cotados. "
    "Leia o documento INTEIRO antes de responder -- nao truncar.\n\n"
    "Retorne um objeto JSON com a chave \"items\" contendo um array. "
    "Cada item do array deve ter EXATAMENTE estes campos:\n\n"
    "- \"item\": nome CURTO e SIMPLES em MAIUSCULAS (ex: \"COPO 200ML PP\", \"PAPEL A4 C/500FLS\", \"HIPOCLORITO 5% 5L\")\n"
    "- \"marca\": string ou null -- UMA UNICA marca\n"
    "- \"quantidade\": numero (float) -- NUNCA string\n"
    "- \"unidade\": APENAS \"UN\", \"CX\", \"PCT\", \"BB\" ou \"KG\"\n"
    "  Resma -> PCT | Fardo -> PCT | Pacote c/N -> PCT | Bombona/Balde -> BB | Galao -> BB\n"
    "- \"preco_unitario\": numero (float) -- preco POR EMBALAGEM exatamente como no documento (nao divida)\n"
    "- \"preco_total\": numero ou null\n"
    "- \"observacao\": string ou null -- apenas observacoes do orcamento (frete, prazo)\n\n"
    "REGRAS CRITICAS:\n"
    "- Numeros NUNCA como strings\n"
    "- Preco por embalagem de venda -- NAO divida: pilha c/4 = R$18,64 -> 18.64 (nao 4.66)\n"
    "- Nome curto -- sem gramatura desnecessaria, sem detalhes de cor obvia\n"
    "- Use null para campos ausentes -- NUNCA invente\n"
    "- Ignore cabecalhos, totais e rodapes\n\n"
    "{preferences}\n\n"
    "ORCAMENTO:\n"
    "{texto}"
)

_NORMALIZATION_SYSTEM = (
    "Voce e um especialista em mapas de compras empresariais brasileiros. "
    "Sua tarefa e cruzar orcamentos de multiplos fornecedores em um mapa unificado.\n\n"
    "Regras absolutas:\n"
    "1. Agrupe itens equivalentes mesmo com nomes diferentes (ex: \"HIPOCLORITO 5% 5L\" = \"CLORO ATIVO 5L\")\n"
    "2. Use nomes CURTOS e PADRONIZADOS em MAIUSCULAS\n"
    "3. Unidades SOMENTE: UN, CX, PCT, BB, KG\n"
    "4. Preco por embalagem de venda -- NAO normalize, nao divida\n"
    "5. REGRA DE EQUIVALENCIA DE PROPORCAO: se um fornecedor vende 2 pacotes de 500g e outro vende "
    "1KG, eles sao equivalentes -- use a mesma unidade e ajuste a quantidade para comparacao justa\n"
    "6. Retorne SEMPRE JSON valido com a chave \"items\""
)

_NORMALIZATION_USER = (
    "Crie o mapa de compras unificado com os dados abaixo.\n\n"
    "FORNECEDORES ({n_fornecedores}):\n"
    "{dados_fornecedores}\n\n"
    "LISTA DE REFERENCIA (itens desejados):\n"
    "{lista_referencia}\n\n"
    "Retorne um objeto JSON com a chave \"items\" onde cada elemento tem EXATAMENTE:\n"
    "{{\n"
    "  \"id\": inteiro comecando em 1,\n"
    "  \"item\": \"NOME CURTO MAIUSCULO\",\n"
    "  \"marca\": \"marca ou null\",\n"
    "  \"quantidade\": numero (float),\n"
    "  \"unidade\": \"UN|CX|PCT|BB|KG\",\n"
    "  \"fornecedores\": {{\n"
    "    \"NOME_FORNECEDOR\": {{\"preco_unit\": numero_ou_null, \"obs\": null}}\n"
    "  }},\n"
    "  \"observacao\": null\n"
    "}}\n\n"
    "REGRA DE PROPORCAO -- exemplos:\n"
    "- 2x500g = 1KG -> unifique como 1KG e ajuste o preco (some os dois)\n"
    "- 1 caixa c/12 unid e 1 caixa c/6 -> normalize para caixa de 6 (menor denominador)\n\n"
    "IMPORTANTE:\n"
    "- Todos os numeros como float, NUNCA strings\n"
    "- Fornecedores que nao cotaram o item -> preco_unit: null\n"
    "- Retorne APENAS o JSON, sem markdown\n\n"
    "{preferences}\n\n"
    "DADOS DOS FORNECEDORES:\n"
    "{dados}"
)


# ── Funcoes Publicas ──────────────────────────────────────────────────────────

def extract_items_from_text(text: str, preferences_context: str = "") -> list:
    """
    Extrai itens de um PDF com texto nativo via Groq (ultra-rapido, 128k context).

    Parametros:
      text                -- texto extraido do PDF por pdf_extractor.py
      preferences_context -- contexto de correcoes aprendidas

    A janela de 128k do llama-3.3-70b-versatile permite enviar PDFs completos
    sem truncamento, eliminando a perda de itens que ocorria com o limite de 14k.

    Retorna lista de dicts com campos padrao + is_suspect/alert_reason.
    """
    # Usa ate 80k chars -- equivale a ~60k tokens, deixando margem para prompt+resposta
    texto_truncado = text[:_MAX_TEXT_CHARS]
    if len(text) > _MAX_TEXT_CHARS:
        logger.warning(
            "[Groq] Texto truncado de %d para %d chars (documento muito longo).",
            len(text), _MAX_TEXT_CHARS,
        )

    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": _EXTRACTION_USER.format(
                preferences=preferences_context or "Nenhuma preferencia registrada ainda.",
                texto=texto_truncado,
            ),
        },
    ]

    raw = _call_groq_with_retry(messages, temperature=0.05, max_tokens=8192)

    try:
        data  = json.loads(raw)
        items = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = []
    except (json.JSONDecodeError, TypeError):
        items = _parse_json_response(raw)

    items = _validate_extraction_items(items)
    items = _sanity_check_extraction(items)
    return items


def normalize_and_match(
    supplier_items: dict,
    reference_list=None,
    preferences_context: str = "",
    catalog: list = None,
) -> list:
    """
    Normaliza e cruza itens de multiplos fornecedores via Groq.

    Pipeline:
      1. Serializa dados de todos os fornecedores num unico prompt
      2. Groq cria mapa unificado aplicando Regra de Proporcao
      3. Remap fornecedor_1..N -> nomes reais
      4. Validacao Pydantic + sanity check
      5. Fuzzy match contra catalogo Supabase (se fornecido)
      6. Guardrail de plausibilidade (retry automatico se poucos itens)
    """
    suppliers         = list(supplier_items.keys())
    total_input_items = sum(len(v) for v in supplier_items.values())

    dados_numerados = "\n\n".join(
        "FORNECEDOR {} -- {}:\n{}".format(
            i + 1, name, json.dumps(items, ensure_ascii=False, indent=2)
        )
        for i, (name, items) in enumerate(supplier_items.items())
    )

    dados_chave_numerica = json.dumps(
        {
            "fornecedor_{}".format(i + 1): items
            for i, (_, items) in enumerate(supplier_items.items())
        },
        ensure_ascii=False,
        indent=2,
    )

    ref_str = (
        json.dumps(reference_list, ensure_ascii=False, indent=2)
        if reference_list
        else "Nao fornecida -- use as quantidades dos orcamentos."
    )

    messages = [
        {"role": "system", "content": _NORMALIZATION_SYSTEM},
        {
            "role": "user",
            "content": _NORMALIZATION_USER.format(
                n_fornecedores=len(suppliers),
                dados_fornecedores=dados_numerados,
                lista_referencia=ref_str,
                preferences=preferences_context or "Nenhuma preferencia registrada ainda.",
                dados=dados_chave_numerica,
            ),
        },
    ]

    raw = _call_groq_with_retry(messages, temperature=0.05, max_tokens=8192)

    try:
        data  = json.loads(raw)
        items = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = []
    except (json.JSONDecodeError, TypeError):
        items = _parse_json_response(raw)

    # Guardrail de plausibilidade
    min_expected = max(1, int(total_input_items * 0.30))
    if len(items) < min_expected:
        logger.warning(
            "[Groq/Normalizacao] Resultado suspeito: %d itens vs %d de entrada. Retentar...",
            len(items), total_input_items,
        )
        raw2 = _call_groq_with_retry(messages, temperature=0.05, max_tokens=8192)
        try:
            data2  = json.loads(raw2)
            items2 = data2.get("items", data2) if isinstance(data2, dict) else data2
            if isinstance(items2, list) and len(items2) > len(items):
                items = items2
        except Exception:
            pass

    # Remap fornecedor_1..N -> nomes reais
    key_map = {"fornecedor_{}".format(i + 1): name for i, name in enumerate(suppliers)}
    for item in items:
        if "fornecedores" in item and isinstance(item["fornecedores"], dict):
            item["fornecedores"] = {
                key_map.get(k, k): v for k, v in item["fornecedores"].items()
            }

    items = _validate_normalized_items(items)
    items = _sanity_check_normalized(items)

    if catalog:
        items = _apply_catalog_matching(items, catalog)

    return items
