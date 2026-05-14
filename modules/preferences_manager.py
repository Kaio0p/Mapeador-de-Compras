# -*- coding: utf-8 -*-
"""
gemini_processor.py
Extração e normalização via Gemini Flash (gemini-flash-latest — sempre atualizado).

Pipeline de validação multi-camada:
  Camada 1 — Structured Outputs: response_schema nativo do Gemini força tipos corretos
  Camada 2 — Pydantic validation: garante conformidade após parse
  Camada 3 — Sanity check: detecta alucinações de preço e unidades inválidas
"""
import json, re, time, logging
from typing import Optional, List
from difflib import SequenceMatcher

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)

# Alias que sempre aponta para o Flash mais recente (atualmente Flash 2.0+)
# Referência: https://ai.google.dev/gemini-api/docs/models
_MODEL_NAME = "gemini-flash-latest"

# Unidades permitidas no mapa de compras
ALLOWED_UNITS = ["UN", "CX", "PCT", "BB", "KG"]

# Limiar acima do qual um preço unitário é considerado suspeito para itens básicos
_PRICE_HIGH_THRESHOLD = 1000.0


def configure(api_key: str):
    genai.configure(api_key=api_key)


def _model():
    return genai.GenerativeModel(_MODEL_NAME)


# ── Camada 1: Schemas de tipagem estrita ──────────────────────────────────────

# Schema para extração de itens (array de objetos)
_EXTRACTION_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "item":          {"type": "string"},
            "marca":         {"type": "string"},
            "quantidade":    {"type": "number"},
            "unidade":       {"type": "string"},
            "preco_unitario":{"type": "number"},
            "preco_total":   {"type": "number"},
            "observacao":    {"type": "string"},
        },
        "required": ["item", "quantidade", "unidade", "preco_unitario"],
    },
}

# Schema para normalização/cruzamento
# Nota: a API Gemini não suporta "additionalProperties" — o campo fornecedores
# é tipado apenas como "object" genérico; a validação de tipos internos é feita
# pelo Pydantic (Camada 2) após o parse.
_NORMALIZATION_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":           {"type": "integer"},
            "item":         {"type": "string"},
            "marca":        {"type": "string"},
            "quantidade":   {"type": "number"},
            "unidade":      {"type": "string"},
            "fornecedores": {"type": "object"},
            "observacao":   {"type": "string"},
        },
        "required": ["id", "item", "quantidade", "unidade", "fornecedores"],
    },
}


# ── Camada 2: Modelos Pydantic para validação pós-parse ───────────────────────

class ExtractionItem(BaseModel):
    item:           str
    marca:          Optional[str]       = None
    quantidade:     float               = 1.0
    unidade:        str                 = "UN"
    preco_unitario: Optional[float]     = None
    preco_total:    Optional[float]     = None
    observacao:     Optional[str]       = None

    @field_validator("quantidade", "preco_unitario", "preco_total", mode="before")
    @classmethod
    def coerce_float(cls, v):
        """Aceita strings numéricas e as converte; devolve None para valores não parseáveis."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.replace("R$", "").replace(".", "").replace(",", ".").strip()
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
    def clean_item_name(cls, v):
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
            cleaned = v.replace("R$", "").replace(".", "").replace(",", ".").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None


class NormalizedItem(BaseModel):
    id:          int
    item:        str
    marca:       Optional[str]              = None
    quantidade:  float                      = 1.0
    unidade:     str                        = "UN"
    fornecedores: dict[str, FornecedorData] = {}
    observacao:  Optional[str]              = None

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
    def clean_item_name(cls, v):
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


# ── Camada 3: Sanity Check ─────────────────────────────────────────────────────

def _sanity_check(items: list) -> list:
    """
    Verifica sanidade dos itens extraídos, detectando alucinações de preço e dados inválidos.

    Flags adicionadas a cada item:
      - is_suspect:    True se alguma anomalia foi detectada
      - alert_reason:  Lista de strings descrevendo os problemas encontrados

    Regras verificadas:
      1. preco_unitario invulgarmente alto (> 1000 para itens de consumo básico)
      2. preco_unitario > preco_total (matematicamente impossível se quantidade >= 1)
      3. unidade não está em ALLOWED_UNITS → corrige automaticamente para "UN"
      4. preco_total inconsistente com quantidade * preco_unitario (desvio > 20%)
    """
    for item in items:
        alerts = []

        preco_unit  = item.get("preco_unitario")
        preco_total = item.get("preco_total")
        quantidade  = item.get("quantidade") or 1.0

        # Regra 1: preço unitário excessivo
        if preco_unit is not None and preco_unit > _PRICE_HIGH_THRESHOLD:
            alerts.append(
                f"Preço unitário suspeito: R$ {preco_unit:.2f} "
                f"(acima do limiar de R$ {_PRICE_HIGH_THRESHOLD:.0f} para item básico)"
            )

        # Regra 2: preço unitário maior que o total
        if preco_unit is not None and preco_total is not None:
            if preco_unit > preco_total and quantidade >= 1.0:
                alerts.append(
                    f"Preço unitário (R$ {preco_unit:.2f}) maior que o total "
                    f"(R$ {preco_total:.2f}) — possível confusão de colunas"
                )

        # Regra 3: consistência quantidade × unitário ≈ total
        if preco_unit is not None and preco_total is not None and preco_total > 0:
            expected = preco_unit * quantidade
            desvio   = abs(expected - preco_total) / preco_total
            if desvio > 0.20:
                alerts.append(
                    f"Total inconsistente: {quantidade} × R$ {preco_unit:.2f} "
                    f"= R$ {expected:.2f}, mas total declarado é R$ {preco_total:.2f} "
                    f"(desvio de {desvio*100:.0f}%)"
                )

        # Regra 4: unidade fora do conjunto permitido
        unidade = item.get("unidade", "UN")
        if unidade not in ALLOWED_UNITS:
            original = unidade
            item["unidade"] = "UN"
            alerts.append(
                f"Unidade '{original}' não reconhecida — corrigida para 'UN'"
            )

        # Aplica flags
        if alerts:
            item["is_suspect"]    = True
            item["alert_reason"]  = alerts
        else:
            item["is_suspect"]    = False
            item["alert_reason"]  = []

    return items


def _sanity_check_normalized(items: list) -> list:
    """
    Versão do sanity check para itens já normalizados (estrutura fornecedores{preco_unit}).
    """
    for item in items:
        alerts   = []
        quantidade = item.get("quantidade") or 1.0

        fornecedores = item.get("fornecedores") or {}
        for fname, fdata in fornecedores.items():
            if not isinstance(fdata, dict):
                continue
            preco_unit = fdata.get("preco_unit")

            if preco_unit is None:
                continue

            # Preço absurdamente alto
            if preco_unit > _PRICE_HIGH_THRESHOLD:
                alerts.append(
                    f"[{fname}] Preço unitário suspeito: R$ {preco_unit:.2f} "
                    f"(acima de R$ {_PRICE_HIGH_THRESHOLD:.0f})"
                )

            # Preço negativo
            if preco_unit < 0:
                alerts.append(f"[{fname}] Preço negativo: R$ {preco_unit:.2f}")

        # Unidade fora do conjunto permitido
        unidade = item.get("unidade", "UN")
        if unidade not in ALLOWED_UNITS:
            original       = unidade
            item["unidade"] = "UN"
            alerts.append(f"Unidade '{original}' não reconhecida — corrigida para 'UN'")

        if alerts:
            item["is_suspect"]   = True
            item["alert_reason"] = alerts
        else:
            item["is_suspect"]   = False
            item["alert_reason"] = []

    return items


# ── Retry com backoff exponencial ─────────────────────────────────────────────

def _extract_retry_delay(msg: str):
    m = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', msg)
    return float(m.group(1)) + 3 if m else None


def _call_with_retry(
    parts,
    max_attempts: int = 8,
    base_delay: float = 20.0,
    response_schema=None,
) -> str:
    """
    Chama o Gemini com retry exponencial puro.
    Quando response_schema é fornecido, ativa Structured Outputs nativos da API.
    """
    is_text_only = isinstance(parts, str)

    # Camada 1: Structured Outputs via response_schema (force JSON tipado)
    if response_schema and is_text_only:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        )
    elif is_text_only:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
        )
    else:
        generation_config = None

    attempt, delay = 0, base_delay
    while attempt < max_attempts:
        try:
            model = _model()
            if generation_config:
                response = model.generate_content(parts, generation_config=generation_config)
            else:
                response = model.generate_content(parts)
            return response.text

        except ResourceExhausted as e:
            attempt += 1
            wait = _extract_retry_delay(str(e)) or delay
            logger.warning(
                f"[Gemini Flash] 429 na tentativa {attempt}/{max_attempts}. "
                f"Aguardando {wait:.0f}s..."
            )
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Limite de requisições do Gemini atingido após {max_attempts} tentativas.\n\n"
                    f"O free tier do Gemini Flash tem ~10 requisições/minuto. "
                    f"Aguarde 1-2 minutos e tente novamente.\n\n"
                    f"Se o problema persistir, verifique sua cota em: https://ai.dev/rate-limit"
                ) from e
            time.sleep(wait)
            delay = min(delay * 2.0, 120)

        except ServiceUnavailable as e:
            attempt += 1
            logger.warning(
                f"[Gemini] Serviço indisponível, tentativa {attempt}. "
                f"Aguardando {delay:.0f}s..."
            )
            time.sleep(delay)
            delay = min(delay * 2, 120)
            if attempt >= max_attempts:
                raise

        except Exception:
            raise


_INTER_CALL_DELAY = 8.0


# ── Prompts ───────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
Você é um assistente especializado em análise de orçamentos de compras empresariais brasileiros.

Analise o orçamento abaixo e extraia TODOS os itens cotados.
Para cada item, retorne um objeto JSON com exatamente os seguintes campos:

- "item": nome CURTO e SIMPLES do produto (string, em MAIÚSCULAS). Use nomes diretos sem detalhes técnicos desnecessários.
  Exemplos corretos: "CAIXA DE ARQUIVO MORTO", "BORRACHA PEQUENA", "COPO 200 ML PP", "ESTILETE LARGO", "PAPEL A4 C/500FLS", "PILHA ALCALINA AA NORMAL", "BALDE PLASTICO 8L"
  Exemplos ERRADOS: "CAIXA ARQUIVO PAPELAO OFICIO", "BORRACHA BRANCA PEQUENA N.01", "COPO DESCARTAVEL TRANSPARENTE 200ML PP", "ESTILETE LARGO 18MM PROFISSIONAL"
- "marca": UMA ÚNICA marca principal do produto (string ou null). NÃO liste múltiplas marcas.
- "quantidade": quantidade numérica cotada pelo fornecedor (float — OBRIGATÓRIO número, nunca string)
- "unidade": unidade de medida. APENAS as seguintes são permitidas: "UN", "CX", "PCT", "BB", "KG"
  Regras de conversão:
  - Resma de papel = PCT
  - Fardo (FD) = PCT
  - Pacote com múltiplas unidades (ex: pilha c/4, copo c/100) = PCT
  - Bombona/Balde/Galão = BB
  - Rolo, Metro, Litro avulso = UN
  - Se não se encaixar em CX/PCT/BB/KG, use UN
- "preco_unitario": preço unitário POR EMBALAGEM em reais (float — OBRIGATÓRIO número, nunca string, sem símbolo R$).
  IMPORTANTE: NÃO divida o preço. Se o pacote de 4 pilhas custa R$18,64, o preco_unitario é 18.64 (não 4.66).
  O preço deve ser EXATAMENTE como está no orçamento, por embalagem/unidade de venda.
- "preco_total": preço total do item em reais (float — número, ou null se não informado)
- "observacao": qualquer observação relevante — frete, prazo, validade, restrição (string ou null).
  NÃO adicione observações sobre normalização ou conversão de unidades.

REGRAS CRÍTICAS:
- quantidade, preco_unitario e preco_total DEVEM ser números (float), NUNCA strings
- Use nomes CURTOS e PADRONIZADOS para os produtos (sem detalhes de gramatura, cor genérica, material óbvio)
- Extraia EXATAMENTE os preços do documento — preço por embalagem de venda, sem dividir
- APENAS UMA marca por item (a principal visível no orçamento)
- Unidades permitidas: UN, CX, PCT, BB, KG — converta qualquer outra para uma dessas
- Se um campo não existir no documento, use null — nunca invente valores
- Ignore linhas de cabeçalho, totais e rodapés — apenas itens com preço
- Retorne APENAS um array JSON válido e completo, sem markdown, sem texto extra

{preferences}

ORÇAMENTO:
{texto}
"""

EXTRACTION_PROMPT_VISION = """
Você é um assistente especializado em análise de orçamentos de compras empresariais brasileiros.

Analise com atenção as imagens deste orçamento e extraia TODOS os itens cotados.
Para cada item, retorne um objeto JSON com exatamente os seguintes campos:

- "item": nome CURTO e SIMPLES do produto (string, em MAIÚSCULAS). Use nomes diretos sem detalhes técnicos desnecessários.
  Exemplos corretos: "CAIXA DE ARQUIVO MORTO", "BORRACHA PEQUENA", "COPO 200 ML PP", "ESTILETE LARGO", "PAPEL A4 C/500FLS", "PILHA ALCALINA AA NORMAL", "BALDE PLASTICO 8L"
  Exemplos ERRADOS: "CAIXA ARQUIVO PAPELAO OFICIO", "BORRACHA BRANCA PEQUENA N.01", "COPO DESCARTAVEL TRANSPARENTE 200ML PP"
- "marca": UMA ÚNICA marca principal do produto (string ou null). NÃO liste múltiplas marcas.
- "quantidade": quantidade numérica cotada (float — número, nunca string)
- "unidade": unidade de medida. APENAS permitidas: "UN", "CX", "PCT", "BB", "KG"
  Regras: Resma=PCT, Fardo=PCT, Pacote com múltiplas unidades=PCT, Bombona/Balde/Galão=BB, Outros=UN
- "preco_unitario": preço unitário POR EMBALAGEM em reais (float — número, sem R$).
  NÃO divida o preço. Mantenha exatamente como está no orçamento (preço por unidade de venda).
- "preco_total": preço total do item (float ou null)
- "observacao": observações relevantes como frete, prazo, validade (string ou null).
  NÃO adicione observações sobre normalização de unidades.

REGRAS:
- quantidade, preco_unitario e preco_total DEVEM ser números (float), NUNCA strings
- Nomes CURTOS e SIMPLES — sem detalhes desnecessários (gramatura, cor genérica, material óbvio)
- APENAS UMA marca por item
- Preço EXATO do documento — por embalagem, sem dividir
- Unidades permitidas: UN, CX, PCT, BB, KG
- Use null para campos ausentes — nunca invente
- Ignore cabeçalhos, totais e rodapés
- Retorne APENAS um array JSON válido, sem markdown

{preferences}
"""

NORMALIZATION_PROMPT = """
Você é um assistente especializado em mapas de compras empresariais brasileiros.

Você receberá os itens extraídos de orçamentos de {n_fornecedores} fornecedores diferentes.
Sua tarefa é criar um mapa unificado para comparação entre fornecedores.

DADOS DOS FORNECEDORES:
{dados_fornecedores}

LISTA DE REFERÊNCIA (itens que precisamos comprar, se fornecida):
{lista_referencia}

INSTRUÇÕES:
1. IDENTIFICAÇÃO: Agrupe itens equivalentes entre fornecedores mesmo que o nome varie ligeiramente
   (ex: "HIPOCLORITO 5% 5L" e "CLORO ATIVO 5L" são o mesmo produto)

2. NOME PADRONIZADO: Use nomes CURTOS e SIMPLES para cada item.
   - Bom: "CAIXA DE ARQUIVO MORTO", "BORRACHA PEQUENA", "COPO 200 ML PP", "PAPEL A4 C/500FLS"
   - Ruim: "CAIXA ARQUIVO PAPELÃO OFÍCIO", "BORRACHA BRANCA PEQUENA Nº01", "COPO DESCARTÁVEL 200ML PP TRANSPARENTE"

3. UNIDADES PERMITIDAS: APENAS "UN", "CX", "PCT", "BB", "KG"
   - Resma de papel → PCT
   - Fardo → PCT
   - Pacote com múltiplas unidades (pilha c/4, copo c/100, etc.) → PCT
   - Bombona, Balde, Galão → BB
   - Qualquer outra → UN

4. PREÇO SEM NORMALIZAR: Use o preço POR EMBALAGEM DE VENDA exatamente como no orçamento.
   - Se pilha vem em pacote de 4 a R$18,64 → preco_unit = 18.64 (NÃO divida por 4)
   - Se copo vem pacote 100 un a R$5,37 → preco_unit = 5.37
   - Se papel A4 vem resma 500 folhas a R$23,00 → preco_unit = 23.00
   - preco_unit DEVE ser número (float), NUNCA string

5. MARCA: Apenas UMA marca por item (a mais comum ou relevante entre os fornecedores)

6. QUANTIDADE: Use a da lista de referência se fornecida; senão, a quantidade que a empresa deseja comprar (em unidades de embalagem)
   - quantidade DEVE ser número (float), NUNCA string

7. ITENS AUSENTES: Se um fornecedor não cotou o item, use null em preco_unit

8. OBSERVAÇÕES: NÃO adicione observações sobre normalização, conversão ou unidade padrão. Apenas observações relevantes do orçamento (frete, prazo, validade, etc.)

Retorne um array JSON onde cada elemento tem EXATAMENTE esta estrutura:
{{
  "id": int começando em 1,
  "item": "NOME CURTO PADRONIZADO EM MAIÚSCULAS",
  "marca": "marca única ou null",
  "quantidade": float (número — NUNCA string),
  "unidade": "UN ou CX ou PCT ou BB ou KG",
  "fornecedores": {{
    "fornecedor_1": {{"preco_unit": float_ou_null, "obs": null}},
    "fornecedor_2": {{"preco_unit": float_ou_null, "obs": null}},
    "fornecedor_3": {{"preco_unit": float_ou_null, "obs": null}},
    "fornecedor_4": {{"preco_unit": float_ou_null, "obs": null}}
  }},
  "observacao": null
}}

{preferences}

IMPORTANTE:
- id, quantidade e preco_unit DEVEM ser números, NUNCA strings
- Retorne APENAS o array JSON completo, sem markdown, sem explicações.
- NÃO normalize/divida preços. Mantenha o preço por embalagem de venda.
- Unidades APENAS: UN, CX, PCT, BB, KG
"""


# ── Funções utilitárias ───────────────────────────────────────────────────────

def _normalize_unit(unit: str) -> str:
    """Normaliza unidade para uma das permitidas: UN, CX, PCT, BB, KG."""
    if not unit:
        return "UN"
    unit = unit.strip().upper()
    unit_map = {
        "UN": "UN", "UND": "UN", "UNID": "UN", "UNIDADE": "UN",
        "CX": "CX", "CAIXA": "CX",
        "PCT": "PCT", "PACOTE": "PCT", "PC": "PCT", "PAC": "PCT",
        "FD": "PCT", "FARDO": "PCT",
        "RESMA": "PCT", "RSM": "PCT",
        "BB": "BB", "BOMBONA": "BB", "BALDE": "BB", "BD": "BB",
        "GL": "BB", "GALAO": "BB", "GALÃO": "BB",
        "KG": "KG", "KILO": "KG", "QUILO": "KG",
        "L": "UN", "LT": "UN", "LITRO": "UN",
        "M": "UN", "MT": "UN", "METRO": "UN",
        "M2": "UN", "ROLO": "UN", "RL": "UN",
    }
    return unit_map.get(unit, "UN")


def _validate_extraction_items(raw_items: list) -> list:
    """Valida lista de itens extraídos com Pydantic (Camada 2)."""
    validated = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            obj = ExtractionItem(**raw)
            validated.append(obj.model_dump())
        except Exception as e:
            logger.warning(f"[Pydantic] Item descartado por falha de validação: {e} — {raw}")
    return validated


def _validate_normalized_items(raw_items: list) -> list:
    """Valida lista de itens normalizados com Pydantic (Camada 2)."""
    validated = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            obj  = NormalizedItem(**raw)
            data = obj.model_dump()
            # Reconverte FornecedorData de volta para dict simples
            data["fornecedores"] = {
                k: {"preco_unit": v["preco_unit"], "obs": v["obs"]}
                for k, v in data["fornecedores"].items()
            }
            validated.append(data)
        except Exception as e:
            logger.warning(f"[Pydantic] Item normalizado descartado: {e} — {raw}")
    return validated


def _post_process_items(items: list) -> list:
    """Pós-processamento para garantir conformidade com regras do mapa (Camada 2 + 3)."""
    for item in items:
        if "unidade" in item:
            item["unidade"] = _normalize_unit(item.get("unidade", "UN"))
        marca = item.get("marca")
        if marca and "/" in marca:
            item["marca"] = marca.split("/")[0].strip()
        if "item" in item:
            item["item"] = " ".join(item["item"].split()).upper()

    # Camada 3: sanity check
    items = _sanity_check(items)
    return items


def _fuzzy_match_catalog(item_name: str, catalog: list) -> tuple:
    """
    Fuzzy matching contra o catálogo oficial.
    Retorna (melhor_match_nome, melhor_score, unidade_padrao).
    Score entre 0 e 1; >= 0.82 é considerado match seguro.
    """
    if not catalog or not item_name:
        return None, 0.0, None
    best_name  = None
    best_score = 0.0
    best_unit  = None
    item_upper = item_name.upper()
    for entry in catalog:
        nome = (entry.get("nome_oficial") or "").upper()
        if not nome:
            continue
        score = SequenceMatcher(None, item_upper, nome).ratio()
        if score > best_score:
            best_score = score
            best_name  = entry.get("nome_oficial")
            best_unit  = entry.get("unidade_padrao")
    return best_name, best_score, best_unit


def _post_process_normalized(items: list, catalog: list = None) -> list:
    """Pós-processamento para itens normalizados (Camada 2 + 3 + fuzzy catalog)."""
    for item in items:
        if "unidade" in item:
            item["unidade"] = _normalize_unit(item.get("unidade", "UN"))
        marca = item.get("marca")
        if marca and "/" in marca:
            item["marca"] = marca.split("/")[0].strip()
        if "item" in item:
            item["item"] = " ".join(item["item"].split()).upper()
        obs = item.get("observacao")
        if obs and any(kw in obs.lower() for kw in ["normaliz", "unidade padrão", "preços convertidos", "preço por unidade"]):
            item["observacao"] = None
        fornecedores = item.get("fornecedores", {})
        for fname, fdata in fornecedores.items():
            if isinstance(fdata, dict):
                fobs = fdata.get("obs")
                if fobs and any(kw in fobs.lower() for kw in ["normaliz", "unidade padrão", "convertid"]):
                    fdata["obs"] = None

    # Camada 3: sanity check para estrutura normalizada
    items = _sanity_check_normalized(items)

    # Camada 4 (opcional): fuzzy matching contra catálogo oficial
    if catalog:
        _FUZZY_THRESHOLD = 0.82  # score mínimo para considerar match
        for item in items:
            nome_item = item.get("item", "")
            best_name, score, best_unit = _fuzzy_match_catalog(nome_item, catalog)
            if score >= _FUZZY_THRESHOLD:
                # Match encontrado — normaliza nome e unidade conforme catálogo
                item["catalog_match"]      = best_name
                item["catalog_score"]      = round(score, 3)
                # Corrige unidade se o catálogo define uma diferente
                if best_unit and best_unit.upper() in ALLOWED_UNITS:
                    item["unidade"] = best_unit.upper()
            else:
                # Sem match — sinaliza para revisão humana
                item["catalog_match"]  = None
                item["catalog_score"]  = round(score, 3)
                item["is_suspect"]     = True
                reasons = list(item.get("alert_reason") or [])
                reasons.append(
                    f"Item não encontrado no catálogo oficial "
                    f"(melhor correspondência: '{best_name}', score {score:.0%})"
                )
                item["alert_reason"] = reasons

    return items


# ── Funções públicas ──────────────────────────────────────────────────────────

def extract_items_from_text(text: str, preferences_context: str = "") -> list:
    """Extrai itens de PDF com texto selecionável."""
    time.sleep(_INTER_CALL_DELAY)
    raw = _call_with_retry(
        EXTRACTION_PROMPT.format(
            preferences=preferences_context or "Nenhuma preferência registrada ainda.",
            texto=text[:14000],
        ),
        response_schema=_EXTRACTION_RESPONSE_SCHEMA,
    )
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        items = _parse_json_response(raw)

    # Camada 2: validação Pydantic
    items = _validate_extraction_items(items)
    # Camada 3: sanity check + normalização
    return _post_process_items(items)


def extract_items_from_images(images_b64: list, preferences_context: str = "") -> list:
    """Extrai itens de PDF escaneado ou imagem via Gemini Vision."""
    time.sleep(_INTER_CALL_DELAY)
    vision_prompt = EXTRACTION_PROMPT_VISION.format(
        preferences=preferences_context or "Nenhuma preferência registrada ainda."
    )
    parts = [vision_prompt] + [
        {"mime_type": "image/png", "data": b64} for b64 in images_b64
    ]
    raw   = _call_with_retry(parts)
    items = _parse_json_response(raw)

    # Camada 2: validação Pydantic
    items = _validate_extraction_items(items)
    # Camada 3: sanity check + normalização
    return _post_process_items(items)


def extract_items_from_jpeg_images(images_b64: list, preferences_context: str = "") -> list:
    """Extrai itens de imagens JPEG via Gemini Vision."""
    time.sleep(_INTER_CALL_DELAY)
    vision_prompt = EXTRACTION_PROMPT_VISION.format(
        preferences=preferences_context or "Nenhuma preferência registrada ainda."
    )
    parts = [vision_prompt] + [
        {"mime_type": "image/jpeg", "data": b64} for b64 in images_b64
    ]
    raw   = _call_with_retry(parts)
    items = _parse_json_response(raw)

    # Camada 2: validação Pydantic
    items = _validate_extraction_items(items)
    # Camada 3: sanity check + normalização
    return _post_process_items(items)


def normalize_and_match(
    supplier_items: dict,
    reference_list=None,
    preferences_context: str = "",
    catalog: list = None,
) -> list:
    """
    Normaliza e cruza itens entre fornecedores.

    Parâmetros:
      supplier_items      — dict {nome_fornecedor: [itens extraídos]}
      reference_list      — lista de referência de compra (opcional)
      preferences_context — contexto de correções aprendidas (opcional)
      catalog             — catálogo oficial de produtos do Supabase (opcional).
                            Se fornecido, aplica fuzzy matching (Camada 4) para
                            sinalizar itens não reconhecidos.

    NOTA sobre response_schema:
    O passo de normalização é um prompt complexo. Usar response_schema aqui
    causa alucinações (ex: 1 item "BMW" sem preços). response_schema é
    DESATIVADO — apenas response_mime_type="application/json" é usado.
    A validação de tipos fica a cargo do Pydantic (Camada 2).
    """
    time.sleep(_INTER_CALL_DELAY)

    suppliers = list(supplier_items.keys())
    total_input_items = sum(len(v) for v in supplier_items.values())

    dados = "\n\n".join(
        f"FORNECEDOR {i} — {name}:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
        for i, (name, items) in enumerate(supplier_items.items(), 1)
    )
    ref_str = json.dumps(reference_list, ensure_ascii=False) if reference_list else "Não fornecida"

    prompt = NORMALIZATION_PROMPT.format(
        n_fornecedores=len(suppliers),
        dados_fornecedores=dados,
        lista_referencia=ref_str,
        preferences=preferences_context or "Nenhuma preferência registrada ainda.",
    )

    # NÃO passa response_schema — o schema estrito causa alucinações neste passo.
    raw = _call_with_retry(prompt, response_schema=None)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = _parse_json_response(raw)

    # ── Guardrail de plausibilidade ───────────────────────────────────────────
    # Menos de 30% dos itens de entrada → provavelmente alucinação/truncamento.
    if isinstance(data, list) and len(data) < max(1, total_input_items * 0.30):
        logger.warning(
            f"[Normalização] Resultado suspeito: {len(data)} itens retornados "
            f"vs {total_input_items} de entrada. Retentando..."
        )
        raw2 = _call_with_retry(prompt, response_schema=None)
        try:
            data2 = json.loads(raw2)
        except (json.JSONDecodeError, TypeError):
            data2 = _parse_json_response(raw2)
        if isinstance(data2, list) and len(data2) > len(data):
            data = data2

    # Remap fornecedor_1..N → nomes reais
    key_map = {f"fornecedor_{i+1}": name for i, name in enumerate(suppliers)}
    for item in data:
        if "fornecedores" in item:
            item["fornecedores"] = {
                key_map.get(k, k): v for k, v in item["fornecedores"].items()
            }

    # Camada 2: validação Pydantic
    data = _validate_normalized_items(data)
    # Camada 3+4: sanity check + fuzzy catalog matching
    data = _post_process_normalized(data, catalog=catalog)

    return data


# ── Utilitário de parse ───────────────────────────────────────────────────────

def _parse_json_response(text: str) -> list:
    """Limpa e parseia resposta JSON do Gemini, tolerante a markdown residual."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []
