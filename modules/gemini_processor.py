# -*- coding: utf-8 -*-
"""
gemini_processor.py -- Motor de Visao (OCR) e Auditoria via Gemini
===================================================================
Este modulo tem duas responsabilidades:

  1. OCR / Extracao visual (APENAS para PDFs escaneados e imagens):
       extract_items_from_images()      -- PNG base64
       extract_items_from_jpeg_images() -- JPEG base64

  2. Auditoria Final Anti-Alucinacao:
       audit_purchase_map(normalized_items, original_texts)
       O Gemini usa sua enorme janela de contexto para cruzar o JSON
       normalizado pelo Groq com os textos/imagens ORIGINAIS dos orcamentos,
       detectando alucinacoes logicas, matematicas e de unidade que o
       sanity check local nao consegue pegar semanticamente.

Fora do escopo (movido para groq_processor.py):
  - extract_items_from_text()  -> Groq LPU (velocidade + 128k context)
  - normalize_and_match()      -> Groq LPU (velocidade + 128k context)

Anti-Rate-Limit:
  - Pool de chaves: antes de CADA tentativa, rotaciona via get_random_gemini_key()
  - Retry exponencial com deteccao do campo retry_delay na resposta 429
  - Backoff com cap de 120s
"""
import json
import re
import time
import logging
from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from pydantic import BaseModel, field_validator

from modules.llm_manager import get_random_gemini_key

logger = logging.getLogger(__name__)

# Alias para o modelo mais recente
_MODEL_NAME = "gemini-flash-latest"

ALLOWED_UNITS = ["UN", "CX", "PCT", "BB", "KG"]

_PRICE_HIGH_THRESHOLD = 1_000.0


# ── Retrocompatibilidade ──────────────────────────────────────────────────────

def configure(api_key: str) -> None:
    """Retrocompatibilidade: configura o Gemini com uma chave especifica."""
    genai.configure(api_key=api_key)


def _model():
    return genai.GenerativeModel(_MODEL_NAME)


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


# ── Utilitarios ───────────────────────────────────────────────────────────────

def _normalize_unit(unit: str) -> str:
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


def _validate_extraction_items(raw_items: list) -> list:
    validated = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            obj = ExtractionItem(**raw)
            validated.append(obj.model_dump())
        except Exception as e:
            logger.warning("[Gemini/Pydantic] Item descartado: %s", e)
    return validated


def _sanity_check(items: list) -> list:
    for item in items:
        alerts  = []
        p_unit  = item.get("preco_unitario")
        p_total = item.get("preco_total")
        qtd     = item.get("quantidade") or 1.0

        if p_unit is not None and p_unit > _PRICE_HIGH_THRESHOLD:
            alerts.append(
                "Preco unitario suspeito: R$ {:.2f} (acima de R$ {:.0f})".format(
                    p_unit, _PRICE_HIGH_THRESHOLD
                )
            )
        if p_unit is not None and p_total is not None and p_unit > p_total and qtd >= 1.0:
            alerts.append(
                "Preco unitario R$ {:.2f} maior que total R$ {:.2f}".format(p_unit, p_total)
            )
        if p_unit is not None and p_total is not None and p_total > 0:
            expected = p_unit * qtd
            desvio   = abs(expected - p_total) / p_total
            if desvio > 0.20:
                alerts.append(
                    "Total inconsistente: {}xR${:.2f}=R${:.2f} != total R${:.2f} (desvio {:.0f}%)".format(
                        qtd, p_unit, expected, p_total, desvio * 100
                    )
                )
        unidade = item.get("unidade", "UN")
        if unidade not in ALLOWED_UNITS:
            item["unidade"] = "UN"
            alerts.append("Unidade '{}' nao reconhecida -> corrigida para 'UN'".format(unidade))

        item["is_suspect"]   = bool(alerts)
        item["alert_reason"] = alerts
    return items


def _post_process_items(items: list) -> list:
    for item in items:
        if "unidade" in item:
            item["unidade"] = _normalize_unit(item.get("unidade", "UN"))
        marca = item.get("marca")
        if marca and "/" in marca:
            item["marca"] = marca.split("/")[0].strip()
        if "item" in item:
            item["item"] = " ".join(item["item"].split()).upper()
    return _sanity_check(items)


def _extract_retry_delay(msg: str) -> Optional[float]:
    m = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', msg)
    return float(m.group(1)) + 3 if m else None


def _parse_json_response(text: str) -> list:
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
    return []


def _parse_json_object(text: str) -> dict:
    """Parse que retorna um dict (para respostas de auditoria)."""
    if not text:
        return {}
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


# ── Retry com rotacao de chave a cada tentativa ───────────────────────────────

def _call_vision_with_retry(
    parts: list,
    max_attempts: int = 8,
    base_delay: float = 20.0,
) -> str:
    """
    Chama o Gemini Vision com retry exponencial e rotacao de chave a cada tentativa.
    Antes de CADA tentativa: rotaciona a chave via get_random_gemini_key().
    """
    attempt = 0
    delay   = base_delay

    while attempt < max_attempts:
        key = get_random_gemini_key()
        if key:
            genai.configure(api_key=key)
        else:
            logger.warning("[Gemini Vision] Nenhuma chave Gemini disponivel no pool.")

        try:
            model    = _model()
            response = model.generate_content(parts)
            return response.text

        except ResourceExhausted as e:
            attempt += 1
            wait = _extract_retry_delay(str(e)) or delay
            logger.warning(
                "[Gemini Vision] 429 ResourceExhausted (tentativa %d/%d). "
                "Aguardando %.0fs... (chave rotacionada na proxima tentativa)",
                attempt, max_attempts, wait,
            )
            if attempt >= max_attempts:
                raise RuntimeError(
                    "Limite de requisicoes do Gemini atingido apos {} tentativas.\n\n"
                    "Dicas: adicione mais chaves em GEMINI_API_KEYS nos secrets, "
                    "ou aguarde 1-2 minutos antes de tentar novamente.".format(max_attempts)
                ) from e
            time.sleep(wait)
            delay = min(delay * 2.0, 120.0)

        except ServiceUnavailable as e:
            attempt += 1
            logger.warning(
                "[Gemini Vision] Servico indisponivel (tentativa %d/%d). Aguardando %.0fs...",
                attempt, max_attempts, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, 120.0)
            if attempt >= max_attempts:
                raise

        except Exception:
            raise


# ── Prompts ───────────────────────────────────────────────────────────────────

_VISION_PROMPT = (
    "Voce e um assistente especializado em analise de orcamentos de compras empresariais brasileiros.\n\n"
    "Analise com atencao as imagens deste orcamento e extraia TODOS os itens cotados.\n"
    "Retorne APENAS um array JSON valido, sem markdown, sem texto extra.\n\n"
    "Cada objeto no array deve ter EXATAMENTE estes campos:\n"
    "- \"item\": nome CURTO e SIMPLES em MAIUSCULAS\n"
    "  Exemplos: \"COPO 200ML PP\", \"PAPEL A4 C/500FLS\", \"HIPOCLORITO 5% 5L\", \"PILHA ALCALINA AA\"\n"
    "- \"marca\": string ou null -- UMA UNICA marca principal\n"
    "- \"quantidade\": numero (float) -- NUNCA string\n"
    "- \"unidade\": APENAS \"UN\", \"CX\", \"PCT\", \"BB\" ou \"KG\"\n"
    "  Resma -> PCT | Fardo -> PCT | Pacote c/N -> PCT | Bombona/Balde/Galao -> BB | Outros -> UN\n"
    "- \"preco_unitario\": numero (float) -- preco POR EMBALAGEM exatamente como no documento\n"
    "  NAO divida: pilha c/4 = R$18,64 -> 18.64 (nao 4.66)\n"
    "- \"preco_total\": numero ou null\n"
    "- \"observacao\": string ou null -- apenas observacoes do orcamento (frete, prazo, validade)\n\n"
    "REGRAS CRITICAS:\n"
    "- Todos os numeros como float, NUNCA strings\n"
    "- Preco por embalagem de venda -- nao normalize, nao divida\n"
    "- Use null para campos ausentes -- NUNCA invente\n"
    "- Ignore cabecalhos, totais e rodapes\n\n"
    "{preferences}"
)

_AUDIT_SYSTEM_PROMPT = (
    "Voce e um Auditor de Compras senior com acesso a uma enorme janela de contexto.\n\n"
    "Sua missao: cruzar o mapa de compras normalizado (produzido pelo Groq) com os textos "
    "ORIGINAIS dos orcamentos. Voce deve identificar QUALQUER discrepancia entre o que o "
    "Groq extraiu/normalizou e o que realmente esta nos documentos originais.\n\n"
    "Tipos de anomalias a detectar:\n"
    "1. Alucinacao de preco: preco no mapa difere do documento original\n"
    "2. Alucinacao de unidade: unidade no mapa nao corresponde ao documento\n"
    "3. Erro de proporcao: 2 pacotes de 500g tratados como 1KG mas preco nao foi ajustado\n"
    "4. Preco absurdo para o contexto: caneta R$50, papel A4 R$500\n"
    "5. Inconsistencia matematica: quantidade x preco_unit != preco_total no documento\n"
    "6. Item inventado: item no mapa que nao existe em nenhum orcamento original\n"
    "7. Preco negativo ou zero onde nao faz sentido\n\n"
    "Retorne SEMPRE um objeto JSON com a chave \"items\" contendo TODOS os itens -- "
    "incluindo os que estao OK (is_suspect: false). "
    "NUNCA altere precos, quantidades ou nomes. APENAS atualize is_suspect e alert_reason."
)

_AUDIT_USER_PROMPT = (
    "Audite o mapa de compras abaixo cruzando com os orcamentos originais.\n\n"
    "Para cada item:\n"
    "- Se OK: \"is_suspect\": false, \"alert_reason\": []\n"
    "- Se anomalia: \"is_suspect\": true, \"alert_reason\": [lista de strings descrevendo o problema]\n\n"
    "NAO remova itens. NAO altere precos ou quantidades. APENAS is_suspect e alert_reason.\n\n"
    "=== ORCAMENTOS ORIGINAIS ===\n"
    "{original_texts}\n\n"
    "=== MAPA NORMALIZADO PELO GROQ (a auditar) ===\n"
    "{normalized_json}\n\n"
    "Retorne objeto JSON com chave \"items\" contendo TODOS os {n_items} itens auditados."
)


# ── Funcoes Publicas ──────────────────────────────────────────────────────────

def extract_items_from_images(
    images_b64: list,
    preferences_context: str = "",
    text_fallback: str = "",
) -> list:
    """
    Extrai itens de PDF escaneado, imagens PNG, ou PDF nativo via Gemini.

    Parâmetros:
      images_b64          — lista de strings base64 (PNG) das páginas do PDF.
                            Pode ser lista vazia se text_fallback for fornecido.
      preferences_context — contexto de correções aprendidas.
      text_fallback       — texto extraído de um PDF nativo (quando não há imagens).
                            Se fornecido, o Gemini lê o texto diretamente sem visão.

    A chave Gemini é rotacionada automaticamente a cada tentativa via pool.
    """
    vision_prompt = _VISION_PROMPT.format(
        preferences=preferences_context or "Nenhuma preferência registrada ainda."
    )

    if text_fallback and not images_b64:
        # PDF nativo — envia texto diretamente (sem imagem)
        # O Gemini usa sua janela de contexto longa para ler o texto completo
        text_prompt = (
            vision_prompt
            + "\n\nORÇAMENTO (texto extraído do PDF):\n"
            + text_fallback[:60000]  # limite conservador para tokens
        )
        parts = [text_prompt]
    else:
        parts = [vision_prompt] + [
            {"mime_type": "image/png", "data": b64} for b64 in images_b64
        ]

    raw   = _call_vision_with_retry(parts)
    items = _parse_json_response(raw)
    items = _validate_extraction_items(items)
    return _post_process_items(items)


def extract_items_from_jpeg_images(images_b64: list, preferences_context: str = "") -> list:
    """
    Extrai itens de imagens JPEG via Gemini Vision.
    A chave Gemini e rotacionada automaticamente a cada tentativa via pool.
    """
    vision_prompt = _VISION_PROMPT.format(
        preferences=preferences_context or "Nenhuma preferencia registrada ainda."
    )
    parts = [vision_prompt] + [
        {"mime_type": "image/jpeg", "data": b64} for b64 in images_b64
    ]
    raw   = _call_vision_with_retry(parts)
    items = _parse_json_response(raw)
    items = _validate_extraction_items(items)
    return _post_process_items(items)


def audit_purchase_map(normalized_items: list, original_texts: dict = None) -> list:
    """
    Agente Auditor Final -- usa a enorme janela de contexto do Gemini para
    cruzar o JSON normalizado pelo Groq com os textos ORIGINAIS dos orcamentos.

    Esta abordagem detecta anomalias semanticas e logicas que o sanity check
    local nao consegue identificar, como:
      - Precos que o Groq extraiu errado em relacao ao documento original
      - Erros de proporcao onde o preco nao foi ajustado corretamente
      - Itens inventados que nao existem em nenhum orcamento original
      - Inconsistencias matematicas (qtd x preco != total no documento)

    Parametros:
      normalized_items -- lista retornada por groq_processor.normalize_and_match()
      original_texts   -- dict {nome_fornecedor: texto_do_pdf} para cross-reference.
                          Pode conter PDFs nativos (texto) ou descricao de OCR.
                          Se None, o Gemini audita apenas pelo contexto semantico.

    Retorna a mesma lista com is_suspect e alert_reason atualizados.
    Falha silenciosa -- se o Gemini falhar, retorna os itens sem modificacao.
    """
    if not normalized_items:
        return normalized_items

    # Prepara versao simplificada para o prompt (sem catalog_match/score)
    items_for_audit = []
    for item in normalized_items:
        simplified = {
            "id":         item.get("id"),
            "item":       item.get("item"),
            "quantidade": item.get("quantidade"),
            "unidade":    item.get("unidade"),
            "fornecedores": {
                fname: {"preco_unit": fdata.get("preco_unit")}
                for fname, fdata in (item.get("fornecedores") or {}).items()
            },
        }
        items_for_audit.append(simplified)

    # Prepara texto dos orcamentos originais
    if original_texts and isinstance(original_texts, dict):
        textos_originais = "\n\n".join(
            "--- ORCAMENTO: {} ---\n{}".format(nome, texto[:8000])
            for nome, texto in original_texts.items()
            if texto and isinstance(texto, str)
        )
    else:
        textos_originais = (
            "Textos originais nao disponibilizados. "
            "Audite com base no conhecimento semantico de precos e unidades tipicos "
            "para itens de limpeza e escritorio no Brasil."
        )

    try:
        prompt_text = (
            _AUDIT_SYSTEM_PROMPT
            + "\n\n"
            + _AUDIT_USER_PROMPT.format(
                original_texts=textos_originais,
                normalized_json=json.dumps(
                    {"items": items_for_audit}, ensure_ascii=False, indent=2
                ),
                n_items=len(items_for_audit),
            )
        )

        # Chama Gemini (texto puro -- sem imagem aqui)
        raw = _call_vision_with_retry([prompt_text])

        # Parse -- espera objeto com chave "items"
        parsed = _parse_json_object(raw)
        if not parsed:
            # Tenta como array direto
            arr = _parse_json_response(raw)
            parsed = {"items": arr} if arr else {}

        audited = parsed.get("items", [])
        if not isinstance(audited, list):
            logger.warning("[Gemini/Auditoria] Resposta nao contem lista de items valida.")
            return normalized_items

        # Merge: aplica APENAS is_suspect e alert_reason do Gemini
        # NUNCA deixa o auditor sobrescrever precos ou quantidades
        audited_map = {str(a.get("id", "")): a for a in audited if isinstance(a, dict)}

        for item in normalized_items:
            item_id      = str(item.get("id", ""))
            audited_item = audited_map.get(item_id)
            if not audited_item:
                continue

            audit_suspect = bool(audited_item.get("is_suspect", False))
            audit_reasons = audited_item.get("alert_reason") or []
            if isinstance(audit_reasons, str):
                audit_reasons = [audit_reasons]

            existing_reasons = list(item.get("alert_reason") or [])
            for reason in audit_reasons:
                if reason and reason not in existing_reasons:
                    existing_reasons.append("[Auditor Gemini] {}".format(reason))

            if audit_suspect or existing_reasons:
                item["is_suspect"]   = True
                item["alert_reason"] = existing_reasons

        n_suspect = sum(1 for i in normalized_items if i.get("is_suspect"))
        logger.info(
            "[Gemini/Auditoria] Concluida. Suspeitos: %d/%d",
            n_suspect, len(normalized_items),
        )
        return normalized_items

    except Exception as e:
        logger.error("[Gemini/Auditoria] Erro (nao bloqueante): %s", e)
        return normalized_items

