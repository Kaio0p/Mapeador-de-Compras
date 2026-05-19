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
       normalizado pelo Cohere com os textos/imagens ORIGINAIS dos orcamentos,
       detectando alucinacoes logicas, matematicas e de unidade.

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

# Alias para o modelo mais recente (puxa automaticamente a última versão do Flash)
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
        "BB": "BB", "BOMBONA": "BB", "BD": "BB",
        "GL": "BB", "GALAO": "BB", "GALÃO": "BB",
        "KG": "KG", "KILO": "KG", "QUILO": "KG",
        "L": "UN", "LT": "UN", "LITRO": "UN",
        "M": "UN", "MT": "UN", "METRO": "UN",
        "M2": "UN", "ROLO": "UN", "RL": "UN",
    }
    # NOTA: "BALDE" NÃO é mapeado aqui propositalmente.
    # Um balde pode ser o PRODUTO (balde plástico → UN) ou a EMBALAGEM de um líquido (→ BB).
    # A decisão é feita pelo contexto no prompt da IA, não aqui.
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
    "Você é um assistente especializado em análise de orçamentos de compras empresariais brasileiros.\n\n"
    "Analise com EXTREMA atenção TODAS as partes da imagem/documento e extraia TODOS os itens cotados, sem exceção.\n"
    "Retorne APENAS um array JSON válido, sem markdown, sem texto extra.\n\n"
    "Cada objeto no array deve ter EXATAMENTE estes campos:\n"
    "- \"item\": nome DESCRITIVO em MAIÚSCULAS incluindo características importantes\n"
    "  Exemplos: \"CAIXA DE ARQUIVO MORTO\", \"BORRACHA PEQUENA\", \"COPO 200ML PP\",\n"
    "            \"PILHA ALCALINA AA C/4\", \"BALDE PLASTICO 8L\", \"PAPEL A4\"\n"
    "  INCLUA: tipo, tamanho, capacidade, material quando relevante\n"
    "  NÃO inclua: código interno, referência do fornecedor\n\n"
    "- \"marca\": string ou null — a MARCA do produto (ex: DURACELL, CHAMEX, BRW, CIS, ECOCOPPO, FBOX)\n"
    "  IMPORTANTE: Extraia a marca SEMPRE que estiver presente no documento.\n"
    "  A marca geralmente aparece ao lado do nome do produto ou em coluna própria.\n"
    "  Use EXATAMENTE a marca que aparece no documento — não invente nem substitua.\n\n"
    "- \"quantidade\": número (float) — quantidade de embalagens cotadas\n"
    "  Se o orçamento mostra \"3 PCT\" de pilhas, quantidade = 3\n"
    "  Se não especificada, use 1\n\n"
    "- \"unidade\": APENAS \"UN\", \"CX\", \"PCT\", \"BB\" ou \"KG\"\n"
    "  REGRAS DE UNIDADE:\n"
    "  - Resma de papel → PCT (cada resma é 1 PCT)\n"
    "  - Fardo → PCT\n"
    "  - Pacote c/N unidades (pilha c/4, etc.) → PCT\n"
    "  - COPO DESCARTÁVEL: se a quantidade é em PEÇAS (ex: 100 copos) → UN; se em pacote/manga → PCT\n"
    "  - Bombona/Galão de LÍQUIDO → BB (ex: hipoclorito 5L, detergente 5L)\n"
    "  - BALDE como PRODUTO (balde plástico, balde de limpeza) → UN (é um produto individual)\n"
    "  - BALDE/BOMBONA como EMBALAGEM de líquido → BB\n"
    "  - Item individual avulso → UN\n\n"
    "- \"preco_unitario\": número (float) — preço POR EMBALAGEM DE VENDA exatamente como no documento\n"
    "  REGRA CRÍTICA DE PREÇOS — LEIA COM ATENÇÃO:\n"
    "  - Pilha c/4 = R$18,64 → preco_unitario = 18.64 (preço do PACOTE, NÃO divida por 4!)\n"
    "  - Copo c/100 = R$5,37 → preco_unitario = 5.37 (preço do pacote de 100)\n"
    "  - PAPEL A4: se o orçamento mostra 50 resmas por R$24,90 cada → preco_unitario = 24.90 (por resma)\n"
    "  - NUNCA divida o preço pelo número de itens dentro da embalagem\n"
    "  - O preço é SEMPRE por embalagem de venda como consta no orçamento\n"
    "  - Se não houver preço visível para um item, use null — NUNCA invente valor\n\n"
    "- \"preco_total\": número ou null — valor total da linha (qtd x preco_unitario)\n\n"
    "- \"observacao\": string ou null — apenas observações do orçamento (frete, prazo, validade)\n\n"
    "REGRAS CRÍTICAS OBRIGATÓRIAS:\n"
    "- Extraia ABSOLUTAMENTE TODOS os itens do orçamento — nenhum pode ser pulado\n"
    "- TODOS os preços visíveis DEVEM ser extraídos — se o item tem preço no documento, preco_unitario NUNCA pode ser null\n"
    "- Todos os números como float, NUNCA strings\n"
    "- Preço por embalagem de venda — NÃO normalize, NÃO divida\n"
    "- Use null APENAS para campos genuinamente ausentes no documento — NUNCA invente valores\n"
    "- Ignore cabeçalhos, totais e rodapés\n"
    "- MARCAS: extraia sempre que visíveis no documento — use EXATAMENTE como escrito\n"
    "- Para PILHAS: manter como pacote (C/4 ou C/2), unidade=PCT, preço do pacote inteiro\n"
    "  Se o orçamento tem pilhas C/2 E pilhas C/4 do mesmo tipo, extraia SEPARADAMENTE com seus preços\n"
    "- Para BALDES (produto): unidade=UN, NÃO use BB para baldes que são o produto em si\n"
    "- NÃO crie itens que não existem no documento — nenhuma alucinação é tolerada\n\n"
    "{preferences}"
)

_AUDIT_SYSTEM_PROMPT = (
    "Você é um Auditor de Compras sênior com acesso a uma enorme janela de contexto.\n\n"
    "Sua missão: cruzar o mapa de compras normalizado com os textos/imagens "
    "ORIGINAIS dos orçamentos e identificar discrepâncias REAIS.\n\n"
    "════════════════════════════════════════\n"
    "O QUE É CORRETO E NÃO DEVE SER SINALIZADO:\n"
    "════════════════════════════════════════\n"
    "✓ Normalização proporcional de embalagens (C/2→C/4 multiplicando o preço por 2) "
    "— isso é CORRETO e ESPERADO. NÃO sinalize como erro.\n"
    "✓ Quantidade baseada na necessidade real da empresa (não na soma dos fornecedores) "
    "— CORRETO.\n"
    "✓ Preços normalizados com observação registrando o ajuste — CORRETO.\n\n"
    "════════════════════════════════════════\n"
    "O QUE DEVE SER SINALIZADO (anomalias reais):\n"
    "════════════════════════════════════════\n"
    "1. Preço no mapa DIFERENTE do que está no documento original "
    "(sem normalização proporcional justificada)\n"
    "2. Fornecedor que cotou o item mas aparece como null no mapa\n"
    "3. Preço absurdo para o contexto (ex: caneta R$50, papel A4 R$500)\n"
    "4. Item no mapa que NÃO existe em nenhum orçamento original\n"
    "5. Preço negativo ou zero onde não faz sentido\n"
    "6. Papel A4 com preço multiplicado incorretamente: se JAE cotou 50 resmas a R$X "
    "por resma, o preco_unit deve ser R$X (não R$2X ou R$50X)\n"
    "7. Quantidade no mapa claramente incompatível com todos os orçamentos e a lista "
    "de referência (ex: todos os fornecedores indicam 3 unidades mas o mapa tem 8)\n\n"
    "IMPORTANTE — DEDUPLICAÇÃO:\n"
    "Cada anomalia deve aparecer UMA única vez no alert_reason. "
    "NÃO repita a mesma informação com palavras diferentes.\n\n"
    "Retorne SEMPRE um objeto JSON com a chave \"items\" contendo TODOS os itens. "
    "NUNCA altere preços, quantidades ou nomes. APENAS atualize is_suspect e alert_reason."
)

_AUDIT_USER_PROMPT = (
    "Audite o mapa de compras abaixo cruzando com os orçamentos originais.\n\n"
    "Para cada item:\n"
    "- Se OK: \"is_suspect\": false, \"alert_reason\": []\n"
    "- Se anomalia REAL: \"is_suspect\": true, \"alert_reason\": "
    "[lista de strings — UMA por anomalia, sem repetições]\n\n"
    "NÃO remova itens. NÃO altere preços ou quantidades. APENAS is_suspect e alert_reason.\n\n"
    "LEMBRETE — NÃO sinalize como erro:\n"
    "• Normalização C/2→C/4 com multiplicação do preço (ex: R$11,11 × 2 = R$22,22) → VÁLIDO\n"
    "• Quantidade da necessidade da empresa (não soma de fornecedores) → VÁLIDO\n"
    "• Preço já normalizado e observação explicando o ajuste → VÁLIDO\n\n"
    "SINALIZE como erro apenas:\n"
    "• Fornecedor que cotou o item mas aparece null no mapa\n"
    "• Preço diferente do original sem justificativa de normalização\n"
    "• Item inventado (não existe em nenhum orçamento)\n"
    "• Quantidade claramente errada (discorda de todos os fornecedores E da lista de referência)\n"
    "• Papel A4 com preço multiplicado incorretamente pelo tamanho do lote\n\n"
    "=== ORÇAMENTOS ORIGINAIS ===\n"
    "{original_texts}\n\n"
    "=== MAPA NORMALIZADO (a auditar) ===\n"
    "{normalized_json}\n\n"
    "Retorne objeto JSON com chave \"items\" contendo TODOS os {n_items} itens auditados."
)


# ── Funcoes Publicas ──────────────────────────────────────────────────────────

def extract_items_from_images(
    images_b64: list,
    preferences_context: str = "",
    text_fallback: str = "",
    catalog: list = None,
) -> list:
    """
    Extrai itens de PDF escaneado, imagens PNG, ou PDF nativo via Gemini.

    Parâmetros:
      images_b64          — lista de strings base64 (PNG) das páginas do PDF.
                            Pode ser lista vazia se text_fallback for fornecido.
      preferences_context — contexto de correções aprendidas.
      text_fallback       — texto extraído de um PDF nativo (quando não há imagens).
                            Se fornecido, o Gemini lê o texto diretamente sem visão.
      catalog             — catálogo de produtos do Supabase para contextualizar nomes.

    A chave Gemini é rotacionada automaticamente a cada tentativa via pool.
    """
    # Monta contexto do catálogo para injetar no prompt
    catalog_context = _build_catalog_context(catalog)

    vision_prompt = _VISION_PROMPT.format(
        preferences=(preferences_context or "Nenhuma preferência registrada ainda.") + catalog_context
    )

    if text_fallback and not images_b64:
        # PDF nativo — envia texto diretamente (sem imagem)
        text_prompt = (
            vision_prompt
            + "\n\nORÇAMENTO (texto extraído do PDF):\n"
            + text_fallback[:60000]
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


def extract_items_from_jpeg_images(
    images_b64: list,
    preferences_context: str = "",
    catalog: list = None,
) -> list:
    """
    Extrai itens de imagens JPEG via Gemini Vision.
    A chave Gemini e rotacionada automaticamente a cada tentativa via pool.
    """
    catalog_context = _build_catalog_context(catalog)

    vision_prompt = _VISION_PROMPT.format(
        preferences=(preferences_context or "Nenhuma preferencia registrada ainda.") + catalog_context
    )
    parts = [vision_prompt] + [
        {"mime_type": "image/jpeg", "data": b64} for b64 in images_b64
    ]
    raw   = _call_vision_with_retry(parts)
    items = _parse_json_response(raw)
    items = _validate_extraction_items(items)
    return _post_process_items(items)


def _build_catalog_context(catalog: list) -> str:
    """
    Monta o bloco de contexto do catálogo Supabase para injeção no prompt.
    Inclui nome_oficial, unidade_padrao e marca_referencia (se disponível).
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
        "\n\nCATÁLOGO DE REFERÊNCIA (nomes, unidades e marcas OFICIAIS — use como guia PRIORITÁRIO):\n"
        + "\n".join(entries)
        + "\n\nSe um item do orçamento corresponder a um item do catálogo, "
        "use EXATAMENTE o nome, a unidade e a marca do catálogo.\n"
    )


def audit_purchase_map(
    normalized_items: list,
    original_texts: dict = None,
    original_images: dict = None,
) -> list:
    """
    Agente Auditor Final -- usa a enorme janela de contexto do Gemini para
    cruzar o JSON normalizado com os textos/imagens ORIGINAIS dos orcamentos.

    Esta abordagem detecta anomalias semanticas e logicas que o sanity check
    local nao consegue identificar, como:
      - Precos extraidos errado em relacao ao documento original
      - Erros de proporcao onde o preco nao foi ajustado corretamente
      - Itens inventados que nao existem em nenhum orcamento original
      - Inconsistencias matematicas (qtd x preco != total no documento)
      - Dados de fornecedor faltando (PDF tem mas o mapa nao)

    Parametros:
      normalized_items -- lista retornada por cohere_processor.normalize_and_match()
      original_texts   -- dict {nome_fornecedor: texto_do_pdf} para cross-reference.
      original_images  -- dict {nome_fornecedor: [lista_b64_images]} para auditoria visual.

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
            "marca":      item.get("marca"),
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

        # Monta parts — se temos imagens originais, inclui para auditoria visual
        parts = [prompt_text]
        if original_images and isinstance(original_images, dict):
            for supplier_name, img_list in original_images.items():
                if isinstance(img_list, list):
                    for img_b64 in img_list[:3]:  # máx 3 páginas por fornecedor
                        parts.append({"mime_type": "image/png", "data": img_b64})

        raw = _call_vision_with_retry(parts)

        # Parse -- espera objeto com chave "items"
        parsed = _parse_json_object(raw)
        if not parsed:
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
