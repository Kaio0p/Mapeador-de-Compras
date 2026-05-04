# -*- coding: utf-8 -*-
"""
gemini_processor.py
Extração e normalização via Gemini 2.5 Flash.
Retry com backoff exponencial puro — sem fallback para modelos inferiores.
Free tier: ~10 RPM. Delay conservador entre chamadas para respeitar o limite.
"""
import json, re, time, logging
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

logger = logging.getLogger(__name__)

# Modelo único — 2.5 flash é o mais capaz do free tier atualmente
_MODEL_NAME = "gemini-2.5-flash"

def configure(api_key: str):
    genai.configure(api_key=api_key)

def _model():
    return genai.GenerativeModel(_MODEL_NAME)

def _extract_retry_delay(msg: str) -> float | None:
    """Extrai o retry_delay sugerido pelo Google da mensagem de erro."""
    m = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', msg)
    return float(m.group(1)) + 3 if m else None   # +3s de margem

def _call_with_retry(parts, max_attempts: int = 8, base_delay: float = 20.0) -> str:
    """
    Chama o Gemini com retry exponencial puro.
    429 → espera o tempo sugerido pelo Google (ou backoff próprio) e tenta novamente.
    Não faz fallback de modelo: qualidade primeiro.
    """
    attempt, delay = 0, base_delay
    while attempt < max_attempts:
        try:
            return _model().generate_content(parts).text

        except ResourceExhausted as e:
            attempt += 1
            wait = _extract_retry_delay(str(e)) or delay
            logger.warning(f"[Gemini 2.5 Flash] 429 na tentativa {attempt}/{max_attempts}. Aguardando {wait:.0f}s...")
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Limite de requisições do Gemini atingido após {max_attempts} tentativas.\n\n"
                    f"O free tier do Gemini 2.5 Flash tem ~10 requisições/minuto. "
                    f"Aguarde 1-2 minutos e tente novamente.\n\n"
                    f"Se o problema persistir, verifique sua cota em: https://ai.dev/rate-limit"
                ) from e
            time.sleep(wait)
            delay = min(delay * 2.0, 120)   # dobra a cada tentativa, cap em 2 min

        except ServiceUnavailable as e:
            attempt += 1
            logger.warning(f"[Gemini] Serviço indisponível, tentativa {attempt}. Aguardando {delay:.0f}s...")
            time.sleep(delay); delay = min(delay * 2, 120)
            if attempt >= max_attempts: raise

        except Exception:
            raise   # qualquer outro erro: propaga imediatamente sem retry

# Delay entre chamadas consecutivas — conservador para free tier
# 10 RPM → 6s entre calls é o mínimo teórico; usamos 8s para margem
_INTER_CALL_DELAY = 8.0


# ── Prompts ───────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
Você é um assistente especializado em análise de orçamentos de compras empresariais brasileiros.

Analise o orçamento abaixo e extraia TODOS os itens cotados.
Para cada item, retorne um objeto JSON com exatamente os seguintes campos:

- "item": nome completo do produto (string, em MAIÚSCULAS)
- "marca": marca do produto (string ou null)
- "quantidade": quantidade numérica cotada pelo fornecedor (float)
- "unidade": unidade de medida original (ex: "UN", "CX", "FD", "KG", "L", "BB", "PCT", "M", "M2", "ROLO", "GL")
- "preco_unitario": preço unitário em reais (float, sem símbolo R$)
- "preco_total": preço total do item em reais (float, ou null se não informado)
- "observacao": qualquer observação relevante — frete, prazo, validade, restrição (string ou null)

REGRAS CRÍTICAS:
- Extraia EXATAMENTE os valores do documento, sem converter unidades ou preços
- Inclua a unidade original mesmo que pareça incomum (ex: "FD" de 100un, "BB" de 5L)
- Se o preço estiver por embalagem (caixa, fardo, etc.), registre o preço da embalagem e a unidade da embalagem
- Se um campo não existir no documento, use null — nunca invente valores
- Ignore linhas de cabeçalho, totais e rodapés — apenas itens com preço
- Retorne APENAS um array JSON válido e completo, sem markdown, sem texto extra

ORÇAMENTO:
{texto}
"""

EXTRACTION_PROMPT_VISION = """
Você é um assistente especializado em análise de orçamentos de compras empresariais brasileiros.

Analise com atenção as imagens deste orçamento e extraia TODOS os itens cotados.
Para cada item, retorne um objeto JSON com exatamente os seguintes campos:

- "item": nome completo do produto (string, em MAIÚSCULAS)
- "marca": marca do produto (string ou null)
- "quantidade": quantidade numérica cotada (float)
- "unidade": unidade de medida original (UN, CX, FD, KG, L, BB, PCT, M, M2, ROLO, GL, etc.)
- "preco_unitario": preço unitário em reais (float, sem R$)
- "preco_total": preço total do item (float ou null)
- "observacao": observações relevantes como frete, prazo, validade (string ou null)

REGRAS:
- Extraia EXATAMENTE os valores visíveis no documento
- Use null para campos ausentes — nunca invente
- Ignore cabeçalhos, totais e rodapés
- Retorne APENAS um array JSON válido, sem markdown
"""

NORMALIZATION_PROMPT = """
Você é um assistente especializado em mapas de compras empresariais brasileiros.

Você receberá os itens extraídos de orçamentos de {n_fornecedores} fornecedores diferentes.
Sua tarefa é criar um mapa unificado com quantidades e unidades igualadas para comparação justa.

DADOS DOS FORNECEDORES:
{dados_fornecedores}

LISTA DE REFERÊNCIA (itens que precisamos comprar, se fornecida):
{lista_referencia}

INSTRUÇÕES:
1. IDENTIFICAÇÃO: Agrupe itens equivalentes entre fornecedores mesmo que o nome seja diferente
   (ex: "HIPOCLORITO 5% 5L" e "CLORO ATIVO 5L" são o mesmo produto)

2. UNIDADE PADRÃO: Escolha a unidade mais granular como padrão (geralmente UN, L, KG, M)
   Fatores de conversão comuns:
   - FD (fardo) → quantidade de unidades especificada na descrição ou embalagem
   - BB (bombona/balde) → geralmente 1 unidade de embalagem maior
   - CX (caixa) → quantidade especificada na embalagem
   - GL (galão) → 1 unidade
   Se não for possível inferir o fator de conversão, mantenha a unidade original

3. PREÇO NORMALIZADO: Calcule o preço por 1 unidade na unidade padrão escolhida

4. ITENS AUSENTES: Se um fornecedor não cotou o item, use null em preco_unit

5. QUANTIDADE: Use a quantidade da lista de referência se fornecida; senão use a do orçamento

Retorne um array JSON onde cada elemento tem EXATAMENTE esta estrutura:
{{
  "id": int começando em 1,
  "item": "NOME PADRONIZADO EM MAIÚSCULAS",
  "marca": "marca ou null",
  "quantidade": float,
  "unidade": "unidade padrão escolhida",
  "fornecedores": {{
    "fornecedor_1": {{"preco_unit": float_ou_null, "obs": "observação ou null"}},
    "fornecedor_2": {{"preco_unit": float_ou_null, "obs": "observação ou null"}},
    "fornecedor_3": {{"preco_unit": float_ou_null, "obs": "observação ou null"}},
    "fornecedor_4": {{"preco_unit": float_ou_null, "obs": "observação ou null"}}
  }},
  "observacao": "observação geral ou null"
}}

IMPORTANTE: Retorne APENAS o array JSON completo, sem markdown, sem explicações.
"""


# ── Funções públicas ──────────────────────────────────────────────────────────

def extract_items_from_text(text: str) -> list[dict]:
    """Extrai itens de PDF com texto selecionável."""
    time.sleep(_INTER_CALL_DELAY)
    raw = _call_with_retry(EXTRACTION_PROMPT.format(texto=text[:15000]))
    return _parse_json_response(raw)


def extract_items_from_images(images_b64: list[str]) -> list[dict]:
    """Extrai itens de PDF escaneado via Gemini Vision."""
    time.sleep(_INTER_CALL_DELAY)
    parts = [EXTRACTION_PROMPT_VISION] + [
        {"mime_type": "image/png", "data": b64} for b64 in images_b64
    ]
    raw = _call_with_retry(parts)
    return _parse_json_response(raw)


def normalize_and_match(
    supplier_items: dict[str, list[dict]],
    reference_list: list[dict] | None = None
) -> list[dict]:
    """Normaliza e cruza itens entre fornecedores."""
    time.sleep(_INTER_CALL_DELAY)

    suppliers = list(supplier_items.keys())
    dados = "\n\n".join(
        f"FORNECEDOR {i} — {name}:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
        for i, (name, items) in enumerate(supplier_items.items(), 1)
    )
    ref_str = json.dumps(reference_list, ensure_ascii=False) if reference_list else "Não fornecida"

    prompt = NORMALIZATION_PROMPT.format(
        n_fornecedores=len(suppliers),
        dados_fornecedores=dados,
        lista_referencia=ref_str,
    )

    raw  = _call_with_retry(prompt)
    data = _parse_json_response(raw)

    # Remap fornecedor_1..N → nomes reais
    key_map = {f"fornecedor_{i+1}": name for i, name in enumerate(suppliers)}
    for item in data:
        if "fornecedores" in item:
            item["fornecedores"] = {
                key_map.get(k, k): v for k, v in item["fornecedores"].items()
            }

    return data


# ── Utilitário de parse ───────────────────────────────────────────────────────

def _parse_json_response(text: str) -> list:
    """Limpa e parseia resposta JSON do Gemini, tolerante a markdown residual."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tenta extrair array do meio do texto
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []
