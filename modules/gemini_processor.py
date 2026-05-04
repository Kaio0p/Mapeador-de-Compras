"""
gemini_processor.py
Toda lógica de IA: extração de itens por fornecedor e normalização cross-supplier.
Usa gemini-2.0-flash (gratuito no AI Studio).
"""
import json
import re
import google.generativeai as genai


def configure(api_key: str):
    genai.configure(api_key=api_key)


def _model():
    return genai.GenerativeModel("gemini-2.0-flash")


# ─────────────────────────────────────────────
# PROMPT 1: Extração de itens de um orçamento
# ─────────────────────────────────────────────

EXTRACTION_PROMPT = """
Você é um assistente especializado em análise de orçamentos de compras empresariais.

Analise o orçamento abaixo e extraia TODOS os itens cotados.
Para cada item, retorne um objeto JSON com os campos:

- "item": nome do produto (string, em maiúsculas)
- "marca": marca do produto (string ou null)
- "quantidade": quantidade numérica cotada pelo fornecedor (float)
- "unidade": unidade de medida original do fornecedor (ex: "UN", "CX", "FD", "KG", "L", "BB", "PCT", "M", "M2", "ROLO")
- "preco_unitario": preço unitário em reais (float, sem R$)
- "preco_total": preço total do item em reais (float, ou null se não informado)
- "observacao": qualquer observação relevante (frete, prazo, etc.) ou null

REGRAS IMPORTANTES:
- Se o preço for por caixa/fardo/pacote e tiver subdivisão, mantenha a unidade original
- Extraia EXATAMENTE os valores do documento, sem conversão
- Se um campo não existir no documento, use null
- Ignore itens sem preço ou completamente ilegíveis
- Retorne APENAS um array JSON válido, sem markdown, sem texto extra

ORÇAMENTO:
{texto}
"""

EXTRACTION_PROMPT_VISION = """
Você é um assistente especializado em análise de orçamentos de compras empresariais.

Analise as imagens deste orçamento e extraia TODOS os itens cotados.
Para cada item, retorne um objeto JSON com os campos:

- "item": nome do produto (string, em maiúsculas)
- "marca": marca do produto (string ou null)
- "quantidade": quantidade numérica cotada pelo fornecedor (float)
- "unidade": unidade de medida original do fornecedor
- "preco_unitario": preço unitário em reais (float)
- "preco_total": preço total em reais (float ou null)
- "observacao": observações relevantes ou null

Retorne APENAS um array JSON válido, sem markdown, sem texto extra.
"""


def extract_items_from_text(text: str) -> list[dict]:
    """Extrai itens de PDF nativo (texto)."""
    prompt = EXTRACTION_PROMPT.format(texto=text[:15000])  # limite de contexto seguro
    response = _model().generate_content(prompt)
    return _parse_json_response(response.text)


def extract_items_from_images(images_b64: list[str]) -> list[dict]:
    """Extrai itens de PDF escaneado (imagens base64)."""
    parts = [EXTRACTION_PROMPT_VISION]
    for b64 in images_b64:
        parts.append({
            "mime_type": "image/png",
            "data": b64
        })
    response = _model().generate_content(parts)
    return _parse_json_response(response.text)


# ─────────────────────────────────────────────
# PROMPT 2: Normalização e matching cross-supplier
# ─────────────────────────────────────────────

NORMALIZATION_PROMPT = """
Você é um assistente especializado em análise de mapas de compras empresariais.

Você receberá os itens extraídos de orçamentos de {n_fornecedores} fornecedores.
Sua tarefa é criar um mapa unificado igualando quantidades e unidades para comparação justa.

FORNECEDORES E SEUS ITENS:
{dados_fornecedores}

QUANTIDADE DE REFERÊNCIA (se fornecida pelo usuário):
{lista_referencia}

TAREFA:
1. Identifique itens equivalentes entre fornecedores (mesmo produto, nomes diferentes)
2. Escolha uma unidade de medida padrão para cada item (a mais comum ou a do item-mestre)
3. Normalize os preços unitários para essa unidade padrão usando os fatores de conversão corretos:
   - 1 FD (fardo) de saco 100L = geralmente 100 unidades → verifique pelo preço e contexto
   - 1 BB (bombona) = 1 unidade de galão/balde
   - 1 CX = quantidade especificada no nome ou embalagem
   - Use contexto do produto para inferir conversões quando não explícito
4. Para itens sem cotação de algum fornecedor, use null no preço

Retorne um array JSON onde cada elemento é:
{{
  "id": número sequencial (int, começa em 1),
  "item": nome padronizado do item (string, maiúsculas),
  "marca": marca de referência ou null,
  "quantidade": quantidade na unidade padronizada (float),
  "unidade": unidade padronizada escolhida (string),
  "fornecedores": {{
    "fornecedor_1": {{"preco_unit": float ou null, "obs": string ou null}},
    "fornecedor_2": {{"preco_unit": float ou null, "obs": string ou null}},
    "fornecedor_3": {{"preco_unit": float ou null, "obs": string ou null}},
    "fornecedor_4": {{"preco_unit": float ou null, "obs": string ou null}}
  }},
  "observacao": string ou null
}}

IMPORTANTE:
- "fornecedor_1", "fornecedor_2", etc. são chaves fixas (até 4)
- preco_unit deve ser o preço por 1 unidade na unidade padronizada
- Se fornecedor não cotou o item, use null em preco_unit
- Retorne APENAS o array JSON, sem markdown, sem texto extra
"""


def normalize_and_match(
    supplier_items: dict[str, list[dict]],
    reference_list: list[dict] | None = None
) -> list[dict]:
    """
    supplier_items: {"Nome Fornecedor": [lista de itens extraídos], ...}
    reference_list: lista de referência opcional [{item, qtd, unidade}, ...]
    Retorna lista normalizada de itens para o mapa de compras.
    """
    suppliers = list(supplier_items.keys())
    n = len(suppliers)

    dados_str_parts = []
    for i, (name, items) in enumerate(supplier_items.items(), 1):
        dados_str_parts.append(f"FORNECEDOR {i} - {name}:\n{json.dumps(items, ensure_ascii=False, indent=2)}")

    dados_str = "\n\n".join(dados_str_parts)
    ref_str = json.dumps(reference_list, ensure_ascii=False) if reference_list else "Nenhuma fornecida"

    prompt = NORMALIZATION_PROMPT.format(
        n_fornecedores=n,
        dados_fornecedores=dados_str,
        lista_referencia=ref_str
    )

    response = _model().generate_content(prompt)
    raw = _parse_json_response(response.text)

    # Remap chaves genéricas para nomes reais dos fornecedores
    key_map = {f"fornecedor_{i+1}": name for i, name in enumerate(suppliers)}
    for item in raw:
        if "fornecedores" in item:
            remapped = {}
            for generic_key, data in item["fornecedores"].items():
                real_name = key_map.get(generic_key, generic_key)
                remapped[real_name] = data
            item["fornecedores"] = remapped

    return raw


# ─────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────

def _parse_json_response(text: str) -> list:
    """Limpa e parseia resposta JSON do Gemini."""
    # Remove blocos markdown se presentes
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()

    # Tenta parse direto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tenta encontrar array JSON dentro do texto
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return []
