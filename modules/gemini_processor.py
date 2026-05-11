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
_MODEL_NAME = "gemini-flash-latest"

# Unidades permitidas no mapa de compras
ALLOWED_UNITS = ["UN", "CX", "PCT", "BB", "KG"]

def configure(api_key: str):
    genai.configure(api_key=api_key)

def _model():
    return genai.GenerativeModel(_MODEL_NAME)

def _extract_retry_delay(msg: str) -> object:
    """Extrai o retry_delay sugerido pelo Google da mensagem de erro."""
    m = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', msg)
    return float(m.group(1)) + 3 if m else None   # +3s de margem

def _call_with_retry(parts, max_attempts: int = 8, base_delay: float = 20.0) -> str:
    """
    Chama o Gemini com retry exponencial puro.
    429 → espera o tempo sugerido pelo Google (ou backoff próprio) e tenta novamente.
    Não faz fallback de modelo: qualidade primeiro.
    Força JSON nativo via response_mime_type quando o prompt é texto puro.
    """
    # Structured output só funciona para prompts de texto puro (não multimodal)
    is_text_only = isinstance(parts, str)
    generation_config = {"response_mime_type": "application/json"} if is_text_only else None

    attempt, delay = 0, base_delay
    while attempt < max_attempts:
        try:
            model = _model()
            if generation_config:
                response = model.generate_content(parts, generation_config=genai.GenerationConfig(**generation_config))
            else:
                response = model.generate_content(parts)
            return response.text

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

- "item": nome CURTO e SIMPLES do produto (string, em MAIÚSCULAS). Use nomes diretos sem detalhes técnicos desnecessários.
  Exemplos corretos: "CAIXA DE ARQUIVO MORTO", "BORRACHA PEQUENA", "COPO 200 ML PP", "ESTILETE LARGO", "PAPEL A4 C/500FLS", "PILHA ALCALINA AA NORMAL", "BALDE PLASTICO 8L"
  Exemplos ERRADOS: "CAIXA ARQUIVO PAPELAO OFICIO", "BORRACHA BRANCA PEQUENA N.01", "COPO DESCARTAVEL TRANSPARENTE 200ML PP", "ESTILETE LARGO 18MM PROFISSIONAL"
- "marca": UMA ÚNICA marca principal do produto (string ou null). NÃO liste múltiplas marcas.
- "quantidade": quantidade numérica cotada pelo fornecedor (float)
- "unidade": unidade de medida. APENAS as seguintes são permitidas: "UN", "CX", "PCT", "BB", "KG"
  Regras de conversão:
  - Resma de papel = PCT
  - Fardo (FD) = PCT
  - Pacote com múltiplas unidades (ex: pilha c/4, copo c/100) = PCT
  - Bombona/Balde/Galão = BB
  - Rolo, Metro, Litro avulso = UN
  - Se não se encaixar em CX/PCT/BB/KG, use UN
- "preco_unitario": preço unitário POR EMBALAGEM em reais (float, sem símbolo R$). 
  IMPORTANTE: NÃO divida o preço. Se o pacote de 4 pilhas custa R$18,64, o preco_unitario é 18.64 (não 4.66).
  O preço deve ser EXATAMENTE como está no orçamento, por embalagem/unidade de venda.
- "preco_total": preço total do item em reais (float, ou null se não informado)
- "observacao": qualquer observação relevante — frete, prazo, validade, restrição (string ou null). 
  NÃO adicione observações sobre normalização ou conversão de unidades.

REGRAS CRÍTICAS:
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
- "quantidade": quantidade numérica cotada (float)
- "unidade": unidade de medida. APENAS permitidas: "UN", "CX", "PCT", "BB", "KG"
  Regras: Resma=PCT, Fardo=PCT, Pacote com múltiplas unidades=PCT, Bombona/Balde/Galão=BB, Outros=UN
- "preco_unitario": preço unitário POR EMBALAGEM em reais (float, sem R$).
  NÃO divida o preço. Mantenha exatamente como está no orçamento (preço por unidade de venda).
- "preco_total": preço total do item (float ou null)
- "observacao": observações relevantes como frete, prazo, validade (string ou null).
  NÃO adicione observações sobre normalização de unidades.

REGRAS:
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

5. MARCA: Apenas UMA marca por item (a mais comum ou relevante entre os fornecedores)

6. QUANTIDADE: Use a da lista de referência se fornecida; senão, a quantidade que a empresa deseja comprar (em unidades de embalagem)

7. ITENS AUSENTES: Se um fornecedor não cotou o item, use null em preco_unit

8. OBSERVAÇÕES: NÃO adicione observações sobre normalização, conversão ou unidade padrão. Apenas observações relevantes do orçamento (frete, prazo, validade, etc.)

Retorne um array JSON onde cada elemento tem EXATAMENTE esta estrutura:
{{
  "id": int começando em 1,
  "item": "NOME CURTO PADRONIZADO EM MAIÚSCULAS",
  "marca": "marca única ou null",
  "quantidade": float (qtd de embalagens),
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
- Retorne APENAS o array JSON completo, sem markdown, sem explicações.
- NÃO normalize/divida preços. Mantenha o preço por embalagem de venda.
- Unidades APENAS: UN, CX, PCT, BB, KG
"""


# ── Funções públicas ──────────────────────────────────────────────────────────

def _normalize_unit(unit: str) -> str:
    """Normaliza unidade para uma das permitidas: UN, CX, PCT, BB, KG."""
    if not unit:
        return "UN"
    unit = unit.strip().upper()
    
    # Mapeamento de unidades comuns para as permitidas
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


def _post_process_items(items: list) -> list:
    """Pós-processamento para garantir conformidade com regras do mapa."""
    for item in items:
        # Normalizar unidade
        if "unidade" in item:
            item["unidade"] = _normalize_unit(item.get("unidade", "UN"))
        
        # Garantir que marca é única (sem "/")
        marca = item.get("marca")
        if marca and "/" in marca:
            # Pega apenas a primeira marca
            item["marca"] = marca.split("/")[0].strip()
        
        # Limpar nome - remover espaços extras
        if "item" in item:
            item["item"] = " ".join(item["item"].split()).upper()
    
    return items


def _post_process_normalized(items: list) -> list:
    """Pós-processamento para itens normalizados."""
    for item in items:
        # Normalizar unidade
        if "unidade" in item:
            item["unidade"] = _normalize_unit(item.get("unidade", "UN"))
        
        # Garantir marca única
        marca = item.get("marca")
        if marca and "/" in marca:
            item["marca"] = marca.split("/")[0].strip()
        
        # Limpar nome
        if "item" in item:
            item["item"] = " ".join(item["item"].split()).upper()
        
        # Remover observações sobre normalização
        obs = item.get("observacao")
        if obs and any(kw in obs.lower() for kw in ["normaliz", "unidade padrão", "preços convertidos", "preço por unidade"]):
            item["observacao"] = None
        
        # Limpar obs dos fornecedores
        fornecedores = item.get("fornecedores", {})
        for fname, fdata in fornecedores.items():
            if isinstance(fdata, dict):
                fobs = fdata.get("obs")
                if fobs and any(kw in fobs.lower() for kw in ["normaliz", "unidade padrão", "convertid"]):
                    fdata["obs"] = None
    
    return items


def extract_items_from_text(text: str, preferences_context: str = "") -> list:
    """Extrai itens de PDF com texto selecionável."""
    time.sleep(_INTER_CALL_DELAY)
    raw = _call_with_retry(EXTRACTION_PROMPT.format(
        preferences=preferences_context or "Nenhuma preferência registrada ainda.",
        texto=text[:14000]
    ))
    # Prompt de texto → JSON nativo garantido pelo generation_config
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        items = _parse_json_response(raw)   # fallback tolerante
    return _post_process_items(items)


def extract_items_from_images(images_b64: list, preferences_context: str = "") -> list:
    """Extrai itens de PDF escaneado via Gemini Vision."""
    time.sleep(_INTER_CALL_DELAY)
    vision_prompt = EXTRACTION_PROMPT_VISION.format(
        preferences=preferences_context or "Nenhuma preferência registrada ainda."
    )
    parts = [vision_prompt] + [
        {"mime_type": "image/png", "data": b64} for b64 in images_b64
    ]
    raw = _call_with_retry(parts)
    items = _parse_json_response(raw)
    return _post_process_items(items)


def normalize_and_match(
    supplier_items: dict,
    reference_list=None,
    preferences_context: str = "",
) -> list:
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
        preferences=preferences_context or "Nenhuma preferência registrada ainda.",
    )

    raw  = _call_with_retry(prompt)
    # Prompt de texto → JSON nativo garantido pelo generation_config
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = _parse_json_response(raw)   # fallback tolerante

    # Remap fornecedor_1..N → nomes reais
    key_map = {f"fornecedor_{i+1}": name for i, name in enumerate(suppliers)}
    for item in data:
        if "fornecedores" in item:
            item["fornecedores"] = {
                key_map.get(k, k): v for k, v in item["fornecedores"].items()
            }

    # Pós-processamento para garantir conformidade
    data = _post_process_normalized(data)

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
