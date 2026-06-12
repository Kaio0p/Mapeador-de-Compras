# -*- coding: utf-8 -*-
"""
llm_manager.py — Roteador e Pool de Chaves Multi-LLM
======================================================
Centraliza o gerenciamento de todos os clientes de LLM do sistema.

Arquitetura:
  • Pool de Chaves Gemini  — rotação round-robin via get_gemini_client(),
                             retorna instância limpa do novo SDK (google-genai)
                             a cada chamada, evitando 429.
  • Cliente Cohere Singleton — inicializado uma única vez com COHERE_API_KEY;
                               command-r-plus para normalização/lógica.

Roteamento:
  ┌──────────────────────────────────────────────────────────────┐
  │  PDF nativo (texto selec.)  →  Gemini (extract_items via genai Client)  │
  │  PDF escaneado / imagem     →  Gemini Vision (OCR)           │
  │  Normalização               →  Cohere command-r-plus         │
  │  Auditoria Final            →  Gemini (janela enorme)        │
  └──────────────────────────────────────────────────────────────┘

Secrets esperados em .streamlit/secrets.toml:
  COHERE_API_KEY  = "..."
  GEMINI_API_KEYS = ["AIza...", "AIza...", "AIza..."]
  # Retrocompatibilidade: GEMINI_API_KEY (string única) como fallback.
"""
import random
import logging
from typing import Optional

import streamlit as st
from google import genai

logger = logging.getLogger(__name__)

# ── Cohere client singleton ───────────────────────────────────────────────────

_cohere_client = None


def get_cohere_client():
    """
    Retorna o cliente Cohere V2, inicializando-o na primeira chamada.
    Usa cohere.ClientV2 (SDK v5+) — API compatível com modelos command-a-*.
    Lança ValueError se COHERE_API_KEY não estiver nos secrets.
    """
    global _cohere_client
    if _cohere_client is not None:
        return _cohere_client

    try:
        import cohere
    except ImportError as e:
        raise ImportError(
            "Biblioteca 'cohere' não instalada. Execute: pip install cohere>=5.0.0"
        ) from e

    api_key = _load_cohere_key()
    if not api_key:
        raise ValueError(
            "COHERE_API_KEY não encontrada nos secrets. "
            "Adicione ao arquivo .streamlit/secrets.toml:\n"
            "  COHERE_API_KEY = \"...\""
        )

    # ClientV2 é a API moderna (SDK >= 5.x) — suporta command-a-* e response_format
    # Timeout de 120s para evitar "read operation timed out" em redes lentas
    _cohere_client = cohere.ClientV2(api_key=api_key, timeout=120.0)
    logger.info("[LLM Manager] Cliente Cohere V2 inicializado com sucesso (timeout=120s).")
    return _cohere_client


def _load_cohere_key() -> Optional[str]:
    """Carrega COHERE_API_KEY dos st.secrets com fallback seguro."""
    try:
        return st.secrets.get("COHERE_API_KEY", "") or ""
    except Exception:
        return ""


# ── Gemini key pool ───────────────────────────────────────────────────────────

def _load_gemini_keys() -> list:
    """
    Carrega a lista de chaves Gemini dos secrets.
    Tenta GEMINI_API_KEYS (lista) e cai para GEMINI_API_KEY (string).
    """
    try:
        keys = st.secrets.get("GEMINI_API_KEYS", None)
        if keys:
            if isinstance(keys, (list, tuple)):
                valid = [k.strip() for k in keys if k and k.strip()]
                if valid:
                    return valid
            if isinstance(keys, str):
                parts = [k.strip() for k in keys.split(",") if k.strip()]
                if parts:
                    return parts

        # Fallback: chave única
        single = st.secrets.get("GEMINI_API_KEY", "") or ""
        if single:
            return [single]

    except Exception as e:
        logger.warning("[LLM Manager] Não foi possível carregar chaves Gemini: %s", e)

    return []


# Índice global para rotação round-robin
_gemini_key_index = 0


def get_random_gemini_key() -> Optional[str]:
    """
    Escolhe uma chave Gemini aleatória do pool.
    Retorna None se nenhuma chave estiver configurada.
    """
    keys = _load_gemini_keys()
    if not keys:
        logger.error("[LLM Manager] Nenhuma chave Gemini encontrada nos secrets.")
        return None
    chosen = random.choice(keys)
    logger.debug("[LLM Manager] Chave Gemini selecionada: ...%s", chosen[-6:])
    return chosen


def get_next_gemini_key() -> Optional[str]:
    """
    Rotação round-robin: retorna a PRÓXIMA chave do pool sequencialmente.
    Garante que cada tentativa use uma chave diferente antes de repetir.
    Retorna None se nenhuma chave estiver configurada.
    """
    global _gemini_key_index
    keys = _load_gemini_keys()
    if not keys:
        logger.error("[LLM Manager] Nenhuma chave Gemini encontrada nos secrets.")
        return None
    chosen = keys[_gemini_key_index % len(keys)]
    _gemini_key_index = (_gemini_key_index + 1) % len(keys)
    logger.debug("[LLM Manager] Chave Gemini round-robin [%d/%d]: ...%s",
                 _gemini_key_index, len(keys), chosen[-6:])
    return chosen


def get_gemini_client() -> genai.Client:
    """
    Cria e retorna uma instância limpa do novo SDK google-genai
    usando a próxima chave disponível na lista (round-robin).

    Cada chamada devolve um Client fresco com a chave rotacionada,
    evitando acúmulo de estado e distribuindo carga entre as chaves.

    Raises:
        RuntimeError: se nenhuma chave Gemini estiver configurada.
    """
    key = get_next_gemini_key()
    if not key:
        raise RuntimeError(
            "Nenhuma chave Gemini configurada. "
            "Adicione GEMINI_API_KEYS ao .streamlit/secrets.toml"
        )
    return genai.Client(api_key=key)


def configure_gemini_with_key(api_key: str) -> None:
    """
    Retrocompatibilidade: valida que a chave pode criar um Client.
    Não configura estado global — o novo SDK é stateless por design.
    """
    # Apenas testa se a chave é utilizável criando um client
    _ = genai.Client(api_key=api_key)


# ── Status helpers ─────────────────────────────────────────────────────────────

def get_system_status() -> dict:
    """
    Retorna status de cada componente do sistema.
    Útil para exibir indicadores na sidebar.
    """
    gemini_keys = _load_gemini_keys()
    cohere_key  = _load_cohere_key()

    return {
        "cohere_configured":   bool(cohere_key),
        "gemini_configured":   bool(gemini_keys),
        "gemini_key_count":    len(gemini_keys),
        "cohere_key_preview":  "...{}".format(cohere_key[-6:]) if cohere_key else "—",
    }
