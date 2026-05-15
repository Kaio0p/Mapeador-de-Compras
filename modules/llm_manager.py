# -*- coding: utf-8 -*-
"""
llm_manager.py — Roteador e Pool de Chaves Multi-LLM
======================================================
Centraliza o gerenciamento de todos os clientes de LLM do sistema.

Arquitetura:
  • Pool de Chaves Gemini  — escolhe chave aleatória da lista GEMINI_API_KEYS
                             (st.secrets) a cada chamada, evitando 429.
  • Cliente Cohere Singleton — inicializado uma única vez com COHERE_API_KEY;
                               command-r-plus para normalização/lógica.

Roteamento:
  ┌──────────────────────────────────────────────────────────────┐
  │  PDF nativo (texto selec.)  →  Gemini (extract_items_from_text via Vision)  │
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
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ── Cohere client singleton ───────────────────────────────────────────────────

_cohere_client = None


def get_cohere_client():
    """
    Retorna o cliente Cohere, inicializando-o na primeira chamada.
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

    import cohere
    _cohere_client = cohere.Client(api_key=api_key)
    logger.info("[LLM Manager] Cliente Cohere inicializado com sucesso.")
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


def configure_gemini_random() -> bool:
    """
    Configura o módulo google.generativeai com uma chave aleatória do pool.
    Retorna True se configurado com sucesso, False caso contrário.
    """
    key = get_random_gemini_key()
    if not key:
        return False
    try:
        genai.configure(api_key=key)
        return True
    except Exception as e:
        logger.error("[LLM Manager] Erro ao configurar Gemini: %s", e)
        return False


def configure_gemini_with_key(api_key: str) -> None:
    """Configura o Gemini com uma chave específica (retrocompatibilidade)."""
    genai.configure(api_key=api_key)


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

