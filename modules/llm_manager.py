# -*- coding: utf-8 -*-
"""
llm_manager.py — Roteador e Pool de Chaves Multi-LLM
======================================================
Centraliza o gerenciamento de todos os clientes de LLM do sistema.

Arquitetura:
  • Pool de Chaves Gemini  — escolhe chave aleatória da lista GEMINI_API_KEYS
                             (st.secrets) a cada chamada, distribuindo carga e
                             evitando bloqueios 429 por rate-limit.
  • Cliente Groq Singleton — inicializado uma única vez com GROQ_API_KEY;
                             ultra-rápido para texto (LPU inference).

Roteamento recomendado:
  ┌──────────────────────────────────────────────────────────────┐
  │  PDF nativo (texto selec.)  →  Groq  (groq_processor.py)    │
  │  PDF escaneado / imagem     →  Gemini Vision                 │
  │                                (gemini_processor.py)         │
  │  Normalização + Auditoria   →  Groq  (groq_processor.py)    │
  └──────────────────────────────────────────────────────────────┘

Secrets esperados em .streamlit/secrets.toml:
  GROQ_API_KEY   = "gsk_..."
  GEMINI_API_KEYS = ["AIza...", "AIza...", "AIza..."]

  # Retrocompatibilidade: se GEMINI_API_KEYS não existir,
  # usa GEMINI_API_KEY (string única) como fallback.
"""
import random
import logging
from typing import Optional

import streamlit as st
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ── Groq client singleton ─────────────────────────────────────────────────────

_groq_client = None


def get_groq_client():
    """
    Retorna o cliente Groq, inicializando-o na primeira chamada.
    Lança ValueError se GROQ_API_KEY não estiver nos secrets.
    """
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError(
            "Biblioteca 'groq' não instalada. Execute: pip install groq>=0.9.0"
        ) from e

    api_key = _load_groq_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY não encontrada nos secrets. "
            "Adicione ao arquivo .streamlit/secrets.toml:\n"
            "  GROQ_API_KEY = \"gsk_...\""
        )

    _groq_client = Groq(api_key=api_key)
    logger.info("[LLM Manager] Cliente Groq inicializado com sucesso.")
    return _groq_client


def _load_groq_key() -> Optional[str]:
    """Carrega GROQ_API_KEY dos st.secrets com fallback seguro."""
    try:
        return st.secrets.get("GROQ_API_KEY", "") or ""
    except Exception:
        return ""


# ── Gemini key pool ───────────────────────────────────────────────────────────

def _load_gemini_keys() -> list:
    """
    Carrega a lista de chaves Gemini dos secrets.

    Tenta em ordem:
      1. GEMINI_API_KEYS (lista de strings) — modo pool
      2. GEMINI_API_KEY  (string única)     — retrocompatibilidade
    """
    try:
        keys = st.secrets.get("GEMINI_API_KEYS", None)
        if keys:
            # Pode vir como lista TOML ou como string separada por vírgula
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
        logger.warning(f"[LLM Manager] Não foi possível carregar chaves Gemini: {e}")

    return []


def get_random_gemini_key() -> Optional[str]:
    """
    Escolhe uma chave Gemini aleatória do pool.

    Estratégia: seleção uniforme aleatória — simples, sem estado, sem bloqueio.
    Em caso de 429, a próxima chamada (retry) provavelmente pega outra chave.

    Retorna None se nenhuma chave estiver configurada.
    """
    keys = _load_gemini_keys()
    if not keys:
        logger.error("[LLM Manager] Nenhuma chave Gemini encontrada nos secrets.")
        return None
    chosen = random.choice(keys)
    logger.debug(f"[LLM Manager] Chave Gemini selecionada: ...{chosen[-6:]}")
    return chosen


def configure_gemini_random() -> bool:
    """
    Configura o módulo google.generativeai com uma chave aleatória do pool.
    Deve ser chamado imediatamente antes de cada requisição ao Gemini.

    Retorna True se configurado com sucesso, False caso contrário.
    """
    key = get_random_gemini_key()
    if not key:
        return False
    try:
        genai.configure(api_key=key)
        return True
    except Exception as e:
        logger.error(f"[LLM Manager] Erro ao configurar Gemini: {e}")
        return False


def configure_gemini_with_key(api_key: str) -> None:
    """
    Configura o Gemini com uma chave específica.
    Usado para retrocompatibilidade com o fluxo antigo de _init_gemini().
    """
    genai.configure(api_key=api_key)


# ── Status helpers ─────────────────────────────────────────────────────────────

def get_system_status() -> dict:
    """
    Retorna um dicionário com o status de cada componente do sistema.
    Útil para exibir indicadores na sidebar.
    """
    gemini_keys = _load_gemini_keys()
    groq_key    = _load_groq_key()

    return {
        "groq_configured":    bool(groq_key),
        "gemini_configured":  bool(gemini_keys),
        "gemini_key_count":   len(gemini_keys),
        "groq_key_preview":   f"...{groq_key[-6:]}" if groq_key else "—",
    }
