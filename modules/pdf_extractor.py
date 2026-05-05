# -*- coding: utf-8 -*-
"""
pdf_extractor.py
Extrai texto de PDFs (nativos ou escaneados).
PDFs com texto escasso são detectados automaticamente e roteados para OCR via Gemini Vision.
"""
import fitz  # PyMuPDF
import base64


MIN_CHARS_PER_PAGE = 100  # limiar para detectar PDF como imagem


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, bool]:
    """
    Retorna (texto_extraido, is_image_based).
    Se is_image_based=True, o caller deve usar extract_images_from_pdf() para OCR.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(
            f"Não foi possível abrir o PDF. Verifique se o arquivo é um PDF válido. Detalhe: {e}"
        )
    pages_text = []
    total_chars = 0

    for page in doc:
        # get_text retorna str em Python 3 — PyMuPDF lida com encoding internamente
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        # Garante que é string válida UTF-8 (remove surrogates se houver)
        text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        pages_text.append(text)
        total_chars += len(text.strip())

    doc.close()

    # Usa mediana para evitar falso positivo por páginas de capa/rodapé em branco
    char_counts = sorted([len(t.strip()) for t in pages_text])
    mid = len(char_counts) // 2
    median_chars = char_counts[mid] if char_counts else 0
    is_image_based = median_chars < MIN_CHARS_PER_PAGE

    return "\n\n--- PÁGINA ---\n\n".join(pages_text), is_image_based


def extract_images_from_pdf(pdf_bytes: bytes, max_pages: int = 10) -> list[str]:
    """
    Converte cada página do PDF em imagem base64 (PNG) para envio ao Gemini Vision.
    Retorna lista de strings base64.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_b64 = []

    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        # DPI alto para melhor OCR
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images_b64.append(b64)

    doc.close()
    return images_b64


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = len(doc)
    doc.close()
    return n
