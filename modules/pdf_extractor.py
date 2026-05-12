# -*- coding: utf-8 -*-
"""
pdf_extractor.py
Extrai texto de PDFs (nativos ou escaneados) e converte imagens PNG/JPEG para base64.
PDFs com texto escasso são detectados automaticamente e roteados para OCR via Gemini Vision.
"""
import fitz  # PyMuPDF
import base64
from io import BytesIO
from PIL import Image


MIN_CHARS_PER_PAGE = 100  # limiar para detectar PDF como imagem


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple:
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
    try:
        for page in doc:
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            pages_text.append(text)
    finally:
        doc.close()

    char_counts = sorted([len(t.strip()) for t in pages_text])
    mid = len(char_counts) // 2
    median_chars = char_counts[mid] if char_counts else 0
    is_image_based = median_chars < MIN_CHARS_PER_PAGE

    return "\n\n--- PÁGINA ---\n\n".join(pages_text), is_image_based


def extract_images_from_pdf(pdf_bytes: bytes, max_pages: int = 10) -> list:
    """
    Converte cada página do PDF em imagem base64 (PNG) para envio ao Gemini Vision.
    Retorna lista de strings base64.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_b64 = []
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            images_b64.append(b64)
    finally:
        doc.close()
    return images_b64


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return len(doc)
    finally:
        doc.close()


# ── Suporte a imagens PNG / JPEG ──────────────────────────────────────────────

def image_to_base64_png(image_bytes: bytes) -> str:
    """
    Converte bytes de imagem (PNG ou JPEG) para base64 PNG normalizado.
    Usa Pillow para garantir conversão correta independente do formato original.
    """
    img = Image.open(BytesIO(image_bytes))
    # Converte para RGB se necessário (ex: RGBA, paleta)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def image_to_base64_jpeg(image_bytes: bytes) -> str:
    """
    Converte bytes de imagem JPEG para base64 JPEG.
    """
    img = Image.open(BytesIO(image_bytes))
    if img.mode not in ("RGB",):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def detect_image_mime(image_bytes: bytes) -> str:
    """
    Detecta o mime type de uma imagem pelos magic bytes.
    Retorna 'image/png' ou 'image/jpeg'.
    """
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if image_bytes[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    # fallback: tenta abrir com Pillow
    try:
        img = Image.open(BytesIO(image_bytes))
        fmt = (img.format or "PNG").upper()
        return "image/jpeg" if fmt == "JPEG" else "image/png"
    except Exception:
        return "image/png"
