# fast_pdf_extract.py
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import io, os
import fitz  # PyMuPDF

@dataclass
class TextBlock:
    doc_id: str
    page: int
    block_id: str
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: Optional[str] = None
    extra: Optional[dict] = None

@dataclass
class ExtractionResult:
    text: str
    blocks: List[TextBlock]

def extract_text_from_pdf_fast(
    pdf: Union[str, bytes, io.BufferedIOBase],
    doc_id: str = "pdf",
    # OCR engine: your Paddle/Tesseract engine implementing ocr_bytes(b, doc_id, page)
    ocr=None,
    # only OCR when the text layer looks empty
    min_total_chars_for_digital: int = 200,
    prefer_markdown: bool = False,   # set True if you later want pymupdf4llm.to_markdown
    ocr_render_scale: float = 2.0,   # 144 dpi is usually enough
):
    # --- open once, in-memory ---
    if isinstance(pdf, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(pdf), filetype="pdf")
    elif isinstance(pdf, str):
        doc = fitz.open(pdf)
    else:
        buf = pdf.read()
        doc = fitz.open(stream=buf, filetype="pdf")

    # --- super fast path: use the text layer ---
    page_texts: List[str] = []
    for p in doc:
        # "text" is fast and good enough; use "blocks" if you want order by blocks
        t = p.get_text("text")
        page_texts.append(t or "")
    total_chars = sum(len(t) for t in page_texts)

    if total_chars >= min_total_chars_for_digital or any(t.strip() for t in page_texts):
        # return raw text quickly (skip slow markdown reconstruction)
        blocks = [TextBlock(doc_id=doc_id, page=i+1, block_id=f"page_{i+1}", text=txt, source="pdf")
                  for i, txt in enumerate(page_texts) if txt.strip()]
        return ExtractionResult(text="\n\n".join(page_texts), blocks=blocks)

    # --- fallback: scanned/OCR path ---
    if ocr is None:
        raise ValueError("No OCR engine provided for scanned PDFs. Reuse a global Paddle/Tesseract instance.")

    M = fitz.Matrix(ocr_render_scale, ocr_render_scale)
    texts, blocks = [], []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=M, alpha=False)
        img_bytes = pix.tobytes("png")
        res = ocr.ocr_bytes(img_bytes, doc_id=doc_id, page=i)
        texts.append(res.text)
        for b in res.blocks:
            b.block_id = f"p{i}_{b.block_id}"
            b.source = "pdf"
            blocks.append(b)

    return ExtractionResult(text="\n\n".join(texts), blocks=blocks)
