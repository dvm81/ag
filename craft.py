
# ingest_text_oss.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import io, os, re, base64

# PDF
import fitz  # PyMuPDF
import pymupdf4llm

# HTML
from bs4 import BeautifulSoup

# Imaging
from PIL import Image
import numpy as np

# -----------------------------
# Data types
# -----------------------------
@dataclass
class TextBlock:
    doc_id: str
    page: int
    block_id: str
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    source: Optional[str] = None  # "pdf|html|img"
    extra: Optional[dict] = None  # any per-block metadata


@dataclass
class ExtractionResult:
    text: str
    blocks: List[TextBlock]


# -----------------------------
# OCR engines (open-source)
# -----------------------------
class OCREngine:
    """Abstract OCR engine returning (full_text, blocks)."""
    def ocr_bytes(self, b: bytes, doc_id="img", page=1) -> ExtractionResult:
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    def __init__(self, lang: str = "en"):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PaddleOCR not installed. `pip install paddleocr` (and paddlepaddle)."
            ) from e
        # Use det+rec only; disable cls unless you need orientation handling
        self.ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)

    def ocr_bytes(self, b: bytes, doc_id="img", page=1) -> ExtractionResult:
        # PaddleOCR accepts file path or ndarray
        img = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
        result = self.ocr.ocr(img, cls=False)

        blocks: List[TextBlock] = []
        lines: List[str] = []

        # result is list per page; when passing ndarray, it's [[line,...]]
        for line_idx, item in enumerate(result[0] if result else []):
            poly, (txt, conf) = item
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            t = (txt or "").strip()
            if not t:
                continue
            lines.append(t)
            blocks.append(TextBlock(
                doc_id=doc_id, page=page, block_id=f"line_{line_idx}",
                text=t, bbox=bbox, source="img",
                extra={"confidence": float(conf)}
            ))

        return ExtractionResult(text="\n".join(lines), blocks=blocks)


class TesseractOCREngine(OCREngine):
    def __init__(self, lang: str = "eng", psm: int = 6, oem: int = 3):
        try:
            import pytesseract  # noqa: F401
            from pytesseract import Output  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "pytesseract not installed. `pip install pytesseract` "
                "and install the Tesseract binary on the system."
            ) from e
        self.lang = lang
        self.psm = psm
        self.oem = oem

    def ocr_bytes(self, b: bytes, doc_id="img", page=1) -> ExtractionResult:
        import pytesseract
        from pytesseract import Output

        img = Image.open(io.BytesIO(b)).convert("RGB")
        cfg = f"--psm {self.psm} --oem {self.oem}"
        data = pytesseract.image_to_data(img, lang=self.lang, output_type=Output.DICT, config=cfg)

        blocks: List[TextBlock] = []
        lines: List[str] = []
        n = len(data["text"])
        for i in range(n):
            t = (data["text"][i] or "").strip()
            conf = float(data.get("conf", ["-1"])[i])
            if not t or conf < 0:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = (x, y, x + w, y + h)
            blocks.append(TextBlock(
                doc_id=doc_id, page=page, block_id=f"token_{i}",
                text=t, bbox=bbox, source="img", extra={"confidence": conf/100.0}
            ))
            lines.append(t)

        # Simple line reconstruction; for robust grouping, post-process by line_num
        if "line_num" in data:
            by_line = {}
            for i in range(n):
                if not (data["text"][i] or "").strip():
                    continue
                key = (data.get("page_num", [page])[i], data.get("block_num", [0])[i],
                       data.get("par_num", [0])[i], data.get("line_num", [0])[i])
                by_line.setdefault(key, []).append((data["left"][i], data["text"][i]))
            lines = [" ".join(t for _, t in sorted(v, key=lambda x: x[0])) for v in by_line.values()]

        return ExtractionResult(text="\n".join(lines), blocks=blocks)


def make_ocr_engine(engine: str = "paddle", **kwargs) -> OCREngine:
    """
    engine: "paddle" or "tesseract"
    kwargs forwarded to the specific engine (e.g., lang="en", psm=6)
    """
    if engine == "paddle":
        return PaddleOCREngine(**kwargs)
    if engine == "tesseract":
        return TesseractOCREngine(**kwargs)
    raise ValueError("engine must be 'paddle' or 'tesseract'")


# -----------------------------
# Helpers
# -----------------------------
_DATA_URL_RE = re.compile(
    r"^data:image/(?P<fmt>png|jpeg|jpg|gif|bmp|tiff|webp);base64,(?P<b64>[A-Za-z0-9+/=\s]+)$",
    re.IGNORECASE
)

def _visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.extract()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _is_scanned_pdf(doc: fitz.Document, char_threshold: int = 50, ratio_threshold: float = 0.7) -> bool:
    """If >= ratio_threshold of pages have < char_threshold extracted characters, treat as scanned."""
    if len(doc) == 0:
        return False
    low_text_pages = sum(1 for p in doc if len(p.get_text("text").strip()) < char_threshold)
    return (low_text_pages / len(doc)) >= ratio_threshold


# -----------------------------
# Core extractors
# -----------------------------
def _ocr_image_bytes(ocr: OCREngine, img_bytes: bytes, doc_id="img", page=1) -> ExtractionResult:
    return ocr.ocr_bytes(img_bytes, doc_id=doc_id, page=page)


def extract_text_from_pdf(
    pdf: Union[str, bytes, io.BufferedIOBase],
    ocr: Optional[OCREngine] = None,
    force_ocr: bool = False,
    doc_id: str = "pdf",
    render_scale: float = 2.0  # ~144 DPI; bump to 3.0 for tough scans
) -> ExtractionResult:
    """
    Convert a PDF to text.
      - Digital PDFs -> PyMuPDF4LLM Markdown (fast, preserves structure)
      - Scanned PDFs (or force_ocr=True) -> render pages -> OCR (Paddle/Tesseract)
    """
    # Load PDF
    if isinstance(pdf, (bytes, bytearray)):
        pdf_bytes = bytes(pdf)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elif isinstance(pdf, str):
        doc = fitz.open(pdf)
    else:
        pdf_bytes = pdf.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Digital: use PyMuPDF4LLM
    if not force_ocr and not _is_scanned_pdf(doc):
        md = pymupdf4llm.to_markdown(doc)
        blocks: List[TextBlock] = []
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text")
            if txt.strip():
                blocks.append(TextBlock(
                    doc_id=doc_id, page=i, block_id=f"page_{i}",
                    text=txt, source="pdf"
                ))
        return ExtractionResult(text=md, blocks=blocks)

    # Scanned: OCR page images
    if ocr is None:
        ocr = make_ocr_engine("paddle")  # default to better accuracy
    M = fitz.Matrix(render_scale, render_scale)
    all_blocks: List[TextBlock] = []
    page_texts: List[str] = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=M, alpha=False)
        img_bytes = pix.tobytes("png")
        res = _ocr_image_bytes(ocr, img_bytes, doc_id=doc_id, page=i)
        page_texts.append(res.text)
        # Prefix block ids with page
        for k, b in enumerate(res.blocks):
            b.block_id = f"p{i}_{b.block_id}"
            b.source = "pdf"
            all_blocks.append(b)
    return ExtractionResult(text="\n\n".join(page_texts), blocks=all_blocks)


def _extract_text_from_data_url_image(data_url: str, ocr: OCREngine) -> Optional[ExtractionResult]:
    m = _DATA_URL_RE.match(data_url)
    if not m:
        return None
    b64 = m.group("b64")
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        return None
    return _ocr_image_bytes(ocr, img_bytes, doc_id="img", page=1)


def extract_text_from_html(
    html: Union[str, bytes],
    ocr: Optional[OCREngine] = None,
    include_base64_images: bool = True,
    doc_id: str = "html"
) -> ExtractionResult:
    """
    Convert HTML (optionally with base64 data-URL images) to text.
       - Returns visible DOM text
       - If include_base64_images=True, OCR each <img src="data:image/..."> and append text
    """
    if isinstance(html, (bytes, bytearray)):
        html_str = html.decode("utf-8", errors="ignore")
    else:
        html_str = html

    dom_text = _visible_text_from_html(html_str)
    blocks: List[TextBlock] = [
        TextBlock(doc_id=doc_id, page=1, block_id="dom", text=dom_text, source="html")
    ]
    parts: List[str] = [dom_text] if dom_text else []

    if include_base64_images:
        if ocr is None:
            ocr = make_ocr_engine("paddle")
        soup = BeautifulSoup(html_str, "lxml")
        img_idx = 0
        for img in soup.find_all("img"):
            src = img.get("src") or ""
            res = _extract_text_from_data_url_image(src, ocr)
            if res and res.text.strip():
                parts.append(res.text)
                # Merge blocks
                for b in res.blocks:
                    blocks.append(TextBlock(
                        doc_id=doc_id, page=1, block_id=f"img_{img_idx}_{b.block_id}",
                        text=b.text, bbox=b.bbox, source="img",
                        extra={"alt": img.get("alt")}
                    ))
                img_idx += 1

    full = "\n\n".join([p for p in parts if p.strip()])
    return ExtractionResult(text=full, blocks=blocks)


# -----------------------------
# Unified router
# -----------------------------
def extract_text(
    content: Union[str, bytes, io.BufferedIOBase],
    content_type: str,
    ocr: Optional[OCREngine] = None,
    **kwargs
) -> ExtractionResult:
    """
    Route based on content_type.
      - "application/pdf" -> extract_text_from_pdf
      - "text/html"       -> extract_text_from_html
      - Otherwise raises.
    kwargs are forwarded to the specific extractor.
    """
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == "application/pdf":
        return extract_text_from_pdf(content, ocr=ocr, **kwargs)
    if ct == "text/html":
        return extract_text_from_html(content, ocr=ocr, **kwargs)
    raise ValueError(f"Unsupported content_type: {content_type}")
