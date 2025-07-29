# ingest_text.py  (open‑source only)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Iterable
import io, os, re, base64, tempfile

# ---------- core deps ----------
import fitz                       # PyMuPDF
import pymupdf4llm                # PDF → Markdown (digital)
from bs4 import BeautifulSoup     # HTML
import pytesseract                # OCR
from PIL import Image             # bitmap wrapper for Tesseract
# --------------------------------

@dataclass
class TextBlock:
    doc_id: str
    page: int
    block_id: str
    text: str
    bbox: Optional[Tuple[int, int, int, int]] = None   # (x0, y0, x1, y1) in px
    source: Optional[str] = None                       # "pdf|html|img"
    extra: Optional[dict] = None


@dataclass
class ExtractionResult:
    text: str
    blocks: List[TextBlock]


# ---------------------------------------------------------------------------
# OCR helpers (Tesseract)
# ---------------------------------------------------------------------------
def _ocr_image_bytes(buf: bytes, psm: int = 3, lang: str = "eng") -> Tuple[str, List[TextBlock]]:
    """
    OCR a single image (bytes) with Tesseract.  
    Returns (full_text, blocks_with_bboxes).  
    Uses Tesseract's TSV output to build simple word‑level blocks.
    """
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    # TSV output: level, page, block, par, line, word, left, top, width, height, conf, text
    tsv = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i, txt in enumerate(tsv["text"]):
        txt = txt.strip()
        if not txt:
            continue
        x, y, w, h = (tsv[k][i] for k in ("left", "top", "width", "height"))
        words.append(txt)
        boxes.append(TextBlock(
            doc_id="doc",
            page=1,
            block_id=f"w{i}",
            text=txt,
            bbox=(x, y, x + w, y + h),
            source="img"
        ))
    return " ".join(words), boxes


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def _is_scanned_pdf(doc: fitz.Document, char_thresh: int = 50, ratio_thresh: float = .7) -> bool:
    low = sum(1 for p in doc if len(p.get_text("text").strip()) < char_thresh)
    return len(doc) > 0 and (low / len(doc)) >= ratio_thresh


def _ocr_pdf_pages(doc: fitz.Document, dpi: int = 200, lang: str = "eng") -> ExtractionResult:
    """
    Render each page to an image and OCR with Tesseract.
    """
    blocks: List[TextBlock] = []
    parts: List[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        txt, word_blocks = _ocr_image_bytes(pix.tobytes(), lang=lang)
        parts.append(txt)
        # re‑label page numbers
        for b in word_blocks:
            b.page = page.number + 1
            b.source = "pdf"
        blocks.extend(word_blocks)
    return ExtractionResult(text="\n".join(parts), blocks=blocks)


def extract_text_from_pdf(
    pdf: Union[str, bytes, io.BufferedIOBase],
    force_ocr: bool = False,
    lang: str = "eng",
    dpi: int = 200,
    doc_id: str = "pdf"
) -> ExtractionResult:
    """
    Digital PDFs → PyMuPDF4LLM Markdown.  
    Scanned PDFs (or force_ocr=True) → Tesseract OCR.
    """
    # load into PyMuPDF
    if isinstance(pdf, (bytes, bytearray)):
        pdf_bytes = bytes(pdf)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elif isinstance(pdf, str):
        doc = fitz.open(pdf)
    else:
        pdf_bytes = pdf.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if not force_ocr and not _is_scanned_pdf(doc):
        md = pymupdf4llm.to_markdown(doc)
        blocks = [TextBlock(doc_id=doc_id, page=i + 1,
                            block_id=f"page_{i+1}",
                            text=p.get_text("text"),
                            source="pdf")
                  for i, p in enumerate(doc)]
        return ExtractionResult(text=md, blocks=blocks)

    # OCR path
    res = _ocr_pdf_pages(doc, dpi=dpi, lang=lang)
    for b in res.blocks:
        b.doc_id = doc_id
    return res


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
_DATA_URL_RE = re.compile(
    r"^data:image/(?:png|jpe?g|gif|bmp|webp|tiff);base64,(?P<b64>[A-Za-z0-9+/=\s]+)$",
    re.IGNORECASE
)

def _visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "template"]):
        t.extract()
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _ocr_base64_image(src: str, lang: str = "eng") -> Tuple[str, List[TextBlock]]:
    m = _DATA_URL_RE.match(src)
    if not m:
        return "", []
    try:
        img_bytes = base64.b64decode(m.group("b64"))
    except Exception:
        return "", []
    return _ocr_image_bytes(img_bytes, lang=lang)


def extract_text_from_html(
    html: Union[str, bytes],
    include_base64_images: bool = True,
    lang: str = "eng",
    doc_id: str = "html"
) -> ExtractionResult:
    if isinstance(html, (bytes, bytearray)):
        html = html.decode("utf-8", errors="ignore")

    dom_text = _visible_text_from_html(html)
    blocks = [TextBlock(doc_id=doc_id, page=1, block_id="dom", text=dom_text, source="html")]
    parts = [dom_text]

    if include_base64_images:
        soup = BeautifulSoup(html, "lxml")
        img_idx = 0
        for img in soup.find_all("img"):
            txt, word_blocks = _ocr_base64_image(img.get("src", ""), lang=lang)
            if txt:
                parts.append(txt)
                for b in word_blocks:
                    b.block_id = f"img_{img_idx}_{b.block_id}"
                    b.page = 1
                    b.source = "img"
                    b.doc_id = doc_id
                blocks.extend(word_blocks)
                img_idx += 1

    full = "\n\n".join(p for p in parts if p.strip())
    return ExtractionResult(text=full, blocks=blocks)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def extract_text(
    content: Union[str, bytes, io.BufferedIOBase],
    content_type: str,
    **kwargs
) -> ExtractionResult:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == "application/pdf":
        return extract_text_from_pdf(content, **kwargs)
    if ct == "text/html":
        return extract_text_from_html(content, **kwargs)
    raise ValueError(f"Unsupported content_type: {content_type}")



#USAGE
from ingest_text import extract_text

# 1) digital vs scanned PDF handled automatically
with open("invoice.pdf", "rb") as f:
    res = extract_text(f, "application/pdf")
print(res.text[:500])

# 2) force OCR if you know it's a scan
res_scan = extract_text(open("scan_form.pdf", "rb"),
                        "application/pdf",
                        force_ocr=True, lang="eng+deu")

# 3) HTML with embedded data‑URL charts
html = open("report.html").read()
res_html = extract_text(html, "text/html", include_base64_images=True)
