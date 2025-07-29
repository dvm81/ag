# html_text_extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import base64, re, io, os, tempfile

from bs4 import BeautifulSoup
from PIL import Image
import numpy as np

# ---------- result types ----------
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

# ---------- data-URL decoder (robust) ----------
def decode_data_url_to_bytes(src: str) -> Optional[bytes]:
    """
    Accepts data:image/*;base64,... (with possible whitespace/newlines).
    Returns raw bytes or None.
    """
    if not isinstance(src, str) or not src.lower().startswith("data:image/"):
        return None
    try:
        _, b64 = src.split(",", 1)
    except ValueError:
        return None
    b64 = re.sub(r"\s+", "", b64)  # tolerate wrapped base64
    try:
        return base64.b64decode(b64, validate=False)
    except Exception:
        return None

# ---------- EasyOCR engine ----------
class EasyOCREngine:
    """Light wrapper around easyocr.Reader with robust byte decoding."""
    def __init__(self, lang_list: list[str] = ["en"], gpu: bool = False):
        import easyocr  # lazy import
        self.reader = easyocr.Reader(lang_list, gpu=gpu)

    def ocr_bytes(self, b: bytes, doc_id: str = "img", page: int = 1) -> ExtractionResult:
        import cv2  # lazy import to avoid import if OCR unused

        if not isinstance(b, (bytes, bytearray)):
            b = bytes(b)

        # Try to decode into a proper ndarray with OpenCV
        img = None
        try:
            arr = np.frombuffer(b, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR ndarray
        except Exception:
            img = None

        # Fallback: write a temp file and pass path to EasyOCR
        try:
            if img is None or not hasattr(img, "shape"):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(b)
                    tmp.flush()
                    tmp_path = tmp.name
                try:
                    result = self.reader.readtext(tmp_path, detail=1)
                finally:
                    try: os.remove(tmp_path)
                    except OSError: pass
            else:
                result = self.reader.readtext(img, detail=1)
        except Exception:
            # Don’t fail the whole pipeline on a bad image
            return ExtractionResult(text="", blocks=[])

        lines, blocks = [], []
        for i, (bbox, txt, conf) in enumerate(result):
            t = (txt or "").strip()
            if not t:
                continue
            # bbox: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            (x0, y0), (_, _), (x2, y2), _ = bbox
            lines.append(t)
            blocks.append(TextBlock(
                doc_id=doc_id, page=page, block_id=f"line_{i}",
                text=t, bbox=(float(x0), float(y0), float(x2), float(y2)),
                source="img", extra={"confidence": float(conf)}
            ))
        return ExtractionResult(text="\n".join(lines), blocks=blocks)

# ---------- HTML extractor ----------
def extract_text_from_html(
    html: Union[str, bytes],
    ocr: Optional[EasyOCREngine] = None,
    include_base64_images: bool = True,
    on_ocr_unavailable: str = "skip",  # "skip" or "error"
    doc_id: str = "html"
) -> ExtractionResult:
    """
    - Extracts visible DOM text.
    - If include_base64_images=True, OCRs any <img src="data:image/...;base64,...">.
    - If OCR engine not provided and on_ocr_unavailable="skip", silently ignores images.
    """
    html_str = html.decode("utf-8", "ignore") if isinstance(html, (bytes, bytearray)) else html

    soup = BeautifulSoup(html_str, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.extract()

    # Visible DOM text
    dom_text = soup.get_text(separator="\n")
    dom_lines = [ln.strip() for ln in dom_text.splitlines()]
    dom_lines = [ln for ln in dom_lines if ln]
    dom_text = "\n".join(dom_lines)

    blocks: List[TextBlock] = [TextBlock(doc_id, 1, "dom", dom_text, source="html")]
    parts: List[str] = [dom_text] if dom_text else []

    # Optional: OCR data-URL images
    if include_base64_images:
        if ocr is None:
            if on_ocr_unavailable == "error":
                raise RuntimeError("OCR requested but no OCR engine provided.")
            # skip image OCR
        else:
            img_idx = 0
            for img in soup.find_all("img"):
                src = img.get("src") or ""
                img_bytes = decode_data_url_to_bytes(src)
                if not img_bytes:  # not a data-URL image
                    continue
                res = ocr.ocr_bytes(img_bytes, doc_id=doc_id, page=1)
                if not res.text.strip():
                    continue
                parts.append(res.text)
                for b in res.blocks:
                    blocks.append(TextBlock(
                        doc_id=doc_id, page=1,
                        block_id=f"img_{img_idx}_{b.block_id}",
                        text=b.text, bbox=b.bbox, source="img",
                        extra={"alt": img.get("alt")}
                    ))
                img_idx += 1

    full = "\n\n".join([p for p in parts if p.strip()])
    return ExtractionResult(text=full, blocks=blocks)



#
from html_text_extractor import extract_text_from_html, EasyOCREngine

ocr = EasyOCREngine(lang_list=["en"], gpu=False)  # GPU=False works anywhere
res = extract_text_from_html(html_str, ocr=ocr, include_base64_images=True)

print(res.text)          # full concatenated text (DOM + OCR from data-URL images)
# Access blocks if you need anchors for RAG/NER highlighting
for b in res.blocks[:5]:
    print(b.block_id, b.source, b.text[:80])
