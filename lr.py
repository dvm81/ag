import pytesseract, os
pytesseract.pytesseract.tesseract_cmd = os.path.expanduser("~/.local/bin/tesseract")  # or your path
from ingest_text_oss import extract_text, make_ocr_engine
ocr = make_ocr_engine("tesseract", lang="eng", psm=6, oem=3)
res = extract_text(html_str, "text/html", ocr=ocr, include_base64_images=True)


#python3 -m pip install easyocr pillow


# minimal EasyOCR engine
import io
from dataclasses import dataclass
from PIL import Image
import numpy as np
import easyocr

@dataclass
class TextBlock: ...
@dataclass
class ExtractionResult: ...

class EasyOCREngine:
    def __init__(self, lang_list=["en"]):
        self.reader = easyocr.Reader(lang_list, gpu=False)
    def ocr_bytes(self, b, doc_id="img", page=1):
        img = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
        result = self.reader.readtext(img, detail=1)
        lines, blocks = [], []
        for i, (bbox, txt, conf) in enumerate(result):
            lines.append(txt)
            (x0,y0),(x1,_),(x2,y2),_ = bbox
            blocks.append(TextBlock(doc_id, page, f"line_{i}", txt, (x0,y0,x2,y2), "img", {"confidence": float(conf)}))
        return ExtractionResult("\n".join(lines), blocks)

# use it in extract_text(html, ocr=EasyOCREngine(), include_base64_images=True)











#or EasyOCREngine.ocr_bytes
import io, os, tempfile
import numpy as np
import cv2
from PIL import Image

def ocr_bytes(self, b: bytes, doc_id="img", page=1):
    """
    Robustly decode base64 image bytes for EasyOCR.
    1) Try OpenCV imdecode (fast, returns np.ndarray).
    2) Fallback: write to a temp file and pass the path to EasyOCR.
    """
    # --- try OpenCV decode ---
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR np.ndarray

    if img is None or not hasattr(img, "shape"):
        # --- fallback: temp file ---
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

    # ---- normalize EasyOCR result -> lines/blocks ----
    lines, blocks = [], []
    for i, (bbox, txt, conf) in enumerate(result):
        t = (txt or "").strip()
        if not t:
            continue
        # bbox is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        (x0,y0), (_, _), (x2,y2), _ = bbox
        lines.append(t)
        blocks.append(TextBlock(
            doc_id=doc_id, page=page, block_id=f"line_{i}",
            text=t, bbox=(float(x0), float(y0), float(x2), float(y2)),
            source="img", extra={"confidence": float(conf)}
        ))
    return ExtractionResult("\n".join(lines), blocks)


# Minimal repro path
from ingest_text_oss import extract_text
ocr = EasyOCREngine()  # your class with the patched ocr_bytes
res = extract_text(html_str, "text/html", ocr=ocr, include_base64_images=True)
print(res.text[:500])
