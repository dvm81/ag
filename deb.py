#replace decoder

import re, base64

def decode_data_url_to_bytes(src: str) -> bytes | None:
    if not src.lower().startswith("data:image/"):
        return None
    try:
        header, b64 = src.split(",", 1)
    except ValueError:
        return None
    # remove whitespace/newlines that are present in your HTML
    b64 = re.sub(r"\s+", "", b64)
    try:
        return base64.b64decode(b64, validate=False)
    except Exception:
        return None


# inside extract_text_from_html(...)
for img in soup.find_all("img"):
    src = img.get("src") or ""
    img_bytes = decode_data_url_to_bytes(src)
    if not img_bytes:
        continue
    res = ocr.ocr_bytes(img_bytes, doc_id=doc_id, page=1)
    ...



#2) Make EasyOCREngine.ocr_bytes robust to decoding

import io, os, tempfile
import numpy as np
import cv2

def ocr_bytes(self, b: bytes, doc_id="img", page=1):
    # ensure real bytes
    if not isinstance(b, (bytes, bytearray)):
        b = bytes(b)

    img = None
    try:
        arr = np.frombuffer(b, dtype=np.uint8)     # 1-D uint8 buffer
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # -> np.ndarray (BGR)
    except Exception:
        img = None

    try:
        if img is None or not hasattr(img, "shape"):
            # Fallback: write a temp file and give EasyOCR a path
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
    except Exception as e:
        # last resort: return empty to avoid crashing the whole pipeline
        return ExtractionResult(text="", blocks=[])

    lines, blocks = [], []
    for i, (bbox, txt, conf) in enumerate(result):
        t = (txt or "").strip()
        if not t: 
            continue
        (x0,y0), (_, _), (x2,y2), _ = bbox
        lines.append(t)
        blocks.append(TextBlock(
            doc_id=doc_id, page=page, block_id=f"line_{i}",
            text=t, bbox=(float(x0), float(y0), float(x2), float(y2)),
            source="img", extra={"confidence": float(conf)}
        ))
    return ExtractionResult("\n".join(lines), blocks)


#python3 -m pip install opencv-python-headless easyocr pillow
