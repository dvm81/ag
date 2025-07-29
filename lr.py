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
