import fitz  # pymupdf
import os
import base64
from pathlib import Path

def extract_images_from_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            b64 = base64.b64encode(image_bytes).decode()
            images.append({
                "page": page_num + 1,
                "index": img_index + 1,
                "ext": image_ext,
                "b64": b64
            })

    doc.close()
    return images