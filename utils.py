import cv2
import numpy as np
import base64
import io
import fitz
import requests
from PIL import Image
import google.generativeai as genai

# === Gemini Config ===
genai.configure(api_key="AIzaSyC1MFhWhGuf0Nxl5uT8eJYnClBxMWI78OA")

def download_image_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Image URL could not be downloaded.")
    img_bytes = io.BytesIO(response.content)
    return Image.open(img_bytes).convert("RGB")

def remove_shadow_opencv_from_image(pil_img):
    cv_img = np.array(pil_img)
    cv_img = cv_img[:, :, ::-1].copy()  
    rgb_planes = cv2.split(cv_img)
    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result_planes.append(norm)
    result = cv2.merge(result_planes)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def convert_pdf_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def pil_to_bytes(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return buffer.getvalue()

def extract_text_with_gemini(image_bytes):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        contents=[{
            "role": "user",
            "parts": [
                {"text": "Extract the text exactly as shown in the image without skipping."},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
            ]
        }]
    )
    return response.text.strip()
