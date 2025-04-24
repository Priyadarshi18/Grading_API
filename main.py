from fastapi import FastAPI, HTTPException, Query, Form, Body
from fastapi.responses import JSONResponse
from typing import Optional
from utils import (
    remove_shadow_opencv_from_image,
    download_image_from_url,
    pil_to_bytes,
    extract_text_with_gemini
)
from PIL import Image
import io

app = FastAPI()

@app.get("/extract-text")
async def extract_text(image_url: str = Query(..., description="Public URL of the image")):
    try:
        pil_img = download_image_from_url(image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download or open image: {e}")

    try:
        shadow_free_img = remove_shadow_opencv_from_image(pil_img)
        image_bytes = pil_to_bytes(shadow_free_img)
        text = extract_text_with_gemini(image_bytes)
        return JSONResponse(content={"extracted_text": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during text extraction: {e}")

class ImageURLRequest(BaseModel):
    image_url: str

@app.post("/extract-text")
async def extract_text_post(
    image_url: Optional[str] = Body(None),
    image_url_form: Optional[str] = Form(None)
):
    try:
        # Prioritize JSON body, fallback to form data
        final_url = image_url or image_url_form
        if not final_url:
            raise HTTPException(status_code=400, detail="No image_url provided.")

        pil_img = download_image_from_url(final_url)
        shadow_free_img = remove_shadow_opencv_from_image(pil_img)
        image_bytes = pil_to_bytes(shadow_free_img)
        text = extract_text_with_gemini(image_bytes)
        return JSONResponse(content={"extracted_text": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
