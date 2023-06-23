import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from skimage.metrics import structural_similarity as compare_ssim
from io import BytesIO
from PIL import Image

app = FastAPI()

def compare_images(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((img1_gray - img2_gray) ** 2)
    if mse == 0:
        return 1.0  # Images are identical

    # Compute Structural Similarity Index (SSIM)
    score, _ = compare_ssim(img1_gray, img2_gray, full=True)
    return score

def read_imagefile(file) -> Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/compare-images/")
async def compare_image_files(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    if image1.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file format for image1. Only JPEG and PNG are allowed.")

    if image2.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file format for image2. Only JPEG and PNG are allowed.")

    try:
        img1 = cv2.cvtColor(np.array(read_imagefile(await image1.read())), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(read_imagefile(await image2.read())), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading images: {e}")

    # Check if the dimensions of the two images are different
    if img1.shape[:2] != img2.shape[:2]:
        # Resize the second image to the dimensions of the first image
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    else:
        img2_resized = img2

    similarity_score = compare_images(img1, img2_resized)
    percentage_score = round(similarity_score * 100, 2)  # Round to 2 decimal places

    return JSONResponse(content={"similarity_score": percentage_score})