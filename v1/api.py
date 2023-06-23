import cv2
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from skimage.metrics import structural_similarity as ssim

app = FastAPI()

def resize_with_aspect_ratio(image, max_width, max_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)
    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def compare_images(image1, image2):

    # Resize images maintaining aspect ratio
    max_width, max_height = 800, 800
    image1 = resize_with_aspect_ratio(image1, max_width, max_height)
    image2 = resize_with_aspect_ratio(image2, max_width, max_height)

    # Pad the smaller image with black pixels
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    max_height, max_width = max(height1, height2), max(width1, width2)

    padded1 = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    padded1[:height1, :width1, :] = image1

    padded2 = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    padded2[:height2, :width2, :] = image2

    # Rest of the compare_images function...
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    max_keypoints = max(len(kp1), len(kp2))
    similarity_orb = (len(matches) / max_keypoints)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(padded1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(padded2, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detector
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)

    # Compute SSIM between edge images
    similarity_ssim = ssim(edges1, edges2, win_size=None)

    # Combine both similarities
    similarity = (similarity_orb + similarity_ssim) / 2 * 100

    return similarity

@app.get("/")
def root():
    return {"api": "v1"}

@app.post("/compare_images/")
async def compare_images_endpoint(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        image1_content = await image1.read()
        image2_content = await image2.read()

        image1 = Image.open(io.BytesIO(image1_content))
        image2 = Image.open(io.BytesIO(image2_content))

        image1 = np.array(image1)
        image2 = np.array(image2)

        similarity = compare_images(image1, image2)

        return {"similarity": similarity}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)