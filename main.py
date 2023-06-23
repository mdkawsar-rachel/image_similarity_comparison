from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
from image_processor import ImageProcessor
from feature_matcher import FeatureMatcher

app = FastAPI()

image_processor = ImageProcessor()
feature_matcher = FeatureMatcher()

@app.post("/image_similarity")
async def image_similarity(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        img1 = cv2.imdecode(np.frombuffer(image1.file.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(image2.file.read(), np.uint8), cv2.IMREAD_COLOR)

        preprocessed_img1 = image_processor.preprocess(img1)
        preprocessed_img2 = image_processor.preprocess(img2)

        features1 = image_processor.extract_features(preprocessed_img1)
        features2 = image_processor.extract_features(preprocessed_img2)

        matches = feature_matcher.match_features(features1, features2)
        similarity = feature_matcher.calculate_similarity(matches)

        return JSONResponse(content={"similarity": similarity})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
