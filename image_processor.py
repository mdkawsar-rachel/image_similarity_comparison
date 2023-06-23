import cv2
import numpy as np

class ImageProcessor:
    def preprocess(self, image):
        image = cv2.resize(image, (300, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def extract_features(self, image):
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return descriptors
