import cv2

class FeatureMatcher:
    def match_features(self, features1, features2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(features1, features2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def calculate_similarity(self, matches):
        max_matches = 500
        similarity = (len(matches) / max_matches) * 100
        return min(similarity, 100)
