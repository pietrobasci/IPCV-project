import cv2
import numpy as np


class PeopleDetector:

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=5, detectShadows=True)
        self.kernelOp = np.ones((5, 5), np.uint8)
        self.kernelCl = np.ones((5, 5), np.uint8)

    def detect(self, frame):
        fgmask = self.fgbg.apply(frame)
        # Threshold operation to obtain the binary mask
        ret, imBin = cv2.threshold(fgmask, 230, 255, cv2.THRESH_BINARY)
        # Opening (erode->dilate) to remove white spots
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, self.kernelOp)
        # Closing (dilate -> erode) to remove black spots inside the object
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernelCl)
        # Find contours of the binary image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi = []
        for cnt in contours:
            # Compute the contour area
            area = cv2.contourArea(cnt)
            # Filter out object with an area below a given threshold
            if area > 120*80:
                # Find the bounding rectangle of each object found
                x, y, w, h = cv2.boundingRect(cnt)
                roi.append([x, y, w, h])

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return mask, roi
