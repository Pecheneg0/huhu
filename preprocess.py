import cv2
import numpy as np

def enhance_contrast(image):
    """ Увеличение контрастности с помощью CLAHE. """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

image = cv2.imread("dataset_generated_1/Ա/0.png", cv2.IMREAD_GRAYSCALE)
enhanced_image = enhance_contrast(image)
cv2.imwrite("enhanced_image.png", enhanced_image)
