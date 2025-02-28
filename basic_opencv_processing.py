import cv2
import numpy as np

# Load an image
image = cv2.imread('mri_scan.png', cv2.IMREAD_UNCHANGED)  # Replace with your image

# Convert to grayscale
color = cv2.cvtColor(image, cv2.GRAY2COLOR_BGR)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# Canny Edge Detection
edges = cv2.Canny(blurred, 50, 150)

# Show results
cv2.imshow('Original', image)
cv2.imshow('Grayscale', color)
cv2.imshow('Blurred', blurred)
cv2.imshow('Edges', edges)
cv2.imshow('Enhanced Image', enhanced_image)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced_image = clahe.apply(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
