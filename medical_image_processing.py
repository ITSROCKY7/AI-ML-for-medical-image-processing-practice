import cv2

# Load a grayscale medical image
image = cv2.imread('medical_image.jpg', cv2.IMREAD_GRAYSCALE) 

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Show results
cv2.imshow('Original', image)
cv2.imshow('Thresholded', thresh)
cv2.imshow('Contours', contour_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
