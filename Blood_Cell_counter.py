# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define dataset path (CHANGE THIS TO YOUR ACTUAL FOLDER)
dataset_path = "/content/drive/My Drive/JPEGImages/"

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    print("❌ Error: Dataset path does not exist. Check your Google Drive path!")
else:
    print("✅ Dataset path found.")
# List all image files in the dataset folder
image_files = [f for f in os.listdir(dataset_path) if f.endswith((".jpg", ".png"))]

# Ensure at least one image is found
if len(image_files) == 0:
    print("❌ Error: No images found in the dataset folder!")
else:
    print(f"✅ Total Images Found: {len(image_files)}")

    # Load the first image
    image_path = os.path.join(dataset_path, image_files[0])
    image = cv2.imread(image_path)

    # Ensure the image is loaded correctly
    if image is None:
        print("❌ Error: Image not loaded. Check the file format and path!")
    else:
        print("✅ Image loaded successfully.")

        # Convert to RGB for displaying
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display Original Image
        plt.figure(figsize=(6,6))
        plt.imshow(image)
        plt.title("Original Blood Smear Image")
        plt.axis("off")
        plt.show()
# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply Otsu’s Thresholding
_, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Display Binary Image
plt.figure(figsize=(6,6))
plt.imshow(binary_img, cmap="gray")
plt.title("Binary Image (Thresholded)")
plt.axis("off")
plt.show()
# Apply Morphological Opening to Remove Noise
kernel = np.ones((2,2), np.uint8)  # Smaller kernel
cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)

# Display Cleaned Image
plt.figure(figsize=(6,6))
plt.imshow(cleaned_img, cmap="gray")
plt.title("Cleaned Image (After Morphological Processing)")
plt.axis("off")
plt.show()
# Find contours
contours, _ = cv2.findContours(cleaned_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw contours
output_image = image.copy()

# Set minimum area for blood cells (adjust as needed)
min_area = 50

# Count detected cells
cell_count = 0

for contour in contours:
    if cv2.contourArea(contour) > min_area:  # Filter small unwanted noise
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cell_count += 1

print(f"✅ Total Blood Cells Detected: {cell_count}")
# Convert the output image back to RGB for displaying
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Display the final image with detected cells
plt.figure(figsize=(6,6))
plt.imshow(output_image_rgb)
plt.title(f"Detected Blood Cells: {cell_count}")
plt.axis("off")
plt.show()

# Save the output image
output_path = os.path.join(dataset_path, "blood_cells_detected.png")
cv2.imwrite(output_path, output_image)
print(f"✅ Processed image saved at: {output_path}")
