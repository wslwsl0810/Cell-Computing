import cv2
import numpy as np
import matplotlib.pyplot as plt

# Helper function to display an image
def display_image(title, img, cmap='gray'):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

# Function to process the image and count cells
def analyze_cells(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image("Grayscale Image", gray)

    # Step 3: Thresholding to create a binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    display_image("Binary Image (Thresholding)", binary)

    # Step 4: Morphological operations to refine the binary image
    cell_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cell_kernel, iterations=1)
    display_image("Refined Binary Image (Morphological Operations)", processed_binary)

    # Step 5: Contour detection
    all_contours, _ = cv2.findContours(processed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw detected contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, all_contours, -1, (0, 255, 0), 1)
    display_image("Detected Contours (Cells)", cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB), cmap='viridis')

    # Output the total number of cells detected
    total_cells = len(all_contours)
    print(f"Total number of cells detected: {total_cells}")
    return total_cells

# Example usage
image_path = "./input/001.jpg"  # Replace with the path to your image
total_cells = analyze_cells(image_path)
