import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
image = cv2.imread("Images/cat1.jpg", cv2.IMREAD_GRAYSCALE)

# Apply filters
gaussian_blur = cv2.GaussianBlur(image, (3, 3), 0)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_both = cv2.magnitude(sobel_x, sobel_y)
scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
scharr_both = cv2.magnitude(scharr_x, scharr_y)
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Plotting
plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Gaussian Blur
plt.subplot(3, 3, 2)
plt.imshow(gaussian_blur, cmap='gray')
plt.title("Gaussian Blur")
plt.axis('off')

# Sobel X
plt.subplot(3, 3, 3)
plt.imshow(sobel_x, cmap='gray')
plt.title("Sobel X")
plt.axis('off')

# Sobel Y
plt.subplot(3, 3, 4)
plt.imshow(sobel_y, cmap='gray')
plt.title("Sobel Y")
plt.axis('off')

# Sobel Combined
plt.subplot(3, 3, 5)
plt.imshow(sobel_both, cmap='gray')
plt.title("Sobel Combined")
plt.axis('off')

# Scharr X
plt.subplot(3, 3, 6)
plt.imshow(scharr_x, cmap='gray')
plt.title("Scharr X")
plt.axis('off')

# Scharr Y
plt.subplot(3, 3, 7)
plt.imshow(scharr_y, cmap='gray')
plt.title("Scharr Y")
plt.axis('off')

# Scharr Combined
plt.subplot(3, 3, 8)
plt.imshow(scharr_both, cmap='gray')
plt.title("Scharr Combined")
plt.axis('off')

# Laplacian
plt.subplot(3, 3, 9)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian")
plt.axis('off')

plt.tight_layout()
plt.show()
