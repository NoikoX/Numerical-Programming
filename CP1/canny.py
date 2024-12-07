import cv2
import numpy as np


def Grayscale(image):
    # here I just convert the image to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def GaussianBlur(image):
    # gaussian blur to reduce the noise in it
    return cv2.GaussianBlur(image, (5, 5), 1.4)


def SobelFilter(image):
    # computing gradients using sobel filter which we've already seen
    image = Grayscale(GaussianBlur(image))
    G_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    G_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.clip((convolved / convolved.max()) * 255, 0, 255).astype('uint8')

    angles = np.rad2deg(np.arctan2(G_y, G_x)) % 180
    return convolved, angles


def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size, dtype=np.uint8)

    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            direction = angles[i, j]
            neighbors = []

            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                neighbors = [image[i, j - 1], image[i, j + 1]]
            elif 22.5 <= direction < 67.5:
                neighbors = [image[i - 1, j + 1], image[i + 1, j - 1]]
            elif 67.5 <= direction < 112.5:
                neighbors = [image[i - 1, j], image[i + 1, j]]
            elif 112.5 <= direction < 157.5:
                neighbors = [image[i - 1, j - 1], image[i + 1, j + 1]]

            if image[i, j] >= max(neighbors):
                suppressed[i, j] = image[i, j]

    return suppressed


def double_threshold_hysteresis(image, low, high):
    weak, strong = 50, 255
    size = image.shape
    result = np.zeros(size, dtype=np.uint8)

    strong_coords = np.argwhere(image >= high)
    weak_coords = np.argwhere((image >= low) & (image < high))

    result[tuple(strong_coords.T)] = strong
    result[tuple(weak_coords.T)] = weak

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    while len(strong_coords):
        x, y = strong_coords[0]
        strong_coords = np.delete(strong_coords, 0, axis=0)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size[0] and 0 <= ny < size[1] and result[nx, ny] == weak:
                result[nx, ny] = strong
                strong_coords = np.vstack([strong_coords, [nx, ny]])

    result[result != strong] = 0
    return result


def Canny(image, low, high):
    # some of the code here is directly from GitHub
    # thanks to my friend "StefanPitur :)"
    image, angles = SobelFilter(image)
    image = non_maximum_suppression(image, angles)
    return double_threshold_hysteresis(image, low, high)
