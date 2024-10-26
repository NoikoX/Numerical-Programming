import cv2

image = cv2.imread("Images/cat1.jpg", cv2.IMREAD_GRAYSCALE)

print(image)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

