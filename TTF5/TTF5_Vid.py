import cv2

cap = cv2.VideoCapture("HomeOnTheRange.mp4")

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

i = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading the frame")
        break

    cv2.imwrite(f'Images/{i}_frame.jpg', frame)
    i += 1

cap.release()
print(f'Frames saved: {i}')


