import cv2
import numpy as np
import time

cap = cv2.VideoCapture('betterCpCars.mp4')

# getting the frame rate(which i already know is 30 but still...)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_rate = int(fps)
time_per_frame = 1 / fps

previous_positions = {'red': None, 'blue': None}
speeds = {'red': 0, 'blue': 0}
red_speeds = []
blue_speeds = []
speed_smooth_window = 10

# color ranges for car detection which are in HSV hue saturation and value as i will mention in the project pdf..
red_lower = np.array([0, 120, 70])
red_upper = np.array([10, 255, 255])
blue_lower = np.array([90, 150, 0])
blue_upper = np.array([130, 255, 255])


def simple_edge_detection(img, threshold1=50, threshold2=150):

    # justt calculating gradients in x and y directions
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # ssimple thresholding which we could use as it was mentioned in the cp desc
    _, edges = cv2.threshold(grad_mag, threshold1, threshold2, cv2.THRESH_BINARY)

    return edges


def simple_clustering(points, k=2, max_iters=100):
    # number of centroids should bee 2 cuz im working with just two cars and no more, just red and blue one
    if len(points) < k:
        return np.arange(len(points))
    # here it the simple logic what is behind the k means..done it 120310239times..
    centroids = np.array(points)[np.random.choice(len(points), k, replace=False)]

    assignments = np.zeros(len(points), dtype=int)

    for _ in range(max_iters):
        # here i assign each point to the nearest centroid
        for i, point in enumerate(points):
            distances = np.linalg.norm(centroids - point, axis=1)
            assignments[i] = np.argmin(distances)

        # here i update centroids
        new_centroids = np.array([np.mean(points[assignments == j], axis=0) for j in range(k)])

        # just a checking for convergence cuz why not..
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return assignments



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # firslty here goes the detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = simple_edge_detection(gray)

    # then color detection (HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # finding contours where the pixels are white in the red & blue masks
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours += cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    current_positions = {'red': None, 'blue': None}
    edge_points = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500: # here i check if the area is > 500 so that I avoid contouring the smaller objects than the cars
            x, y, w, h = cv2.boundingRect(contour)

            # here i identify car color and store position
            if cv2.countNonZero(red_mask[y:y + h, x:x + w]) > 0:
                color = 'red'
                current_positions['red'] = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # this is just the red bounding box
            else:
                color = 'blue'
                current_positions['blue'] = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # same but the blue :D


            for point in contour:
                x, y = point[0]
                if edges[y, x] == 255:
                    edge_points.append((x, y))


    if len(edge_points) >= 2:
        cluster_assignments = simple_clustering(edge_points, k=2)
    # and then i can use cluster_assignments to further analyze or group the edges somehow

    # here i calculate and display speed
    for color in ['red', 'blue']:
        if current_positions[color] and previous_positions[color]:
            distance = np.linalg.norm(np.array(current_positions[color]) - np.array(previous_positions[color]))
            speed = distance / time_per_frame
            speeds[color] = speed

            if color == 'red':
                red_speeds.append(speeds[color])
                if len(red_speeds) > speed_smooth_window:
                    red_speeds.pop(0)
                smoothed_red_speed = np.mean(red_speeds)
                cv2.putText(frame, f"Red speed: {smoothed_red_speed:.1f} px/s",
                            (10, 60),  # Adjusted y-coordinate
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif color == 'blue':
                blue_speeds.append(speeds[color])
                if len(blue_speeds) > speed_smooth_window:
                    blue_speeds.pop(0)
                smoothed_blue_speed = np.mean(blue_speeds)
                cv2.putText(frame, f"Blue speed: {smoothed_blue_speed:.1f} px/s",
                            (10, 90),  # Adjusted y-coordinate
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        previous_positions[color] = current_positions[color]

    cv2.imshow('Speed detection of red and blue cars(im the blue one)', frame)


    time_taken = time.time() - start_time
    time_to_wait = max(1, int(1000 / frame_rate - time_taken * 1000))
    cv2.waitKey(time_to_wait)

    # i can break the loop if i press the q on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()