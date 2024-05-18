import cv2
import numpy as np

def detect_colors_from_camera(lower_bounds, upper_bounds):
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture each frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Initialize a combined mask
        combined_mask = np.zeros_like(hsv_frame[:, :, 0])

        # Loop through each color range and create a mask
        for lower_color, upper_color in zip(lower_bounds, upper_bounds):
            lower_bound = np.array(lower_color)
            upper_bound = np.array(upper_color)
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Apply the combined mask to the original frame
        result = cv2.bitwise_and(frame, frame, mask=combined_mask)

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Color Detected', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lower_red = [0, 100, 100]
    upper_red = [10, 255, 255]
    lower_yellow = [20, 100, 100]
    upper_yellow = [30, 255, 255]

    lower_bounds = [lower_red, lower_yellow]
    upper_bounds = [upper_red, upper_yellow]

    detect_colors_from_camera(lower_bounds, upper_bounds)
