import cv2
import numpy as np
from src.webcam_constants import (
    WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY,
    FILTER_NONE_KEY, MENU_TEXT, MENU_POSITION, MENU_FONT,
    MENU_FONT_SCALE, MENU_FONT_THICKNESS, MENU_COLOR
)
from src.facial_landmark_detection import detect_facial_landmarks, get_yaw_angle_from_landmarks, draw_facial_landmarks

def open_webcam_with_filter_switching():
    """
    Open the webcam and switch filters based on detected facial landmarks and their orientation.
    """
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            break

        # Detect facial landmarks
        landmarks = detect_facial_landmarks(frame)

        # Draw landmarks on the frame
        draw_facial_landmarks(frame, landmarks)

        # Calculate the yaw angle if landmarks are detected
        yaw_angle = None
        if landmarks:
            yaw_angle = get_yaw_angle_from_landmarks(landmarks[0])  # Use the first detected face

        # Display the yaw angle on the frame
        if yaw_angle is not None:
            cv2.putText(frame, f"Yaw Angle: {yaw_angle:.2f}°", (10, 30), 
                        MENU_FONT, MENU_FONT_SCALE, MENU_COLOR, MENU_FONT_THICKNESS)

        # Display the menu on the webcam feed
        for i, line in enumerate(MENU_TEXT.split("\n")):
            cv2.putText(frame, line, (MENU_POSITION[0], MENU_POSITION[1] + i * 20),
                        MENU_FONT, MENU_FONT_SCALE, MENU_COLOR, MENU_FONT_THICKNESS)

        cv2.imshow(WINDOW_NAME, frame)

        # Check for exit key
        key = cv2.waitKey(FRAME_WAIT_KEY) & 0xFF
        if key == ord(EXIT_KEY):
            break

    # Release the webcam and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the webcam capture with filter switching
# if __name__ == "__main__":
open_webcam_with_filter_switching()
